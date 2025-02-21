import io
import os
import sqlite3
from datetime import datetime
from pprint import pprint


import discord
import numpy as np
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from discord import app_commands

scaleway_token = os.environ['SCW_TOKEN']
discord_token = os.environ['DISCORD_TOKEN']
default_summary_prompt = 'Summarize the conversation(s). If there are several conversations, summarize them individually.'

ai_client = AsyncOpenAI(
    base_url="https://api.scaleway.ai/v1",
    api_key=scaleway_token
)

os.makedirs('data', exist_ok=True)

db_con = sqlite3.connect('data/cache.db')
db_con.execute('CREATE TABLE IF NOT EXISTS authors(id INT PRIMARY KEY, name TEXT)')
db_con.execute('CREATE TABLE IF NOT EXISTS messages(id PRIMARY KEY, content TEXT, author_id INT, channel_id INT,'
               'previous_message_id INT, next_message_id INT)')
db_con.commit()

class DiscordClient(discord.Client):
    async def on_ready(self):
        print(f"Logged in as {self.user.name} (ID: {self.user.id})")

    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def on_guild_available(self, guild: discord.Guild):
        print(f"Setting up hook for {guild.name} (ID: {guild.id})")
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)

class CachedDiscordAuthor(discord.Object):
    name: str
    id: int
    def __init__(self, name: str, id: int):
        self.name = name
        self.id = id

class CachedDiscordMessage(discord.Object):
    content: str
    id: int
    before_id: int
    after_id: int
    author: CachedDiscordAuthor
    channel_id: int
    def __init__(self, content: str = '', id: int | None = None, before: int | None = None, after: int | None = None,
                 author: CachedDiscordAuthor | None = None, channel_id: int | None = None):
        self.content = content
        self.id = id
        self.before_id = before
        self.after_id = after
        self.author = author
        self.channel_id = channel_id

intents = discord.Intents.default()
intents.message_content = True
client = DiscordClient(intents=intents)

def format_message_list(messages: [discord.Message]):
    formatted_messages = []

    for discord_message in messages:
        formatted_messages.append(f'{discord_message.author.name}: {discord_message.content}')

    pprint(formatted_messages)

    return formatted_messages

def format_message_list_for_embeddings(messages: [discord.Message]):
    formatted_messages = []

    for discord_message in messages:
        formatted_messages.append(f'{discord_message.content}')

    return formatted_messages

def discord_author_to_cached_author(author: discord.User):
    return CachedDiscordAuthor(name=author.name, id=author.id)

def discord_messages_to_cached_messages(discord_messages: [discord.Message]):
    messages = []
    for discord_message in discord_messages:
        messages.append(CachedDiscordMessage(content=discord_message.content, id=discord_message.id,
                                             author=discord_author_to_cached_author(discord_message.author),
                                             channel_id=discord_message.channel.id))

    return messages


# For this, we expect the messages to be in oldest -> newest order.
async def commit_messages_to_cache(messages: [CachedDiscordMessage], channel: discord.TextChannel):
    cur = db_con.cursor()
    for idx, message in enumerate(messages):
        # Set known prev/after message IDs
        if idx != 0:
            message.before_id = messages[idx - 1].id

        if idx != len(messages) - 1:
            message.after_id = messages[idx + 1].id

        # If prev or after are None, check if the message is in the database, and check if we have that saved there
        if message.before_id is None:
            cur.execute('SELECT previous_message_id FROM messages WHERE id = ?', (message.id,))
            result = cur.fetchone()
            print(f'before: {result} in cache for message {message.id} ({message.content})')
            if result is not None and result[0] is not None:
                message.before_id = int(result[0])

        if message.after_id is None:
            cur.execute('SELECT next_message_id FROM messages WHERE id = ?', (message.id,))
            result = cur.fetchone()
            print(f'after: {result} in cache for message {message.id} ({message.content})')
            # The result may be none if none was put in
            if result is not None and result[0] is not None:
                message.after_id = int(result[0])

        # Commit message and author (TODO: optimize this) to SQL cache
        cur.execute('INSERT OR REPLACE INTO messages(id, content, author_id, channel_id, previous_message_id,'
                    'next_message_id) VALUES (?, ?, ?, ?, ?, ?)', (message.id, message.content, message.author.id,
                                                                   message.channel_id, message.before_id, message.after_id))

        cur.execute('INSERT OR REPLACE INTO authors(id, name) VALUES (?, ?)', (message.author.id,
                                                                               message.author.name))
    cur.close()
    db_con.commit()

async def fetch_author_from_cache(author_id: int):
    cur = db_con.cursor()
    cur.execute('SELECT name FROM authors WHERE id = ?', (author_id,))
    result = cur.fetchone()
    cur.close()
    if result is not None:
        return CachedDiscordAuthor(id=author_id, name=result[0])
    else:
        return None

async def fetch_from_cache(limit: int, before: CachedDiscordMessage | discord.Message | None = None):
    cur = db_con.cursor()
    cached_messages = []

    assert before is not None
    while limit > 0:
        # Check if we can get the message
        cur.execute('SELECT id, content, author_id, channel_id, previous_message_id, next_message_id'
                    ' FROM messages WHERE next_message_id = ?', (before.id,))
        result = cur.fetchone()

        if result is not None:
            # We have a cached result, add it to the cached messages list
            # Try to get the author
            message_id, content, author_id, channel_id, previous_message_id, next_message_id = result
            author = await fetch_author_from_cache(author_id)
            if author is None:
                # We can't fetch the author, break
                break

            cached_message = CachedDiscordMessage(id=message_id, content=content, author=author,
                                                  channel_id=channel_id, before=previous_message_id,
                                                  after=next_message_id)
            cached_messages.append(cached_message)
            before = cached_message
            limit -= 1
        else:
            # We have reached our limit
            break

    if len(cached_messages):
        print(f'Fetched {len(cached_messages)} messages from cache')

    return cached_messages


async def get_messages(channel: discord.TextChannel, limit = 50, before = None, after = None):
    messages = []

    while limit > 0:
        # TODO: handle after being true
        assert after is None

        # If before or after are both none, no requests have been made
        # Therefore, request the first batch of messages
        # Whenever calling to discord API, we max the limit to 50, as that is the maximum the discord API itself supports
        if before is None and after is None:
            discord_messages = [message async for message in channel.history(limit=min(limit, 50))]
            before = discord_messages[-1]
        elif before is not None:
            # Check to see if we have a cache hit, and if we do, pull as many messages from there as we can (up to limit)
            cached_messages = await fetch_from_cache(limit, before)
            if len(cached_messages):
                messages.extend(cached_messages)
                limit -= len(cached_messages)
                before = cached_messages[-1]
                continue
            else:
                # Messages not in cache, fetch them from discord
                discord_messages = [message async for message in channel.history(limit=min(limit, 50),
                                                                                 before=before)]
                before = discord_messages[-1]

        # If we got here, we need to convert the discord messages to cached messages
        messages.extend(discord_messages_to_cached_messages(discord_messages))
        limit -= min(limit, 50)

    if before is not None:
        # We fetched from newest, reverse array
        messages = messages[::-1]

    print(f'commiting {len(messages)} messages to cache')
    await commit_messages_to_cache(messages, channel)
    return messages

async def run_llm(interaction: discord.Interaction, llm_messages: [], embed: discord.Embed):
    model = 'deepseek-r1-distill-llama-70b'
    temperature = 0.7
    max_tokens = 4096

    completion = await ai_client.chat.completions.create(
        model=model,
        messages=llm_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
    )

    # Add llm info to the embed
    embed.add_field(name='LLM Info', value=f'Model: {model}, Max Tokens: {max_tokens}, '
                                           f'Temperature: {temperature}')

    # Don't edit too much
    last_edit = None
    summary = ''

    async for chunk in completion:
        if len(chunk.choices) == 0:
            break

        summary += chunk.choices[0].delta.content

        if len(summary) > 4096:
            embed.description = (f'Response is getting too long, please wait for it to be done...\n'
                                 f'(currently {len(summary)} chars)')
        else:
            embed.description = summary

        if last_edit is None or (datetime.now() - last_edit).total_seconds() > 2:
            await interaction.edit_original_response(embed=embed, content='')
            last_edit = datetime.now()

    embed.title = 'Summary'

    if len(summary) >= 4096:
        f = io.StringIO(summary)
        await interaction.edit_original_response(embed=embed, file=discord.File(fp=f, filename='generated.txt'))
    else:
        await interaction.edit_original_response(embed=embed)

async def run_embeddings(interaction: discord.Interaction, base_sentence: str, messages: []):
    model = 'bge-multilingual-gemma2'

    message_embeddings_response = await ai_client.embeddings.create(
        input=messages,
        model=model,
    )

    question_embeddings_response = await ai_client.embeddings.create(
        input=base_sentence,
        model=model,
    )

    # Convert the embedding responses into float arrays
    message_embeddings = []
    question_embeddings = question_embeddings_response.data[0].embedding
    for embedding_response in message_embeddings_response.data:
        message_embeddings.append(embedding_response.embedding)

    # Compare embeddings
    best_matches = cosine_similarity(np.array(message_embeddings), np.array(question_embeddings).reshape(1, -1))

    for message, match_percentage in zip(messages, best_matches):
        print(f'{message} -> {match_percentage * 100}%')

    return question_embeddings, message_embeddings


async def create_summary(interaction: discord.Interaction, discord_messages: [], summarize_prompt: str,
                         footer_text: str):
    llm_messages = [{'role': 'system',
                     'content': f'You should summarize the following messages according to this prompt: '
                                f'"{summarize_prompt}". Use only information mentioned in the following messages.'
                     }]

    for message in format_message_list(discord_messages):
        llm_messages.append({'role': 'user', 'content': message})

    # Stream the summary
    embed = discord.Embed(title='Generating summary')
    embed.set_footer(text=footer_text)

    await run_llm(interaction, llm_messages, embed)

async def create_topic_summary(interaction: discord.Interaction, discord_messages: [], topic: str, footer_text: str):
    llm_messages = [{'role': 'system',
                     'content': f'You are given the name of a concept first and then a series of chat messages from a'
                                f'chat room about the Managarm operating system. Based on these messages, explain how'
                                f'the given concept works in technical terms. If possible, the explanation should be'
                                f'suitable to document the concept in a manual. Ignore messages that do not relate to'
                                f'the concept.'
                     }, {'role': 'user',
                         'content': f'Concept: {topic}'}]

    formatted_message_list = format_message_list(discord_messages)

    await run_embeddings(interaction, topic, format_message_list_for_embeddings(discord_messages))

    for message in formatted_message_list:
        llm_messages.append({'role': 'user', 'content': message})

    # Stream the summary
    embed = discord.Embed(title='Generating summary')
    embed.set_footer(text=footer_text)

    await run_llm(interaction, llm_messages, embed)


@client.tree.command()
async def summarize(interaction: discord.Interaction, count_msgs: int = None, channel: discord.TextChannel = None,
                    summarize_prompt: str = default_summary_prompt) -> None:
    await interaction.response.send_message('Doing stuff, might take a (long) while...')

    try:
        if channel is None:
            channel = interaction.channel

        await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (compiling messages)')

        # Get the messages, then invert the array
        discord_messages = await get_messages(channel=channel, limit=count_msgs)
        footer_text = f'Summary of the last {len(discord_messages)} messages in {channel.name} | Summary prompt: {summarize_prompt}'

        await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (Firing up AI)')
        await create_summary(interaction, discord_messages, summarize_prompt, footer_text)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')


@client.tree.command(description='Summarize for topic')
async def summarize_topic(interaction: discord.Interaction, topic: str, count_msgs: int = None,
                          channel: discord.TextChannel = None) -> None:
    await interaction.response.send_message('Doing stuff, might take a (long) while...')

    try:
        if channel is None:
            channel = interaction.channel

        await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (compiling messages)')

        # Get the messages, then invert the array
        discord_messages = await get_messages(channel=channel, limit=count_msgs)

        footer_text = f'Summary of the last {len(discord_messages)} messages in {channel.name} | Topic: {topic}'

        await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (Firing up AI)')
        await create_topic_summary(interaction, discord_messages, topic, footer_text)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')

def main() -> None:
    client.run(discord_token)

if __name__ == '__main__':
    main()
