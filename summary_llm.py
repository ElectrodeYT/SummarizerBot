import discord
import os
from datetime import datetime
import cache

from openai import AsyncOpenAI
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

scaleway_token = os.environ['SCW_TOKEN']

ai_client = AsyncOpenAI(
    base_url="https://api.scaleway.ai/v1",
    api_key=scaleway_token
)


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
    return cache.CachedDiscordAuthor(name=author.name, id=author.id)


def discord_messages_to_cached_messages(discord_messages: [discord.Message]):
    messages = []
    for discord_message in discord_messages:
        messages.append(cache.CachedDiscordMessage(content=discord_message.content, id=discord_message.id,
                                                   author=discord_author_to_cached_author(discord_message.author),
                                                   channel_id=discord_message.channel.id))

    return messages


async def get_messages(channel: discord.TextChannel, limit=50, before=None, after=None):
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
            cached_messages = await cache.fetch_from_cache(limit, before)
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
    await cache.commit_messages_to_cache(messages, channel)
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
