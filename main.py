import os
from openai import AsyncOpenAI
from pprint import pprint

import discord
from discord import app_commands

import summary_llm
import cache

discord_token = os.environ['DISCORD_TOKEN']
default_summary_prompt = 'Summarize the conversation(s). If there are several conversations, summarize them individually.'

scaleway_token = os.environ['SCW_TOKEN']

ai_client = AsyncOpenAI(
    base_url="https://api.scaleway.ai/v1",
    api_key=scaleway_token
)


class DiscordClient(discord.Client):
    async def on_ready(self):
        print(f"Logged in as {self.user.name} (ID: {self.user.id})")
        await self.tree.sync()

    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)
        self.setup_guilds = []

    async def setup_guild_stuff(self, guild: discord.Guild):
        if guild.id in self.setup_guilds:
            return

        print(f"Setting up hook for {guild.name} (ID: {guild.id})")
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)

        self.setup_guilds.append(guild.id)

    async def on_guild_available(self, guild: discord.Guild):
        await self.setup_guild_stuff(guild)

    async def on_guild_join(self, guild: discord.Guild):
        await self.setup_guild_stuff(guild)

    async def on_message(self, message: discord.Message) -> None:
        # Ignore messages without a channel (safety) or from webhooks
        try:
            if message.channel is None:
                return
        except Exception:
            return

        # Best-effort: commit message to cache. Do not block on failures.
        try:
            await cache.commit_single_message_to_cache(message)
        except Exception:
            # swallow exceptions from caching to avoid disrupting bot operation
            return


intents = discord.Intents.default()
intents.message_content = True
client = DiscordClient(intents=intents)


@client.tree.command()
async def summarize(interaction: discord.Interaction, count_msgs: int = 1500, channel: discord.TextChannel = None,
                    summarize_prompt: str = default_summary_prompt) -> None:
    await interaction.response.send_message('Doing stuff, might take a (long) while...')

    try:
        if channel is None:
            channel = interaction.channel

        await interaction.edit_original_response(
            content='Doing stuff, might take a (long) while... (compiling messages)')

        # Get the messages, then invert the array
        discord_messages = await summary_llm.get_messages(channel=channel, limit=count_msgs)
        footer_text = f'Summary of the last {len(discord_messages)} messages in {channel.name} | Summary prompt: {summarize_prompt}'

        await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (Firing up AI)')
        await summary_llm.create_summary(interaction, discord_messages, summarize_prompt, footer_text)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.command(description='Summarize for topic')
async def summarize_topic(interaction: discord.Interaction, topic: str, count_msgs: int = 1500,
                          channel: discord.TextChannel = None) -> None:
    await interaction.response.send_message('Doing stuff, might take a (long) while...')

    try:
        if channel is None:
            channel = interaction.channel

        await interaction.edit_original_response(
            content='Doing stuff, might take a (long) while... (compiling messages)')

        # Get the messages, then invert the array
        discord_messages = await summary_llm.get_messages(channel=channel, limit=count_msgs)

        footer_text = f'Summary of the last {len(discord_messages)} messages in {channel.name} | Topic: {topic}'

        await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (Firing up AI)')
        await summary_llm.create_topic_summary(interaction, discord_messages, topic, footer_text)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.command(description='Pre-fetch older messages and reconcile channel links in the DB')
async def reconcile_messages(interaction: discord.Interaction, count_msgs: int = 1000,
                             channel: discord.TextChannel = None) -> None:
    await interaction.response.send_message('Starting reconciliation, this may take a while...')

    try:
        if channel is None:
            channel = interaction.channel

        # We'll update the original response periodically to show progress
        last_reported = {'processed': 0}

        async def progress_cb(processed, total, type="Fetched"):
            # Only edit when notable progress is made
            if processed - last_reported['processed'] >= 25:
                try:
                    await interaction.edit_original_response(content=f'{type} {processed}/{total} messages...')
                    last_reported['processed'] = processed
                except Exception:
                    pass

        processed = await cache.reconcile_channel_links(channel, limit=count_msgs, progress_callback=progress_cb)

        await interaction.edit_original_response(content=f'Reconciled {processed} messages for channel {channel.name}')
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise

#TODO
#@client.tree.command(description="Find a message by using a sentence as a search query. (Only searches cached messages)")

async def find_message(interaction: discord.Interaction, query: str, channel: discord.TextChannel = None) -> None:
    await interaction.response.send_message('Searching, this may take a while...', ephemeral=True)

    try:
        if channel is None:
            channel = interaction.channel

        # Fetch messages from cache
        cached_messages = await cache.get_all_messages_in_channel_from_cache(channel)

        if len(cached_messages) == 0:
            await interaction.edit_original_response(content=f'No cached messages found for channel {channel.name}')
            return

        await interaction.edit_original_response(content=f'Found {len(cached_messages)} cached messages, running search...')

        # Format messages into plain text for the embeddings
        cached_messages_text = [msg.content for msg in cached_messages if msg.content is not None and len(msg.content) > 0]

        # Run embeddings and find best match
        question_embedding, message_embeddings = await summary_llm.run_embeddings(interaction, query, cached_messages_text)

        best_match_index = message_embeddings.index(max(message_embeddings, key=lambda x: x[0]))
        best_match_message = cached_messages[best_match_index]
        best_match_score = message_embeddings[best_match_index][0] * 100

        embed = discord.Embed(title="Best Match", description=best_match_message)
        embed.add_field(name="Match Score", value=f"{best_match_score:.2f}%")
        embed.set_footer(text=f"From channel {channel.name}")

        await interaction.edit_original_response(content='', embed=embed)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise

async def uwuify_impl(interaction: discord.Interaction, message: discord.Message, ephemeral: bool):
    await interaction.response.send_message('Doing stuff, might take a while...', ephemeral=ephemeral)

    model = 'llama-3.3-70b-instruct'
    temperature = 1.26
    top_p = 0.81
    max_tokens = 4096
    presence_penalty = 0

    messages = [{'role': 'system', 'content': 'UwU-ify the users message, and echo it, without saying anything else. '
                                              'Use emoticons too.'},
                {'role': 'user', 'content': f'{message.content}'}]

    pprint(messages)

    try:
        completion = await ai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            top_p=top_p,
            presence_penalty=presence_penalty
        )

        embed = discord.Embed(description=completion.choices[0].message.content)
        embed.add_field(name='LLM Info', value=f'Model: {model}, Max Tokens: {max_tokens}, '
                                               f'Temperature: {temperature}')
        embed.set_footer(text=f'Original message by {message.author}')

        await interaction.edit_original_response(content='', embed=embed)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.context_menu(name='UwU-ify message :3')
@app_commands.allowed_installs(guilds=False, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def uwuify(interaction: discord.Interaction, message: discord.Message):
    await uwuify_impl(interaction=interaction, message=message, ephemeral=False)


@client.tree.context_menu(name='UwU-ify message (Keep Private)')
@app_commands.allowed_installs(guilds=False, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def uwuify_ephemeral(interaction: discord.Interaction, message: discord.Message):
    await uwuify_impl(interaction=interaction, message=message, ephemeral=True)


async def zoomer_translator_impl(interaction: discord.Interaction, message: discord.Message, ephemeral: bool):
    await interaction.response.send_message('Doing stuff, might take a while...', ephemeral=ephemeral)

    model = 'deepseek-r1-distill-llama-70b'
    temperature = 0.70
    top_p = 0.95
    max_tokens = 4096
    presence_penalty = 0

    messages = [{'role': 'system', 'content': 'You are given messages that use many acronyms, phrases, and words '
                                              'associated with \"zoomers\" and zoomer culture. You should translate '
                                              'the messages into more conventional language and grammar, and then '
                                              'say the translated message, and nothing else.'},
                {'role': 'user', 'content': f'{message.content}'}]

    pprint(messages)

    try:
        completion = await ai_client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
            top_p=top_p,
            presence_penalty=presence_penalty
        )

        embed = discord.Embed(description=completion.choices[0].message.content)
        embed.add_field(name='LLM Info', value=f'Model: {model}, Max Tokens: {max_tokens}, '
                                               f'Temperature: {temperature}')
        embed.set_footer(text=f'Original message by {message.author}')

        await interaction.edit_original_response(content='', embed=embed)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise

@client.tree.context_menu(name='Zoomer Translator')
@app_commands.allowed_installs(guilds=False, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def zoomer_translator(interaction: discord.Interaction, message: discord.Message):
    await zoomer_translator_impl(interaction=interaction, message=message, ephemeral=False)

@client.tree.context_menu(name='Zoomer Translator (Keep Private)')
@app_commands.allowed_installs(guilds=False, users=True)
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def zoomer_translator_ephemeral(interaction: discord.Interaction, message: discord.Message):
    await zoomer_translator_impl(interaction=interaction, message=message, ephemeral=True)


def main() -> None:
    client.run(discord_token)


if __name__ == '__main__':
    main()
