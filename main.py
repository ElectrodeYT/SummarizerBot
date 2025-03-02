import os
from openai import AsyncOpenAI
from pprint import pprint

import discord
from discord import app_commands

import summary_llm

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


async def uwuify_impl(interaction: discord.Interaction, message: discord.Message, ephemeral: bool):
    await interaction.response.send_message('Doing stuff, might take a while...', ephemeral=ephemeral)

    model = 'deepseek-r1-distill-llama-70b'
    temperature = 0.6
    top_p = 0.95
    max_tokens = 4096
    presence_penalty = 0

    messages = [{'role': 'system', 'content': 'UwU-ify every user message, and echo it, without saying'
                                              ' anything else.'},
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


def main() -> None:
    client.run(discord_token)


if __name__ == '__main__':
    main()
