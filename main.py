import tempfile
import os
from datetime import datetime
from pprint import pprint


import discord
from openai import AsyncOpenAI
from discord import app_commands

scaleway_token = os.environ['SCW_TOKEN']
discord_token = os.environ['DISCORD_TOKEN']
default_summary_prompt = 'Summarize the conversation(s). If there are several conversations, summarize them individually.'

ai_client = AsyncOpenAI(
    base_url="https://api.scaleway.ai/v1",
    api_key=scaleway_token
)

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


intents = discord.Intents.default()
intents.message_content = True
client = DiscordClient(intents=intents)

def format_message_list(messages: [discord.Message]):
    formatted_messages = []

    for discord_message in messages:
        formatted_messages.append(f'{discord_message.author.name}: {discord_message.content}')

    return formatted_messages

async def run_llm(interaction: discord.Interaction, llm_messages: [], embed: discord.Embed):
    completion = await ai_client.chat.completions.create(
        model='deepseek-r1-distill-llama-70b',
        messages=llm_messages,
        temperature=0.7,
        max_tokens=4096,
        stream=True
    )

    # Don't edit too much
    last_edit = None
    summary = ''

    async for chunk in completion:
        pprint(chunk)

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
        with tempfile.NamedTemporaryFile() as tmpfile:
            tmpfile.write(summary.encode('utf-8'))
            await interaction.edit_original_response(embed=embed, file=discord.File(filename=tmpfile.name))
    else:
        await interaction.edit_original_response(embed=embed)

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

    for message in format_message_list(discord_messages):
        llm_messages.append({'role': 'user', 'content': message})

    # Stream the summary
    embed = discord.Embed(title='Generating summary')
    embed.set_footer(text=footer_text)

    await run_llm(interaction, llm_messages, embed)


@client.tree.command()
async def summarize(interaction: discord.Interaction, count_msgs: int = None, channel: discord.TextChannel = None,
                    summarize_prompt: str = default_summary_prompt) -> None:
    if channel is None:
        channel = interaction.channel

    await interaction.response.send_message('Doing stuff, might take a (long) while... (compiling messages)')

    # Get the messages, then invert the array
    discord_messages = [message async for message in channel.history(limit=count_msgs)][::-1]
    footer_text = f'Summary of the last {len(discord_messages)} messages in {channel.name} | Summary prompt: {summarize_prompt}'

    await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (Firing up AI)')
    await create_summary(interaction, discord_messages, summarize_prompt, footer_text)

@client.tree.context_menu(name='Summarize after this message')
async def summarize_after(interaction: discord.Interaction, message: discord.Message) -> None:
    channel = message.channel
    summarize_prompt = default_summary_prompt

    await interaction.response.send_message('Doing stuff, might take a (long) while... (compiling messages)', ephemeral=True)

    discord_messages = [message async for message in channel.history(after=message, oldest_first=True, limit=None)]
    footer_text = (f'Summary of {len(discord_messages)} messages in {channel.name} | Summary prompt: {summarize_prompt}\n'
                   f'After: [This message]({str(message.jump_url)})')

    await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (Firing up AI)')
    await create_summary(interaction, discord_messages, summarize_prompt, footer_text)

@client.tree.context_menu(name='Summarize around this message')
async def summarize_around(interaction: discord.Interaction, message: discord.Message) -> None:
    channel = message.channel
    summarize_prompt = default_summary_prompt

    await interaction.response.send_message('Doing stuff, might take a (long) while... (compiling messages)', ephemeral=True)

    discord_messages = [message async for message in channel.history(around=message, oldest_first=True, limit=101)]
    footer_text = f'Summary of {len(discord_messages)} messages in {channel.name} | Summary prompt: {summarize_prompt}\n'

    await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (Firing up AI)')
    await create_summary(interaction, discord_messages, summarize_prompt, footer_text)

@client.tree.command(description='Summarize for topic')
async def summarize_topic(interaction: discord.Interaction, topic: str, count_msgs: int = None,
                          channel: discord.TextChannel = None) -> None:
    if channel is None:
        channel = interaction.channel

    await interaction.response.send_message('Doing stuff, might take a (long) while... (compiling messages)')

    # Get the messages, then invert the array
    discord_messages = [message async for message in channel.history(limit=count_msgs)][::-1]

    footer_text = f'Summary of the last {len(discord_messages)} messages in {channel.name} | Topic: {topic}'

    await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (Firing up AI)')
    await create_topic_summary(interaction, discord_messages, topic, footer_text)

def main() -> None:
    client.run(discord_token)

if __name__ == '__main__':
    main()
