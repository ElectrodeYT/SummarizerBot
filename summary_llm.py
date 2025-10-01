import discord
import os
from datetime import datetime
from cache import *
import io

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

    # pprint(formatted_messages)

    return formatted_messages


def format_message_list_for_embeddings(messages: [discord.Message]):
    formatted_messages = []

    for discord_message in messages:
        formatted_messages.append(f'{discord_message.content}')

    return formatted_messages


async def run_llm(interaction: discord.Interaction, llm_messages: [], embed: discord.Embed):
    model = 'gpt-oss-120b'
    temperature = 0.7
    max_tokens = 8192

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

        delta_content = chunk.choices[0].delta.content
        if delta_content is not None:
            summary += delta_content

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
        await interaction.edit_original_response(embed=embed, attachments=[discord.File(fp=f, filename='^generated.txt')])
    else:
        await interaction.edit_original_response(embed=embed)


async def fetch_and_cache_embeddings(messages: [],  model: str):
    message_embeddings_response = await ai_client.embeddings.create(
        input=messages,
        model=model,
    )

    # Commit the generated embeddings to cache
    for message, embedding_response in zip(messages, message_embeddings_response.data):
        await commit_embedding_to_cache(message, embedding_response.embedding, model)

    return message_embeddings_response

async def run_embeddings(interaction: discord.Interaction, base_sentence: str, messages: []):
    model = 'bge-multilingual-gemma2'

    message_embeddings = []
    message_embeddings_to_fetch = []

    # Try to fetch from cache first
    for message in messages:
        cached_embedding = await fetch_embedding_from_cache(message, model)
        if cached_embedding is not None:
            message_embeddings.append(cached_embedding)
        else:
            message_embeddings_to_fetch.append(message)

    # Fetch embeddings for messages not in cache
    # We do this in chunks of 1024 messages as the API doesn't support more than that
    message_embeddings_response = []
    if len(message_embeddings_to_fetch) != 0:
        for i in range(0, len(message_embeddings_to_fetch), 1024):
            chunk = message_embeddings_to_fetch[i:i + 1024]
            fetched_message_embeddings = await fetch_and_cache_embeddings(chunk, model)
            message_embeddings_response.extend(fetched_message_embeddings.data)

    question_embeddings_response = await ai_client.embeddings.create(
        input=base_sentence,
        model=model,
    )

    # Convert the embedding responses into float arrays
    question_embeddings = question_embeddings_response.data[0].embedding
    for embedding_response in message_embeddings_response:
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
