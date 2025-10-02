import discord
import os
from datetime import datetime
from cache import *
import io
import hashlib

from dataclasses import dataclass
from typing import List, Optional, Callable, Awaitable

from openai import AsyncOpenAI
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

scaleway_token = os.environ['SCW_TOKEN']

ai_client = AsyncOpenAI(
    base_url="https://api.scaleway.ai/v1",
    api_key=scaleway_token
)


@dataclass
class SearchEmbedding:
    author: CachedDiscordAuthor
    content: str
    message_ids: List[int]
    embedding: Optional[List[float]]
    model: str


async def create_search_embeddings(channel: discord.TextChannel, model: str = 'bge-multilingual-gemma2', limit: int | None = None,
                                   progress_callback: Optional[Callable[[str, int, int], Awaitable[None]]] = None) -> List[SearchEmbedding]:
    """Create search embeddings by concatenating consecutive messages from the same author.

    - Reads messages from cache only (uses get_all_messages_in_channel_from_cache)
    - Groups consecutive messages by the same author (oldest->newest order)
    - For each grouped block, attempts to fetch embedding from cache; if missing, requests embeddings
      from the AI API in batches and commits them to cache.
    Returns a list of SearchEmbedding objects.
    """
    # Fetch messages from cache and sort oldest->newest by id
    cached_msgs = await get_all_messages_in_channel_from_cache(channel)
    if not cached_msgs:
        return []

    cached_msgs.sort(key=lambda m: m.id)

    # Optionally limit to the most recent `limit` messages
    if limit is not None and len(cached_msgs) > limit:
        cached_msgs = cached_msgs[-limit:]

    # Group consecutive messages by the same author
    groups = []  # list of (author_obj, [messages])
    messages_seen = 0
    total_messages = len(cached_msgs)
    for m_idx, m in enumerate(cached_msgs):
        if m.content is None or len(m.content.strip()) == 0:
            messages_seen += 1
            # occasional progress update while grouping
            if progress_callback is not None and messages_seen % 50 == 0:
                try:
                    await progress_callback('grouping', messages_seen, total_messages)
                except Exception:
                    pass
            continue

        if not groups or groups[-1][0].id != m.author.id:
            groups.append((m.author, [m]))
        else:
            groups[-1][1].append(m)

        messages_seen += 1
        # occasional progress update while grouping
        if progress_callback is not None and messages_seen % 50 == 0:
            try:
                await progress_callback('grouping', messages_seen, total_messages)
            except Exception:
                pass

    # Build concatenated texts for each grouped block (include usernames)
    blocks = []  # tuples (author, concat_text, [ids])
    for author, msgs in groups:
        ids = [int(x.id) for x in msgs]
        # Prefix each message with the author's username to preserve speaker attribution
        concat = '\n'.join([f"{m.author.name}: {m.content}" for m in msgs if m.content is not None])
        blocks.append((author, concat, ids))

    total = len(blocks)
    # Report initial progress (starting checking cache stage)
    if progress_callback is not None:
        try:
            await progress_callback('checking_cache', 0, total)
        except Exception:
            pass

    # Try to fetch cached embeddings for each block using a batched DB query
    need_fetch_texts = []
    need_fetch_indices = []
    results: List[SearchEmbedding] = []

    # Build contexted texts: for each block, include up to 3 previous and 3 next block texts
    context_radius = 3
    context_texts: List[str] = []
    for i in range(len(blocks)):
        start = max(0, i - context_radius)
        end = min(len(blocks), i + context_radius + 1)
        # join the block texts in context window; keep order oldest->newest
        window_texts = [blocks[j][1] for j in range(start, end)]
        context_text = '\n'.join(window_texts)
        context_texts.append(context_text)

    # Compute SHA1 hashes for each contexted block (same scheme as cache)
    hashes = [hashlib.sha1(txt.encode('utf-8')).hexdigest() for txt in context_texts]

    # Fetch all found embeddings in a single (chunked) DB call for the contexted texts
    cached_map = await fetch_embeddings_for_hashes(hashes, model)

    cache_searched = 0
    for idx, (author, _concat, ids) in enumerate(blocks):
        h = hashes[idx]
        emb = cached_map.get(h)
        # Use the contexted text as the content associated with the embedding
        se = SearchEmbedding(author=author, content=context_texts[idx], message_ids=ids, embedding=emb, model=model)
        results.append(se)
        if emb is None:
            need_fetch_texts.append(context_texts[idx])
            need_fetch_indices.append(idx)
        cache_searched += 1
        # report after checking cache in batches
        if progress_callback is not None and cache_searched % 50 == 0:
            try:
                await progress_callback('checking_cache', cache_searched, total)
            except Exception:
                pass

    # Fetch embeddings for missing blocks in chunks, reporting progress after each chunk
    if need_fetch_texts:
        fetched = []
        processed_messages_count = sum(len(results[i].message_ids) for i in range(len(results)) if results[i].embedding is not None)
        # report after checking cache
        if progress_callback is not None:
            try:
                await progress_callback('checking_cache', processed_messages_count, total_messages)
            except Exception:
                pass
        for i in range(0, len(need_fetch_texts), 1024):
            chunk = need_fetch_texts[i:i + 1024]
            resp = await ai_client.embeddings.create(input=chunk, model=model)
            fetched.extend([d.embedding for d in resp.data])

            # After each chunk, commit and report progress
            for idx_in_chunk, emb in enumerate([d.embedding for d in resp.data]):
                global_idx = i + idx_in_chunk
                if global_idx < len(need_fetch_indices):
                    result_idx = need_fetch_indices[global_idx]
                    # avoid double-counting
                    if results[result_idx].embedding is None:
                        results[result_idx].embedding = emb
                        await commit_embedding_to_cache(results[result_idx].content, emb, model)
                        processed_messages_count += len(results[result_idx].message_ids)

            # report processed count
            if progress_callback is not None:
                try:
                    await progress_callback('fetching', processed_messages_count, total_messages)
                except Exception:
                    pass

    return results


def format_message_list(messages: List[discord.Message]):
    formatted_messages = []

    for discord_message in messages:
        formatted_messages.append(f'{discord_message.author.name}: {discord_message.content}')

    # pprint(formatted_messages)

    return formatted_messages


def format_message_list_for_embeddings(messages: List[discord.Message]):
    formatted_messages = []

    for discord_message in messages:
        formatted_messages.append(f'{discord_message.content}')

    return formatted_messages


async def run_llm(interaction: discord.Interaction, llm_messages: list, embed: discord.Embed):
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


async def fetch_and_cache_embeddings(messages: list,  model: str):
    message_embeddings_response = await ai_client.embeddings.create(
        input=messages,
        model=model,
    )

    # Commit the generated embeddings to cache
    for message, embedding_response in zip(messages, message_embeddings_response.data):
        await commit_embedding_to_cache(message, embedding_response.embedding, model)

    return message_embeddings_response

async def run_embeddings(interaction: discord.Interaction, base_sentence: str, messages: list):
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


async def create_summary(interaction: discord.Interaction, discord_messages: List[discord.Message], summarize_prompt: str,
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


async def create_topic_summary(interaction: discord.Interaction, discord_messages: List[discord.Message], topic: str, footer_text: str):
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
