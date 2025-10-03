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
import cProfile
import bisect
import asyncio

if 'OPENAI_API_KEY' not in os.environ or not os.environ['OPENAI_API_KEY']:
    raise Exception('Missing OPENAI_API_KEY environment variable')

if 'OPENAI_API_BASE' not in os.environ or not os.environ['OPENAI_API_BASE']:
    raise Exception('Missing OPENAI_API_BASE environment variable')

OPENAPI_TOKEN = os.environ['OPENAI_API_KEY']

ai_client = AsyncOpenAI(
    base_url=os.environ.get('OPENAI_API_BASE'),
    api_key=OPENAPI_TOKEN
)

SUMMARIZER_MODEL = os.environ.get('SUMMARIZER_MODEL', 'google/gemini-2.5-flash-lite-preview-09-2025')
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')


# Module-level caches to avoid repeated expensive work
# Cache for tiktoken encoders per model name
_tiktoken_encoders: dict = {}
# Simple in-memory token count cache: sha1(model_name + text) -> token_count
_token_count_cache: dict = {}
# Small in-memory cache for resolved member display names and roles across calls
_member_cache: dict = {}

def estimate_tokens(text: str, model_name: str | None = None) -> int:
    """Estimate token count for a text. Tries to use tiktoken if available, otherwise falls back to a simple heuristic (chars/4)."""
    if not text:
        return 0
    # Use a small in-memory cache to avoid re-encoding identical texts repeatedly
    try:
        key = hashlib.sha1(((model_name or '') + ':' + text).encode('utf-8')).hexdigest()
    except Exception:
        # fallback if hashing fails for some reason
        key = None

    if key is not None and key in _token_count_cache:
        return _token_count_cache[key]

    # Try to use tiktoken if available and cache encoders per-model
    try:
        import tiktoken
        enc = None
        enc_key = model_name or '__default__'
        if enc_key in _tiktoken_encoders:
            enc = _tiktoken_encoders[enc_key]
        else:
            try:
                if model_name:
                    enc = tiktoken.encoding_for_model(model_name)
                else:
                    enc = tiktoken.get_encoding('o200k_harmony')
            except Exception:
                try:
                    enc = tiktoken.get_encoding('o200k_harmony')
                except Exception:
                    enc = None
            if enc is not None:
                _tiktoken_encoders[enc_key] = enc

        if enc is not None:
            tok_count = len(enc.encode(text))
            if key is not None:
                _token_count_cache[key] = tok_count
            return tok_count
    except Exception:
        # tiktoken not available or failed; fall back to heuristic below
        pass

    # conservative heuristic: 1 token per 4 characters
    tok_count = max(1, int(len(text) / 4))
    if key is not None:
        _token_count_cache[key] = tok_count
    return tok_count


def count_tokens(obj, model_name: str | None = None, per_message_overhead: bool = True) -> int:
    """Count tokens for various inputs.

    Supports:
    - strings: treated as a single message content (adds per-message overhead when per_message_overhead=True)
    - list of strings: each element treated as a separate message
    - dict with keys 'role' and 'content': counts tokens for role and content and adds per-message overhead
    - list of dicts: sums over the dicts

    The per_message_overhead accounts for the "start/end" tokens (4 tokens) per message when True.
    """
    OVERHEAD_PER_MESSAGE = 4 if per_message_overhead else 0

    # Single string -> count tokens for the content and add overhead
    if isinstance(obj, str):
        return estimate_tokens(obj, model_name) + OVERHEAD_PER_MESSAGE

    # Dict with role/content
    if isinstance(obj, dict):
        role = obj.get('role', '')
        content = obj.get('content', '')
        return estimate_tokens(str(role), model_name) + estimate_tokens(str(content), model_name) + OVERHEAD_PER_MESSAGE

    # List -> sum elements (support list of strings or list of dicts)
    if isinstance(obj, list):
        total = 0
        for elem in obj:
            # For nested lists/dicts/strings, recurse; keep per_message_overhead as-is
            total += count_tokens(elem, model_name=model_name, per_message_overhead=per_message_overhead)
        return total

    # Fallback: stringify and count
    return estimate_tokens(str(obj), model_name) + OVERHEAD_PER_MESSAGE


@dataclass
class SearchEmbedding:
    author: CachedDiscordAuthor
    content: str
    message_ids: List[int]
    embedding: Optional[List[float]]
    model: str


async def create_search_embeddings(channel: discord.TextChannel, model: str = 'bge-multilingual-gemma2', limit: int | None = None,
                                   progress_callback: Optional[Callable[[str, int, int], Awaitable[None]]] = None,
                                   max_tokens: int = 8192) -> List[SearchEmbedding]:
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

    # Build contexted texts: for each block, include up to `context_radius` previous and next blocks
    context_radius = 3
    # Conservative char->token estimate (approx 4 chars per token)
    max_chars = max(1, int(max_tokens * 4))

    def estimate_tokens_for_text(t: str) -> int:
        # Use the lighter-weight estimate for grouping/embeddings stage (no per-message overhead)
        return estimate_tokens(t)

    context_texts: List[Optional[str]] = []
    for i in range(len(blocks)):
        central = blocks[i][1]
        # If the central block alone exceeds the limit, skip it
        if estimate_tokens_for_text(central) > max_tokens:
            context_texts.append(None)
            continue

        # Try decreasing context radius until the contexted text fits within max_tokens
        radius = context_radius
        chosen_text = None
        while radius >= 0:
            start = max(0, i - radius)
            end = min(len(blocks), i + radius + 1)
            window_texts = [blocks[j][1] for j in range(start, end)]
            context_text = '\n'.join(window_texts)
            if estimate_tokens_for_text(context_text) <= max_tokens:
                chosen_text = context_text
                break
            # reduce radius (this removes context around central block first)
            radius -= 1

        # As a fallback, use the central block (we already checked it's <= max_tokens)
        if chosen_text is None:
            chosen_text = central

        context_texts.append(chosen_text)

    # Prepare hashes for non-skipped context_texts and fetch cached embeddings in one batch
    hashes: List[Optional[str]] = [ (hashlib.sha1(txt.encode('utf-8')).hexdigest() if txt is not None else None) for txt in context_texts ]
    # Only pass non-None hashes to DB helper
    lookup_hashes = [h for h in hashes if h is not None]
    cached_map = await fetch_embeddings_for_hashes(lookup_hashes, model) if lookup_hashes else {}

    cache_searched = 0
    for idx, (author, _concat, ids) in enumerate(blocks):
        txt = context_texts[idx]
        if txt is None:
            # skipped due to being too large
            continue

        h = hashes[idx]
        emb = cached_map.get(h) if h is not None else None
        se = SearchEmbedding(author=author, content=txt, message_ids=ids, embedding=emb, model=model)
        result_idx = len(results)
        results.append(se)
        if emb is None:
            need_fetch_texts.append(txt)
            need_fetch_indices.append(result_idx)
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


def format_message_list_for_embeddings(messages: List[discord.Message]) -> List[str]:
    return [f"{msg.content}" for msg in messages]
 
async def run_llm(interaction: discord.Interaction, llm_messages: list, embed: discord.Embed):
    model = SUMMARIZER_MODEL
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
    model = EMBEDDING_MODEL

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

async def _create_summary_lmm_messages(interaction: discord.Interaction, discord_messages: List[discord.Message], summarize_prompt: str,
                         footer_text: str, show_when_context_too_large: bool = True):
    # We'll build the participants summary after truncating messages to the token budget.
    # (Avoid expensive member resolution before truncation.)

    # Prepare formatted messages
    formatted_messages = format_message_list(discord_messages)
    # First: prepare formatted messages and cull by token budget BEFORE building participants summary.
    formatted_messages = format_message_list(discord_messages)
    MODEL_CONTEXT_TOKENS = 300_000
    RESERVED_RESPONSE_TOKENS = 8_192
    RESERVED_PARTICIPANT_TOKENS = 2_000

    # Base system text without participants summary
    system_base = (
        f'You should summarize the following messages according to this prompt: '
        f'"{summarize_prompt}". Use only information mentioned in the following messages.\n\n'
        f'If some messages do not contain any information relevant to the prompt, ignore them.\n\n'
        f'In your response, refer to people by their display name, rather than user name, when possible.'
    )

    # Count system_base as a system message (include role + overhead)
    system_base_tokens = count_tokens({'role': 'system', 'content': system_base})
    allowed_for_messages = MODEL_CONTEXT_TOKENS - RESERVED_RESPONSE_TOKENS - RESERVED_PARTICIPANT_TOKENS - system_base_tokens
    if allowed_for_messages < 0:
        allowed_for_messages = 0

    # Count each formatted message as a separate user message (include per-message overhead)
    # Use cached token counts where available; build an array of token counts
    msg_tokens = [count_tokens(m) for m in formatted_messages]
    total_msg_tokens = sum(msg_tokens)
    print(f'Total message tokens: {total_msg_tokens}, allowed for messages: {allowed_for_messages}')

    removed_count = 0
    # Instead of popping from the front repeatedly (O(n^2)), compute prefix sums and find cutoff index.
    if total_msg_tokens > allowed_for_messages and formatted_messages:
        # prefix_sums[i] = total tokens of first i messages
        prefix_sums = [0]
        for t in msg_tokens:
            prefix_sums.append(prefix_sums[-1] + t)

        total = prefix_sums[-1]
        # We need smallest k such that total - prefix_sums[k] <= allowed_for_messages
        # i.e., prefix_sums[k] >= total - allowed_for_messages
        need_at_most = total - allowed_for_messages
        # find leftmost index where prefix_sums[idx] >= need_at_most
        k = bisect.bisect_left(prefix_sums, need_at_most)
        if k < 0:
            k = 0
        if k >= len(formatted_messages):
            # everything would be removed; keep last message as a fallback
            k = len(formatted_messages) - 1

        # Remove the first k messages in one slice operation
        formatted_messages = formatted_messages[k:]
        msg_tokens = msg_tokens[k:]
        removed_count = k
        total_msg_tokens = sum(msg_tokens)

    # Note truncation in footer if messages were removed
    if removed_count > 0:
        footer_text = footer_text + f'\n\nNote: {removed_count} earlier message(s) were omitted to fit the model context.'

    # Recompute the truncated discord_messages slice so participants are based on included messages
    truncated_discord_messages = discord_messages[-len(formatted_messages):] if formatted_messages else []

    # Now build participants summary from truncated messages and fit it into RESERVED_PARTICIPANT_TOKENS
    participants: dict = {}
    for m in truncated_discord_messages:
        try:
            author = getattr(m, 'author', None)
            if author is None:
                continue
            uid = getattr(author, 'id', None)
            name = getattr(author, 'name', 'Unknown')
            if uid not in participants:
                participants[uid] = {'name': name, 'count': 0}
            participants[uid]['count'] += 1
        except Exception:
            continue

    # Prepare to fetch member info as needed and include as many top participants as fit
    guild = getattr(interaction, 'guild', None)
    # Sort participants by message count desc
    sorted_participants = sorted(participants.items(), key=lambda kv: kv[1].get('count', 0), reverse=True)

    parts = []
    used_participant_tokens = 0

    # Helper to fetch a single member with semaphore protection
    async def _fetch_member_with_sem(uid: int, sem: asyncio.Semaphore):
        async with sem:
            try:
                return await guild.fetch_member(uid)
            except Exception:
                return None

    # Process participants in small chunks. For each chunk, concurrently fetch missing members
    # with a bounded semaphore to avoid excessive concurrent requests to Discord.
    CONCURRENCY = 8
    CHUNK_SIZE = 32
    sem = asyncio.Semaphore(CONCURRENCY)

    # Iterate over participants in chunks (sorted by message count desc)
    for i in range(0, len(sorted_participants), CHUNK_SIZE):
        chunk = sorted_participants[i:i + CHUNK_SIZE]

        # First, prepare which UIDs need fetching
        to_fetch = []  # list of uids that need guild.fetch_member
        fetch_tasks = []
        for uid, info in chunk:
            if uid is None:
                continue
            # If cached in member cache, skip
            if uid in _member_cache:
                continue
            # Try local cached guild member first (non-async)
            try:
                member = guild.get_member(uid) if guild is not None else None
            except Exception:
                member = None
            if member is not None:
                # populate member cache
                try:
                    roles_set = set()
                    for r in getattr(member, 'roles', []):
                        rname = getattr(r, 'name', None)
                        if rname and rname != '@everyone':
                            roles_set.add(rname)
                except Exception:
                    roles_set = set()
                _member_cache[uid] = {'display_name': getattr(member, 'display_name', info.get('name', 'Unknown')), 'roles': roles_set}
            else:
                to_fetch.append(uid)

        # Launch concurrent fetches for this chunk, bounded by semaphore
        if to_fetch and guild is not None:
            for uid in to_fetch:
                fetch_tasks.append(asyncio.create_task(_fetch_member_with_sem(uid, sem)))

            # Gather results and populate cache
            if fetch_tasks:
                fetched_members = await asyncio.gather(*fetch_tasks)
                for uid, member in zip(to_fetch, fetched_members):
                    if member is None:
                        # store a sentinel to avoid refetching repeatedly
                        _member_cache[uid] = {'display_name': None, 'roles': set()}
                    else:
                        roles_set = set()
                        try:
                            for r in getattr(member, 'roles', []):
                                rname = getattr(r, 'name', None)
                                if rname and rname != '@everyone':
                                    roles_set.add(rname)
                        except Exception:
                            pass
                        _member_cache[uid] = {'display_name': getattr(member, 'display_name', None), 'roles': roles_set}

        # Now process this chunk in order and add participant lines until we run out of participant token budget
        for uid, info in chunk:
            display_name = info.get('name', 'Unknown')
            roles_set = set()
            if uid in _member_cache:
                cached = _member_cache.get(uid, {})
                if cached.get('display_name'):
                    display_name = cached.get('display_name')
                roles_set = cached.get('roles', set()) or set()
            else:
                # fallback: try guild.get_member one last time
                try:
                    member = guild.get_member(uid) if guild is not None else None
                    if member is not None:
                        display_name = getattr(member, 'display_name', display_name)
                        try:
                            for r in getattr(member, 'roles', []):
                                rname = getattr(r, 'name', None)
                                if rname and rname != '@everyone':
                                    roles_set.add(rname)
                        except Exception:
                            pass
                except Exception:
                    pass

            username = info.get('name', 'Unknown')
            count = info.get('count', 0)
            roles = sorted(roles_set)
            roles_str = ', '.join(roles) if roles else 'none'
            line = f"- {display_name} (username: {username}) ({count} message{'s' if count!=1 else ''}) — roles: {roles_str}"
            # Participant summary lines are part of the system content, do NOT include per-message overhead
            tok = count_tokens(line, per_message_overhead=False)
            if used_participant_tokens + tok > RESERVED_PARTICIPANT_TOKENS:
                # no room for this participant line, stop adding participants
                break
            parts.append(line)
            used_participant_tokens += tok

        # If we've filled the participant token budget, break out of chunk loop early
        if used_participant_tokens >= RESERVED_PARTICIPANT_TOKENS:
            break

    participants_summary = 'Major Participants:\n' + '\n'.join(parts) if parts else ''

    # Build final system content including the participants summary
    system_content = system_base + '\n\n' + participants_summary if participants_summary else system_base

    # Assemble llm_messages from truncated formatted_messages
    llm_messages = [{'role': 'system', 'content': system_content}]
    for message in formatted_messages:
        llm_messages.append({'role': 'user', 'content': message})

    # Stream the summary
    embed = discord.Embed(title='Generating summary')
    embed.set_footer(text=footer_text)
    return llm_messages, embed

async def create_summary(interaction: discord.Interaction, discord_messages: List[discord.Message], summarize_prompt: str,
                         footer_text: str, show_when_context_too_large: bool = True):
    #profiler = cProfile.Profile()
    #profiler.enable()
    llm_messages, embed = await _create_summary_lmm_messages(interaction, discord_messages, summarize_prompt,
                                                            footer_text, show_when_context_too_large)
    #profiler.disable()
    #profiler.print_stats(sort='time')
    await run_llm(interaction, llm_messages, embed)
    #return profiler

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
