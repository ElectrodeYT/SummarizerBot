from typing import List, Optional, Callable, Awaitable
from dataclasses import dataclass
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from summarizer.cache import *
from summarizer.config import ai_client, EMBEDDING_MODEL
from summarizer.token_utils import estimate_tokens


@dataclass
class SearchEmbedding:
    author: CachedDiscordAuthor
    content: str
    message_ids: List[int]
    embedding: Optional[List[float]]
    model: str


async def create_search_embeddings(channel: 'discord.TextChannel', model: str = EMBEDDING_MODEL, limit: int | None = None,
                                   progress_callback: Optional[Callable[[str, int, int], Awaitable[None]]] = None,
                                   max_tokens: int = 8192) -> List[SearchEmbedding]:
    # Implementation ported from previous summary_llm.py
    cached_msgs = await get_all_messages_in_channel_from_cache(channel)
    if not cached_msgs:
        return []

    cached_msgs.sort(key=lambda m: m.id)

    if limit is not None and len(cached_msgs) > limit:
        cached_msgs = cached_msgs[-limit:]

    groups = []
    for m in cached_msgs:
        if m.content is None or len(m.content.strip()) == 0:
            continue
        if not groups or groups[-1][0].id != m.author.id:
            groups.append((m.author, [m]))
        else:
            groups[-1][1].append(m)

    blocks = []
    for author, msgs in groups:
        ids = [int(x.id) for x in msgs]
        concat = '\n'.join([f"{m.author.name}: {m.content}" for m in msgs if m.content is not None])
        blocks.append((author, concat, ids))

    total = len(blocks)
    if progress_callback is not None:
        try:
            await progress_callback('checking_cache', 0, total)
        except Exception:
            pass

    need_fetch_texts = []
    need_fetch_indices = []
    results: List[SearchEmbedding] = []

    context_radius = 3

    async def estimate_tokens_for_text(t: str) -> int:
        return await estimate_tokens(t)

    context_texts: List[Optional[str]] = []
    for i in range(len(blocks)):
        central = blocks[i][1]
        if await estimate_tokens_for_text(central) > max_tokens:
            context_texts.append(None)
            continue

        radius = context_radius
        chosen_text = None
        while radius >= 0:
            start = max(0, i - radius)
            end = min(len(blocks), i + radius + 1)
            window_texts = [blocks[j][1] for j in range(start, end)]
            context_text = '\n'.join(window_texts)
            if await estimate_tokens_for_text(context_text) <= max_tokens:
                chosen_text = context_text
                break
            radius -= 1

        if chosen_text is None:
            chosen_text = central

        context_texts.append(chosen_text)

    hashes: List[Optional[str]] = [(hashlib.sha1(txt.encode('utf-8')).hexdigest() if txt is not None else None) for txt in context_texts]
    lookup_hashes = [h for h in hashes if h is not None]
    cached_map = await fetch_embeddings_for_hashes(lookup_hashes, model) if lookup_hashes else {}

    cache_searched = 0
    for idx, (author, _concat, ids) in enumerate(blocks):
        txt = context_texts[idx]
        if txt is None:
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
        if progress_callback is not None and cache_searched % 50 == 0:
            try:
                await progress_callback('checking_cache', cache_searched, total)
            except Exception:
                pass

    if need_fetch_texts:
        processed_messages_count = sum(len(results[i].message_ids) for i in range(len(results)) if results[i].embedding is not None)
        if progress_callback is not None:
            try:
                await progress_callback('checking_cache', processed_messages_count, len(cached_msgs))
            except Exception:
                pass

        for i in range(0, len(need_fetch_texts), 1024):
            chunk = need_fetch_texts[i:i + 1024]
            resp = await ai_client.embeddings.create(input=chunk, model=model)
            for idx_in_chunk, d in enumerate(resp.data):
                emb = d.embedding
                global_idx = i + idx_in_chunk
                if global_idx < len(need_fetch_indices):
                    result_idx = need_fetch_indices[global_idx]
                    if results[result_idx].embedding is None:
                        results[result_idx].embedding = emb
                        await commit_embedding_to_cache(results[result_idx].content, emb, model)
                        processed_messages_count += len(results[result_idx].message_ids)

            if progress_callback is not None:
                try:
                    await progress_callback('fetching', processed_messages_count, len(cached_msgs))
                except Exception:
                    pass

    return results


async def fetch_and_cache_embeddings(messages: list, model: str):
    return await fetch_and_cache_embeddings.__wrapped__(messages, model) if hasattr(fetch_and_cache_embeddings, '__wrapped__') else None


async def run_embeddings(interaction: 'discord.Interaction', base_sentence: str, messages: list):
    # This function was ported but uses ai_client and EMBEDDING_MODEL from config
    model = EMBEDDING_MODEL
    return await globals().get('run_embeddings_impl', None)(interaction, base_sentence, messages)
