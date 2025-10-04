import discord
import io
import asyncio
from datetime import datetime
from typing import List, Optional

from pprint import pprint
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from summarizer.config import ai_client, SUMMARIZER_MODEL, EMBEDDING_MODEL
from summarizer.token_utils import estimate_tokens, count_tokens
from summarizer.cache import *
from summarizer.embeddings import run_embeddings


def format_message_list(messages: List[discord.Message]):
    formatted_messages = []
    for discord_message in messages:
        formatted_messages.append(f'{discord_message.author.name}: {discord_message.content}')
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

    embed.add_field(name='LLM Info', value=f'Model: {model}, Max Tokens: {max_tokens}, '
                                           f'Temperature: {temperature}')

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


async def _create_summary_lmm_messages(interaction: discord.Interaction, discord_messages: List[discord.Message], summarize_prompt: str,
                         footer_text: str, show_when_context_too_large: bool = True):
    # Build participant summary AFTER truncation
    formatted_messages = format_message_list(discord_messages)
    MODEL_CONTEXT_TOKENS = 300_000
    RESERVED_RESPONSE_TOKENS = 8_192
    RESERVED_PARTICIPANT_TOKENS = 6_000

    system_base = (
        f'You should summarize the following messages according to this prompt: '
        f'"{summarize_prompt}". Use only information mentioned in the following messages.\n\n'
        f'If some messages do not contain any information relevant to the prompt, ignore them.\n\n'
        f'In your response, refer to people by their display name, rather than user name, when possible.'
    )

    # Use model tokenization when available
    MODEL = SUMMARIZER_MODEL
    system_base_tokens = await count_tokens({'role': 'system', 'content': system_base}, model_name=MODEL)
    allowed_for_messages = MODEL_CONTEXT_TOKENS - RESERVED_RESPONSE_TOKENS - RESERVED_PARTICIPANT_TOKENS - system_base_tokens
    if allowed_for_messages < 0:
        allowed_for_messages = 0

    # Count tokens for each formatted message concurrently
    if formatted_messages:
        msg_tokens = await asyncio.gather(*[count_tokens(m, model_name=MODEL) for m in formatted_messages])
    else:
        msg_tokens = []
    total_msg_tokens = sum(msg_tokens)

    removed_count = 0
    if total_msg_tokens > allowed_for_messages and formatted_messages:
        prefix_sums = [0]
        for t in msg_tokens:
            prefix_sums.append(prefix_sums[-1] + t)
        total = prefix_sums[-1]
        need_at_most = total - allowed_for_messages
        import bisect
        k = bisect.bisect_left(prefix_sums, need_at_most)
        if k < 0:
            k = 0
        if k >= len(formatted_messages):
            k = len(formatted_messages) - 1
        formatted_messages = formatted_messages[k:]
        msg_tokens = msg_tokens[k:]
        removed_count = k
        total_msg_tokens = sum(msg_tokens)

    if removed_count > 0:
        footer_text = footer_text + f'\n\nNote: {removed_count} earlier message(s) were omitted to fit the model context.'

    truncated_discord_messages = discord_messages[-len(formatted_messages):] if formatted_messages else []

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

    guild = getattr(interaction, 'guild', None)
    sorted_participants = sorted(participants.items(), key=lambda kv: kv[1].get('count', 0), reverse=True)

    parts = []
    used_participant_tokens = 0

    # Resolve display names and roles using guild.get_member (local cache). Avoid fetch_member network calls.
    CHUNK_SIZE = 32
    _member_cache = {}
    for i in range(0, len(sorted_participants), CHUNK_SIZE):
        chunk = sorted_participants[i:i + CHUNK_SIZE]
        for uid, info in chunk:
            display_name = info.get('name', 'Unknown')
            roles_set = set()
            if uid is None:
                continue
            # Prefer any in-memory or guild cached member
            try:
                member = guild.get_member(uid) if guild is not None else None
            except Exception:
                member = None

            if member is not None:
                try:
                    display_name = getattr(member, 'display_name', display_name)
                except Exception:
                    pass
                try:
                    for r in getattr(member, 'roles', []):
                        rname = getattr(r, 'name', None)
                        if rname and rname != '@everyone':
                            roles_set.add(rname)
                except Exception:
                    pass

            _member_cache[uid] = {'display_name': display_name, 'roles': roles_set}

        for uid, info in chunk:
            display_name = info.get('name', 'Unknown')
            roles_set = set()
            if uid in _member_cache:
                cached = _member_cache.get(uid, {})
                if cached.get('display_name'):
                    display_name = cached.get('display_name')
                roles_set = cached.get('roles', set()) or set()

            username = info.get('name', 'Unknown')
            count = info.get('count', 0)
            roles = sorted(roles_set)
            roles_str = ', '.join(roles) if roles else 'none'
            line = f"- {display_name} (username: {username}) ({count} message{'s' if count!=1 else ''}) — roles: {roles_str}"
            tok = await count_tokens(line, model_name=MODEL, per_message_overhead=False)
            if used_participant_tokens + tok > RESERVED_PARTICIPANT_TOKENS:
                break
            parts.append(line)
            used_participant_tokens += tok

        if used_participant_tokens >= RESERVED_PARTICIPANT_TOKENS:
            break

    participants_summary = 'Major Participants:\n' + '\n'.join(parts) if parts else ''
    pprint(participants_summary)

    system_content = system_base + '\n\n' + participants_summary if participants_summary else system_base

    llm_messages = [{'role': 'system', 'content': system_content}]
    for message in formatted_messages:
        llm_messages.append({'role': 'user', 'content': message})

    embed = discord.Embed(title='Generating summary')
    embed.set_footer(text=footer_text)
    return llm_messages, embed


async def create_summary(interaction: discord.Interaction, discord_messages: List[discord.Message], summarize_prompt: str,
                         footer_text: str, show_when_context_too_large: bool = True):
    llm_messages, embed = await _create_summary_lmm_messages(interaction, discord_messages, summarize_prompt,
                                                            footer_text, show_when_context_too_large)
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

    embed = discord.Embed(title='Generating summary')
    embed.set_footer(text=footer_text)

    await run_llm(interaction, llm_messages, embed)
