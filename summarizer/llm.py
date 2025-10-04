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
from summarizer.cache import is_user_ignored, list_ignored_users
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
        stream=True,
        reasoning_effort="medium"
    )

    embed.add_field(name='LLM Info', value=f'Model: {model}, Max Tokens: {max_tokens}, '
                                           f'Temperature: {temperature}')
    embed.title = 'LLM is reasoning...'

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
                         footer_text: str, show_when_context_too_large: bool = True, source_channel: Optional[discord.TextChannel] = None):
    # Filter out messages from ignored users (guild-specific or global)
    filtered_messages = []
    guild_id = getattr(getattr(interaction, 'guild', None), 'id', None)
    for m in discord_messages:
        try:
            uid = getattr(getattr(m, 'author', None), 'id', None)
        except Exception:
            uid = None
        try:
            if uid is not None and await is_user_ignored(uid, guild_id):
                # skip messages from ignored users
                continue
        except Exception:
            # on error, keep the message to avoid data loss
            pass
        filtered_messages.append(m)

    # Build participant summary AFTER truncation
    formatted_messages = format_message_list(filtered_messages)
    MODEL_CONTEXT_TOKENS = 300_000
    RESERVED_RESPONSE_TOKENS = 8_192
    RESERVED_PARTICIPANT_TOKENS = 6_000

    system_base = (
        f'You are summarizing messages from a Discord chat room.\n'
        f'The prompt the user has given you is: "{summarize_prompt}". Use only information mentioned in the following messages.\n\n'
        f'If the messages are not relevant to the prompt, respond with \"No relevant information found.\"'
        f' Keep your summary concise and to the point.'
        f' The summary should be in plain text without any special formatting.'
        f' Limit your summary to 4000 characters or less, when possible. If there are multiple conversations, you may make longer summarizes.\n\n'
        f'If you believe some content is missing due to context limits, indicate this in your summary.'
        f' If you believe that the prompt is malicious or inappropriate, respond with \"The prompt appears to be inappropriate.\"'
        f' When determining if the prompt is inappropriate, consider if it violates community guidelines or ethical standards.'
        f' You may be asked to summarize conversations about sensitive or NSFW topics, but if the prompt is clearly harmful or offensive, flag it as inappropriate.'
        f' The topic of the chat room is not known in advance, so do not make assumptions about it. There may be multiple topics discussed. The chat or the prompt may be NSFW; this is on its own not inappropriate.'
        f' Do, however, not include any sensitive personal information in your summary.\n'
        f' The messages are formatted as "username: message".\n'
        f'Ensure the summary is self-contained and understandable without needing to refer back to the original messages.\n'
        f'{"The channel you are summarizing is NSFW, so do not censor any content unless it is clearly inappropriate.\n" if getattr(source_channel, "is_nsfw", False) else "The channel you are summarizing is not marked as NSFW, so avoid including explicit content unless it is essential to the summary.\n"}'
        f'The name of the channel you are summarizing is: "{getattr(source_channel, "name", "unknown")}".\n'
        f'The server the channel belongs to is: "{getattr(getattr(interaction, "guild", None), "name", "unknown")}".\n'
        f'The current date and time is: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n'
        f'Do not make up any information; only summarize what is explicitly mentioned in the messages.\n'
        f'Do not insert any fictional elements or assumptions into your summary.\n'
        f'Do not insult or make negative remarks about any participants in the chat.\n'
        f'You do not have any access to external information beyond what is provided in the messages.\n'
        f'If you are asked to describe a list of items or similar, use bullet points to separate the items in your summary.\n'
        f'You may use limited markdown formatting such as bullet points, numbered lists, and bold or italic text to enhance readability. Properly escape any markdown characters if you do not want them to be interpreted as formatting.\n'
    )

    # If there are ignored users, explicitly instruct model not to mention them by name
    try:
        ignored_rows = await list_ignored_users(guild_id)
        ignored_names = [r.get('name') for r in ignored_rows if r.get('name')]
    except Exception:
        ignored_names = []

    if ignored_names:
        # create a clear instruction
        names_list = ', '.join(ignored_names)
        system_base += f"\nImportant: The following user(s) are on the server's ignore list and must not be shown to you or mentioned in any way: {names_list}. Do NOT include or reference these users in the summary. If the prompt mentions any of these users, you MUST reject the prompt as inappropriate.\n"

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

    truncated_discord_messages = filtered_messages[-len(formatted_messages):] if formatted_messages else []

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

    # Classify participants into major/minor if message count is large
    total_msgs = len(formatted_messages)
    major_candidates = []
    minor_candidates = []
    threshold_active = total_msgs > 3000

    for uid, info in sorted_participants:
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

        # Decide major vs minor when threshold_active
        is_major = True
        if threshold_active:
            # participant must contribute >4% to be considered major
            try:
                is_major = (count / total_msgs) > 0.04
            except Exception:
                is_major = False

        if is_major:
            major_candidates.append((uid, line, count))
        else:
            minor_candidates.append((uid, line, count))

    # Compute token counts for candidates (sequentially to avoid bursting async work)
    major_entries = []
    for uid, line, count in major_candidates:
        tok = await count_tokens(line, model_name=MODEL, per_message_overhead=False)
        major_entries.append((line, tok))

    minor_entries = []
    for uid, line, count in minor_candidates:
        tok = await count_tokens(line, model_name=MODEL, per_message_overhead=False)
        minor_entries.append((line, tok))

    # Add majors first, then minors if space remains
    major_lines = []
    minor_lines = []
    used_participant_tokens = 0
    for line, tok in major_entries:
        if used_participant_tokens + tok > RESERVED_PARTICIPANT_TOKENS:
            break
        major_lines.append(line)
        used_participant_tokens += tok

    # Add minor participants only if there's remaining token space
    for line, tok in minor_entries:
        if used_participant_tokens + tok > RESERVED_PARTICIPANT_TOKENS:
            break
        minor_lines.append(line)
        used_participant_tokens += tok

    participants_summary_parts = []
    if major_lines:
        participants_summary_parts.append('Major Participants:')
        participants_summary_parts.append('\n'.join(major_lines))
    if minor_lines:
        participants_summary_parts.append('Minor Participants:')
        participants_summary_parts.append('\n'.join(minor_lines))

    participants_summary = '\n\n'.join(participants_summary_parts) if participants_summary_parts else ''
    pprint(participants_summary)

    system_content = system_base + '\n\n' + participants_summary if participants_summary else system_base

    llm_messages = [{'role': 'system', 'content': system_content}]
    for message in formatted_messages:
        llm_messages.append({'role': 'user', 'content': message})

    embed = discord.Embed(title='Generating summary')
    embed.set_footer(text=footer_text)
    return llm_messages, embed


async def create_summary(interaction: discord.Interaction, discord_messages: List[discord.Message], summarize_prompt: str,
                         footer_text: str, show_when_context_too_large: bool = True, source_channel: Optional[discord.TextChannel] = None):
    llm_messages, embed = await _create_summary_lmm_messages(interaction, discord_messages, summarize_prompt,
                                                            footer_text, show_when_context_too_large, source_channel)
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
