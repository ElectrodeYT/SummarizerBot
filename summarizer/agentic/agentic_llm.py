from pydantic import BaseModel
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
    output_guardrail,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    function_tool,
    ItemHelpers,
    set_tracing_disabled,
    ModelSettings
)
from openai.types.shared import Reasoning
import summarizer.cache as cache
from summarizer.config import ai_client, AGENTIC_SUMMARIZER_MODEL, AGENTIC_GUARDRAIL_MODEL
import discord
from dataclasses import dataclass, field
import io
import asyncio
import re
from datetime import datetime
from collections import defaultdict
from typing import Any, cast

class OurModelProvider(ModelProvider):
    def get_model(self, model_name: str | None = None):
        return OpenAIChatCompletionsModel(model=model_name or AGENTIC_SUMMARIZER_MODEL, openai_client=ai_client)

set_tracing_disabled(True)

OUR_RUN_CONFIG = RunConfig(
    model_provider=OurModelProvider()
)

@dataclass
class DiscordAgenticContext:
    guild: discord.Guild
    message_snapshots: dict[int, list[discord.Message | cache.CachedDiscordMessage]] = field(default_factory=lambda: defaultdict(list))  # channel_id -> list of messages
    message_limit: int | None = None
    source_channel_id: int | None = None
    source_channel_name: str | None = None
    tool_calls: list[str] = field(default_factory=list)

    @property
    async def ignored_users(self) -> list[str]:
        ignored_users = await cache.list_ignored_users(self.guild.id)
        result: list[str] = []
        for row in ignored_users:
            user_id = row.get('user_id')
            if user_id is None:
                continue
            member = self.guild.get_member(int(user_id))
            if member is not None:
                result.append(f"{member.display_name}: {member.name} ({member.id})")
            else:
                result.append(f"Unknown User ({int(user_id)})")
        return result


async def _ignored_user_id_set(guild_id: int | None) -> set[int]:
    ignored_rows = cast(list[dict[str, Any]], await cache.list_ignored_users(guild_id))
    result: set[int] = set()
    for row in ignored_rows:
        user_id = row.get('user_id')
        if user_id is None:
            continue
        try:
            result.add(int(user_id))
        except Exception:
            continue
    return result


def _log_tool_call(ctx: DiscordAgenticContext, entry: str) -> None:
    ctx.tool_calls.append(entry)
    if len(ctx.tool_calls) > 200:
        del ctx.tool_calls[0 : len(ctx.tool_calls) - 200]


def _tool_name_from_call_entry(entry: str) -> str:
    head = entry.split('(', 1)[0].strip()
    return head or 'unknown tool'


def _latest_tool_name(ctx: DiscordAgenticContext) -> str | None:
    if not ctx.tool_calls:
        return None
    return _tool_name_from_call_entry(ctx.tool_calls[-1])


def _coerce_text(obj: Any) -> str:
    if obj is None:
        return ''
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple, set)):
        parts: list[str] = []
        for raw_part in cast(Any, obj):
            part = _coerce_text(raw_part).strip()
            if part:
                parts.append(part)
        return ' | '.join(parts)
    if isinstance(obj, dict):
        for key in ('content', 'text', 'summary', 'message'):
            value = obj.get(key)
            if value:
                return _coerce_text(value)
        return str(obj)

    model_dump = getattr(obj, 'model_dump', None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            text = _coerce_text(dumped)
            if text:
                return text
        except Exception:
            pass

    for attr in ('content', 'text', 'summary'):
        try:
            value = getattr(obj, attr, None)
        except Exception:
            value = None
        if value:
            return _coerce_text(value)
    return str(obj)


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return cast(Any, obj.get(key, default))
    return getattr(obj, key, default)


def _resolve_text_channel(ctx: DiscordAgenticContext, channel_id: Any, *, tool_name: str) -> discord.TextChannel | None:
    parsed_channel_id: int | None = None
    try:
        if channel_id is not None:
            parsed_channel_id = int(channel_id)
    except Exception:
        parsed_channel_id = None

    if parsed_channel_id is not None:
        direct = ctx.guild.get_channel(parsed_channel_id)
        if isinstance(direct, discord.TextChannel):
            return direct

    if ctx.source_channel_id is not None:
        fallback = ctx.guild.get_channel(ctx.source_channel_id)
        if isinstance(fallback, discord.TextChannel):
            _log_tool_call(
                ctx,
                f'{tool_name}(channel_id={channel_id}) -> Fallback to source channel {fallback.name} ({fallback.id})'
            )
            return fallback

    if ctx.source_channel_name:
        lowered = ctx.source_channel_name.lower()
        by_name = next((c for c in ctx.guild.text_channels if c.name.lower() == lowered), None)
        if by_name is not None:
            _log_tool_call(
                ctx,
                f'{tool_name}(channel_id={channel_id}) -> Fallback to source channel name {by_name.name} ({by_name.id})'
            )
            return by_name

    _log_tool_call(ctx, f'{tool_name}(channel_id={channel_id}) -> Error: channel not found')
    return None


def _resolve_member_by_username(ctx: DiscordAgenticContext, username: str, *, tool_name: str) -> discord.Member | None:
    query = (username or '').strip()
    if not query:
        _log_tool_call(ctx, f'{tool_name}(username={username!r}) -> Error: empty username')
        return None

    direct = next((m for m in ctx.guild.members if getattr(m, 'name', None) == query), None)
    if direct is not None:
        return direct

    lowered = query.lower()
    exact_ci = next((m for m in ctx.guild.members if str(getattr(m, 'name', '')).lower() == lowered), None)
    if exact_ci is not None:
        return exact_ci

    partial = [m for m in ctx.guild.members if lowered in str(getattr(m, 'name', '')).lower()]
    if len(partial) == 1:
        return partial[0]

    if len(partial) > 1:
        candidates = ', '.join(f'{m.name} ({m.id})' for m in partial[:5])
        suffix = ' …' if len(partial) > 5 else ''
        _log_tool_call(ctx, f'{tool_name}(username={username!r}) -> Ambiguous username; matches: {candidates}{suffix}')
        return None

    _log_tool_call(ctx, f'{tool_name}(username={username!r}) -> Username not found in guild members')
    return None


def _extract_tool_call_name_and_args(item: Any) -> tuple[str, str | None]:
    candidates: list[Any] = [item]
    raw_item = _get_attr_or_key(item, 'raw_item', None)
    if raw_item is not None:
        candidates.append(raw_item)

    function_names = (
        'name',
        'tool_name',
        'function_name',
        'call_name',
    )
    args_names = (
        'arguments',
        'args',
        'input',
        'payload',
    )

    tool_name: str | None = None
    args_text: str | None = None

    for candidate in candidates:
        func_obj = _get_attr_or_key(candidate, 'function', None)
        search_targets: list[Any] = [candidate]
        if func_obj is not None:
            search_targets.append(func_obj)

        for target in search_targets:
            for key in function_names:
                value = _get_attr_or_key(target, key, None)
                if value:
                    tool_name = str(value)
                    break
            if tool_name:
                break

        for target in search_targets:
            for key in args_names:
                value = _get_attr_or_key(target, key, None)
                if value is not None and str(value) != '':
                    args_text = _coerce_text(value)
                    break
            if args_text:
                break

        if tool_name:
            break

    if not tool_name:
        fallback = _get_attr_or_key(item, 'type', None)
        tool_name = str(fallback) if fallback else 'unknown tool'

    return tool_name, args_text


def _extract_reasoning_text(item: Any) -> str:
    if item is None:
        return ''

    chunks: list[str] = []
    for key in ('summary', 'reasoning', 'content', 'text', 'message'):
        value = _get_attr_or_key(item, key, None)
        if value:
            text = _coerce_text(value).strip()
            if text:
                chunks.append(text)

    raw_item = _get_attr_or_key(item, 'raw_item', None)
    if raw_item is not None:
        for key in ('summary', 'reasoning', 'content', 'text', 'message'):
            value = _get_attr_or_key(raw_item, key, None)
            if value:
                text = _coerce_text(value).strip()
                if text:
                    chunks.append(text)

    try:
        helper_text = ItemHelpers.text_message_output(item).strip()
    except Exception:
        helper_text = ''
    if helper_text:
        chunks.append(helper_text)

    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        normalized = _normalize_reasoning_text(chunk)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)

    return ' | '.join(deduped)


def _extract_content_text_repr(text: str) -> str:
    marker = 'Content(text='
    start = text.find(marker)
    if start < 0:
        return ''

    idx = start + len(marker)
    if idx >= len(text):
        return ''
    quote = text[idx]
    if quote not in ("'", '"'):
        return ''
    idx += 1
    out_chars: list[str] = []
    escaped = False
    while idx < len(text):
        ch = text[idx]
        if escaped:
            if ch == 'n':
                out_chars.append('\n')
            elif ch == 't':
                out_chars.append('\t')
            else:
                out_chars.append(ch)
            escaped = False
            idx += 1
            continue
        if ch == '\\':
            escaped = True
            idx += 1
            continue
        if ch == quote:
            break
        out_chars.append(ch)
        idx += 1
    return ''.join(out_chars).strip()


def _normalize_reasoning_text(text: str) -> str:
    normalized = (text or '').replace('\r\n', '\n').replace('\r', '\n').strip()
    if not normalized:
        return ''

    extracted_repr = _extract_content_text_repr(normalized)
    if extracted_repr:
        normalized = extracted_repr

    if any(token in normalized for token in ('RunItemStreamEvent(', 'ReasoningItem(', 'FunctionTool(', 'Agent(name=')):
        return ''

    normalized = re.sub(r'\n{3,}', '\n\n', normalized)
    normalized = '\n'.join(line.strip() for line in normalized.split('\n')).strip()
    return normalized


def _reasoning_fingerprint(text: str) -> str:
    normalized = _normalize_reasoning_text(text).lower()
    if not normalized:
        return ''
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized[:320]


def _merge_reasoning_text(current: str, incoming: str) -> str:
    current_clean = _normalize_reasoning_text(current)
    incoming_clean = _normalize_reasoning_text(incoming)
    if not incoming_clean:
        return current_clean
    if not current_clean:
        return incoming_clean

    if incoming_clean == current_clean:
        return current_clean
    if incoming_clean.startswith(current_clean):
        return incoming_clean
    if current_clean.startswith(incoming_clean):
        return current_clean
    if incoming_clean in current_clean:
        return current_clean
    if current_clean in incoming_clean:
        return incoming_clean

    if current_clean.endswith(incoming_clean):
        return current_clean
    if incoming_clean.endswith(current_clean):
        return incoming_clean

    return f'{current_clean}\n\n{incoming_clean}'


def _extract_reasoning_delta_from_event(event: Any) -> str:
    event_name = str(_get_attr_or_key(event, 'name', '') or '').lower()
    event_type = str(_get_attr_or_key(event, 'type', '') or '').lower()
    if 'reason' not in event_name and 'reason' not in event_type:
        return ''

    candidates = [
        _get_attr_or_key(event, 'delta', None),
        _get_attr_or_key(event, 'item', None),
        _get_attr_or_key(event, 'data', None),
        _get_attr_or_key(event, 'raw_item', None),
        event,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        text = _extract_reasoning_text(candidate)
        text = _normalize_reasoning_text(text)
        if text:
            return text
    return ''


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + '…'


def _build_embed(status: str, footer_text: str, details: str | None = None) -> discord.Embed:
    embed = discord.Embed(title=status)
    if details:
        embed.description = _truncate_text(details, 3900)
    embed.set_footer(text=footer_text)
    try:
        embed.add_field(name='Model', value=str(AGENTIC_SUMMARIZER_MODEL), inline=False)
    except Exception:
        pass
    return embed


def _format_msg(msg: discord.Message | cache.CachedDiscordMessage) -> str:
    author_name = getattr(getattr(msg, 'author', None), 'name', 'Unknown')
    content = getattr(msg, 'content', '') or ''
    message_id = getattr(msg, 'id', None)
    if message_id is None:
        return f'{author_name}: {content}'
    return f'{author_name} (message ID: {message_id}): {content}'


def _format_msg_compact(msg: discord.Message | cache.CachedDiscordMessage) -> str:
    author_name = getattr(getattr(msg, 'author', None), 'name', 'Unknown')
    content = getattr(msg, 'content', '') or ''
    message_id = getattr(msg, 'id', None)
    if message_id is None:
        return f'{author_name}: {content}'
    return f'[{message_id}] {author_name}: {content}'


def _format_msg_compact_with_channel(channel: discord.TextChannel, msg: discord.Message | cache.CachedDiscordMessage) -> str:
    base = _format_msg_compact(msg)
    return f'[{channel.name} ({channel.id})] {base}'


async def _get_cached_messages_for_channel(
    ctx: DiscordAgenticContext,
    channel: discord.TextChannel,
    *,
    fetch_if_missing: bool = True,
    fetch_limit: int = 300,
) -> list[discord.Message | cache.CachedDiscordMessage]:
    if channel.id in ctx.message_snapshots:
        cached_msgs = list(ctx.message_snapshots[channel.id])
    else:
        cached_msgs = cast(list[discord.Message | cache.CachedDiscordMessage], await cache.get_all_messages_in_channel_from_cache(channel))
        if not cached_msgs and fetch_if_missing:
            cached_msgs = cast(list[discord.Message | cache.CachedDiscordMessage], await cache.get_messages(channel, limit=fetch_limit))
            _log_tool_call(ctx, f'populate_channel_cache(channel_id={channel.id}, limit={fetch_limit}) -> {len(cached_msgs)} messages')
        ignored_user_ids = await _ignored_user_id_set(ctx.guild.id)
        cached_msgs = [m for m in cached_msgs if getattr(getattr(m, 'author', None), 'id', None) not in ignored_user_ids]
        ctx.message_snapshots[channel.id] = list(cached_msgs)

    cached_msgs.sort(key=lambda m: getattr(m, 'id', 0))
    if ctx.message_limit is not None and ctx.message_limit >= 0 and len(cached_msgs) > ctx.message_limit:
        cached_msgs = cached_msgs[-ctx.message_limit:]
    return cached_msgs


def _render_markdown_report(*, title: str, footer_text: str, prompt: str, guild: discord.Guild | None,
                            source_channel: discord.TextChannel | None, status_lines: list[str],
                            tool_log: list[str], reasoning_messages: list[str],
                            observed_tool_calls: list[str], exact_tool_calls: list[str] | None,
                            final_summary: str, warnings: list[str]) -> str:
    lines: list[str] = [
        f'# {title}',
        '',
        f'- Generated: {datetime.now().astimezone().isoformat(timespec="seconds")}',
        f'- Guild: {guild.name if guild is not None else "unknown"}',
        f'- Channel: {source_channel.name if source_channel is not None else "unknown"}',
        f'- Prompt: {prompt}',
        f'- Footer: {footer_text}',
        '',
        '## Status',
    ]
    if status_lines:
        lines.extend(f'- {line}' for line in status_lines)
    else:
        lines.append('- No status updates were recorded.')

    lines.extend(['', '## Tool call summary'])
    executed_count = len(exact_tool_calls) if exact_tool_calls else 0
    lines.append(f'- Observed in stream events: {len(observed_tool_calls)}')
    lines.append(f'- Executed in tool wrappers: {executed_count}')

    lines.extend(['', '## Stream log'])
    if tool_log:
        lines.extend(f'- {entry}' for entry in tool_log)
    else:
        lines.append('- No stream log entries were recorded.')

    lines.extend(['', '## Reasoning messages'])
    if reasoning_messages:
        lines.extend(f'- {entry}' for entry in reasoning_messages)
    else:
        lines.append('- No explicit reasoning messages were recorded.')

    lines.extend(['', '## Observed tool calls (stream events)'])
    if observed_tool_calls:
        lines.extend(f'- {entry}' for entry in observed_tool_calls)
    else:
        lines.append('- No tool calls were observed in stream events.')

    lines.extend(['', '## Executed tool calls (function wrappers)'])
    if exact_tool_calls:
        lines.extend(f'- {entry}' for entry in exact_tool_calls)
    else:
        lines.append('- No executed tool calls were recorded.')

    if warnings:
        lines.extend(['', '## Warnings'])
        lines.extend(f'- {warning}' for warning in warnings)

    lines.extend(['', '## Final summary', ''])
    lines.append(final_summary.strip() or 'No final summary text was produced.')
    lines.append('')
    return '\n'.join(lines)


async def _edit_or_followup(interaction: discord.Interaction, *, content: str = '', embed: discord.Embed | None = None,
                            file: discord.File | None = None) -> None:
    try:
        kwargs: dict[str, Any] = {'content': content}
        if embed is not None:
            kwargs['embed'] = embed
        if file is not None:
            kwargs['attachments'] = [file]
        await interaction.edit_original_response(**kwargs)
    except Exception:
        if file is not None:
            try:
                file.fp.seek(0)
            except Exception:
                pass
        kwargs = {'content': content}
        if embed is not None:
            kwargs['embed'] = embed
        if file is not None:
            kwargs['file'] = file
        await interaction.followup.send(**kwargs)

@function_tool
async def get_ignored_users(ctx: RunContextWrapper[DiscordAgenticContext]) -> list[str]:
    """Function tool to get the list of ignored users in the guild."""
    ignored_users = await ctx.context.ignored_users
    if ctx.context is not None:
        _log_tool_call(ctx.context, f'get_ignored_users() -> {len(ignored_users)} entries')
    return ignored_users

@function_tool
async def get_channel_list(ctx: RunContextWrapper[DiscordAgenticContext]) -> list[str]:
    """Function tool to get the list of text channels in the guild."""
    channels = ctx.context.guild.text_channels
    result = [f"{channel.name} ({channel.id})" for channel in channels]
    _log_tool_call(ctx.context, f'get_channel_list() -> {len(result)} channels')
    return result

@function_tool
async def get_messages_in_channel(ctx: RunContextWrapper[DiscordAgenticContext], channel_id: int, limit: int = 200, skip: int = 0) -> list[str]:
    """Function tool to get the list of messages in a specific channel by ID."""
    print(f'get_messages_in_channel: channel_id={channel_id}, limit={limit}, skip={skip}')
    channel = _resolve_text_channel(ctx.context, channel_id, tool_name='get_messages_in_channel')
    if channel is None:
        return []
    cached_msgs = await _get_cached_messages_for_channel(ctx.context, channel)

    if not cached_msgs:
        _log_tool_call(ctx.context, f'get_messages_in_channel -> 0 messages')
        return []
    # Sort by ID (chronological)
    cached_msgs.sort(key=lambda m: m.id)
    # Apply skip and limit
    if skip > 0:
        cached_msgs = cached_msgs[skip:]
    if limit is not None and len(cached_msgs) > limit:
        cached_msgs = cached_msgs[:limit]
    # Format each message as "username: content"
    result: list[str] = []
    for msg in cached_msgs:
        if msg.content:
            result.append(_format_msg(msg))
    print(f'get_messages_in_channel: returning {len(result)} messages')
    _log_tool_call(ctx.context, f'get_messages_in_channel(channel_id={channel_id}, limit={limit}, skip={skip}) -> {len(result)} messages')
    return result


@function_tool
async def fetch_recent_messages_in_channel(ctx: RunContextWrapper[DiscordAgenticContext], channel_id: int, limit: int = 300) -> list[str]:
    """Fetch the newest messages from a channel, transparently warming the cache from Discord when needed."""
    print(f'fetch_recent_messages_in_channel: channel_id={channel_id}, limit={limit}')
    channel = _resolve_text_channel(ctx.context, channel_id, tool_name='fetch_recent_messages_in_channel')
    if channel is None:
        return []

    cached_msgs = await _get_cached_messages_for_channel(ctx.context, channel, fetch_if_missing=True, fetch_limit=max(limit, 300))
    if not cached_msgs:
        _log_tool_call(ctx.context, f'fetch_recent_messages_in_channel(channel_id={channel_id}, limit={limit}) -> 0 messages')
        return []

    cached_msgs = sorted(cached_msgs, key=lambda m: m.id)
    recent = cached_msgs[-limit:] if limit > 0 else []
    result = [_format_msg_compact(msg) for msg in recent if msg.content]
    _log_tool_call(ctx.context, f'fetch_recent_messages_in_channel(channel_id={channel_id}, limit={limit}) -> {len(result)} messages')
    return result


@function_tool
async def fetch_messages_older_than(ctx: RunContextWrapper[DiscordAgenticContext], channel_id: int, before_message_id: int, limit: int = 300) -> list[str]:
    """Fetch messages older than a given message ID in a channel."""
    print(f'fetch_messages_older_than: channel_id={channel_id}, before_message_id={before_message_id}, limit={limit}')
    channel = _resolve_text_channel(ctx.context, channel_id, tool_name='fetch_messages_older_than')
    if channel is None:
        return []

    before_marker = discord.Object(id=before_message_id)
    older_msgs = cast(list[discord.Message | cache.CachedDiscordMessage], await cache.get_messages(channel=channel, limit=limit, before=before_marker))
    ignored_user_ids = await _ignored_user_id_set(ctx.context.guild.id)
    older_msgs = [m for m in older_msgs if getattr(getattr(m, 'author', None), 'id', None) not in ignored_user_ids]
    older_msgs = sorted(older_msgs, key=lambda m: m.id)
    result = [_format_msg_compact(msg) for msg in older_msgs if msg.content]
    _log_tool_call(ctx.context, f'fetch_messages_older_than(channel_id={channel_id}, before_message_id={before_message_id}, limit={limit}) -> {len(result)} messages')
    return result

@function_tool
async def search_messages_in_channel(ctx: RunContextWrapper[DiscordAgenticContext], channel_id: int, query: str, max_results: int = 100, max_tokens: int = 12000) -> list[str]:
    """Function tool to search messages in a specific channel by ID using simple substring matching."""
    print(f'search_messages_in_channel: channel_id={channel_id}, query="{query}", max_results={max_results}, max_tokens={max_tokens}')
    channel = _resolve_text_channel(ctx.context, channel_id, tool_name='search_messages_in_channel')
    if channel is None:
        return []
    cached_msgs = await _get_cached_messages_for_channel(ctx.context, channel)

    if not cached_msgs:
        print('search_messages_in_channel: no cached messages')
        return []
    # Sort by ID (chronological)
    cached_msgs.sort(key=lambda m: m.id)
    # Simple substring match for now; could be improved with embeddings or more advanced search
    matched_msgs: list[discord.Message | cache.CachedDiscordMessage] = []
    total_tokens = 0
    for msg in cached_msgs:
        if msg.content and query.lower() in msg.content.lower():
            # Estimate tokens as len(content) / 4
            estimated_tokens = max(1, int(len(msg.content) / 4))
            if total_tokens + estimated_tokens > max_tokens:
                break
            matched_msgs.append(msg)
            total_tokens += estimated_tokens
            if len(matched_msgs) >= max_results:
                break
    # Format each message as "username: content"
    result: list[str] = []
    for msg in matched_msgs:
        if msg.content:
            result.append(_format_msg(msg))
    print(f'search_messages_in_channel: returning {len(result)} messages')
    _log_tool_call(ctx.context, f'search_messages_in_channel(channel_id={channel_id}, query={query!r}, max_results={max_results}, max_tokens={max_tokens}) -> {len(result)} messages')
    return result

@function_tool
async def get_context_around_message(ctx: RunContextWrapper[DiscordAgenticContext], channel_id: int, message_id: int, radius: int = 10, max_tokens: int = 4000) -> list[str]:
    """Function tool to get the context around a specific message by ID in a specific channel by ID."""
    channel = _resolve_text_channel(ctx.context, channel_id, tool_name='get_context_around_message')
    if channel is None:
        return []
    cached_msgs = await _get_cached_messages_for_channel(ctx.context, channel)

    if not cached_msgs:
        _log_tool_call(ctx.context, f'get_messages_in_channel -> 0 messages')
        return []
    # Sort by ID (chronological)
    cached_msgs.sort(key=lambda m: m.id)
    # Find the message with the given ID
    target_index = None
    for i, msg in enumerate(cached_msgs):
        if msg.id == message_id:
            target_index = i
            break
    if target_index is None:
        return []
    # Get context messages within the radius
    start_index = max(0, target_index - radius)
    end_index = min(len(cached_msgs), target_index + radius + 1)
    context_msgs = cached_msgs[start_index:end_index]
    # Limit by max_tokens
    result: list[str] = []
    total_tokens = 0
    for msg in context_msgs:
        if msg.content:
            estimated_tokens = max(1, int(len(msg.content) / 4))
            if total_tokens + estimated_tokens > max_tokens:
                break
            result.append(_format_msg(msg))
            total_tokens += estimated_tokens
    _log_tool_call(ctx.context, f'get_context_around_message(channel_id={channel_id}, message_id={message_id}, radius={radius}, max_tokens={max_tokens}) -> {len(result)} messages')
    return result

@function_tool
async def get_server_information(ctx: RunContextWrapper[DiscordAgenticContext]) -> str:
    """Function tool to get basic information about the server."""
    guild = ctx.context.guild
    info = f"Server Name: {guild.name}\n"
    info += f"Server ID: {guild.id}\n"
    info += f"Member Count: {guild.member_count}\n"
    info += f"Owner: {guild.owner}\n"
    info += f"Created At: {guild.created_at}\n"
    _log_tool_call(ctx.context, 'get_server_information() -> 5 lines')
    return info


@function_tool
async def get_user_information(ctx: RunContextWrapper[DiscordAgenticContext], username: str) -> str:
    """Get information about a specific user in the guild by username."""
    guild = ctx.context.guild
    member = _resolve_member_by_username(ctx.context, username, tool_name='get_user_information')
    ignored_user_ids = await _ignored_user_id_set(guild.id)
    is_ignored = member.id in ignored_user_ids if member is not None else False

    if member is None:
        result = f'Username query: {username}\nStatus: not a current guild member\nIgnored: unknown'
    else:
        role_names = [r.name for r in getattr(member, 'roles', []) if getattr(r, 'name', None) and r.name != '@everyone']
        result = (
            f'User ID: {member.id}\n'
            f'Username: {member.name}\n'
            f'Display name: {member.display_name}\n'
            f'Bot: {member.bot}\n'
            f'Joined at: {getattr(member, "joined_at", None)}\n'
            f'Created at: {getattr(member, "created_at", None)}\n'
            f'Roles: {", ".join(role_names) if role_names else "none"}\n'
            f'Ignored: {is_ignored}'
        )

    _log_tool_call(ctx.context, f'get_user_information(username={username!r}) -> {"member" if member is not None else "unknown user"}')
    return result


@function_tool
async def fetch_messages_by_user(ctx: RunContextWrapper[DiscordAgenticContext], channel_id: int, username: str, limit: int = 300) -> list[str]:
    """Fetch cached or recently loaded messages from a specific username in a channel."""
    channel = _resolve_text_channel(ctx.context, channel_id, tool_name='fetch_messages_by_user')
    if channel is None:
        return []

    member = _resolve_member_by_username(ctx.context, username, tool_name='fetch_messages_by_user')
    if member is None:
        return []

    cached_msgs = await _get_cached_messages_for_channel(ctx.context, channel, fetch_if_missing=True, fetch_limit=max(limit, 300))
    filtered = [m for m in cached_msgs if getattr(getattr(m, 'author', None), 'id', None) == member.id]
    filtered = sorted(filtered, key=lambda m: m.id)
    result = [_format_msg_compact(msg) for msg in filtered[-limit:] if msg.content]
    _log_tool_call(ctx.context, f'fetch_messages_by_user(channel_id={channel_id}, username={member.name!r}, limit={limit}) -> {len(result)} messages')
    return result


@function_tool
async def search_messages_by_user(ctx: RunContextWrapper[DiscordAgenticContext], channel_id: int, username: str, query: str, max_results: int = 100, max_tokens: int = 12000) -> list[str]:
    """Search cached or recently loaded messages from a specific username in a channel."""
    channel = _resolve_text_channel(ctx.context, channel_id, tool_name='search_messages_by_user')
    if channel is None:
        return []

    member = _resolve_member_by_username(ctx.context, username, tool_name='search_messages_by_user')
    if member is None:
        return []

    cached_msgs = await _get_cached_messages_for_channel(ctx.context, channel, fetch_if_missing=True, fetch_limit=max(max_results * 2, 300))
    matched: list[discord.Message | cache.CachedDiscordMessage] = []
    total_tokens = 0
    for msg in cached_msgs:
        author_id = getattr(getattr(msg, 'author', None), 'id', None)
        if author_id != member.id:
            continue
        content = getattr(msg, 'content', '') or ''
        if query.lower() not in content.lower():
            continue
        estimated_tokens = max(1, int(len(content) / 4))
        if total_tokens + estimated_tokens > max_tokens:
            break
        matched.append(msg)
        total_tokens += estimated_tokens
        if len(matched) >= max_results:
            break

    result = [_format_msg_compact(msg) for msg in matched if msg.content]
    _log_tool_call(ctx.context, f'search_messages_by_user(channel_id={channel_id}, username={member.name!r}, query={query!r}, max_results={max_results}, max_tokens={max_tokens}) -> {len(result)} messages')
    return result


@function_tool
async def fetch_messages_by_user_across_server(
    ctx: RunContextWrapper[DiscordAgenticContext],
    username: str,
    per_channel_limit: int = 300,
    max_total: int = 2000,
) -> list[str]:
    """Fetch messages from a specific username across all text channels in the guild."""
    member = _resolve_member_by_username(ctx.context, username, tool_name='fetch_messages_by_user_across_server')
    if member is None:
        return []

    collected: list[tuple[int, str]] = []
    channels_scanned = 0
    fetch_limit = max(100, per_channel_limit)

    for channel in ctx.context.guild.text_channels:
        channels_scanned += 1
        cached_msgs = await _get_cached_messages_for_channel(ctx.context, channel, fetch_if_missing=True, fetch_limit=fetch_limit)
        if not cached_msgs:
            continue

        user_msgs = [m for m in cached_msgs if getattr(getattr(m, 'author', None), 'id', None) == member.id and getattr(m, 'content', '')]
        if not user_msgs:
            continue

        selected = user_msgs[-per_channel_limit:] if per_channel_limit > 0 else []
        for msg in selected:
            msg_id = int(getattr(msg, 'id', 0) or 0)
            collected.append((msg_id, _format_msg_compact_with_channel(channel, msg)))

    collected.sort(key=lambda pair: pair[0])
    if max_total > 0 and len(collected) > max_total:
        collected = collected[-max_total:]

    result = [entry for _, entry in collected]
    _log_tool_call(
        ctx.context,
        f'fetch_messages_by_user_across_server(username={member.name!r}, per_channel_limit={per_channel_limit}, max_total={max_total}) -> {len(result)} messages across {channels_scanned} channels'
    )
    return result


@function_tool
async def search_messages_in_server(
    ctx: RunContextWrapper[DiscordAgenticContext],
    query: str,
    max_results: int = 200,
    max_tokens: int = 20000,
    per_channel_limit: int = 500,
) -> list[str]:
    """Search messages across all text channels in the guild using substring matching."""
    if not query.strip():
        _log_tool_call(ctx.context, 'search_messages_in_server(query=<empty>) -> 0 messages')
        return []

    lowered_query = query.lower()
    channels_scanned = 0
    fetch_limit = max(100, per_channel_limit)
    candidates: list[tuple[int, str, int]] = []

    for channel in ctx.context.guild.text_channels:
        channels_scanned += 1
        cached_msgs = await _get_cached_messages_for_channel(ctx.context, channel, fetch_if_missing=True, fetch_limit=fetch_limit)
        if not cached_msgs:
            continue

        for msg in cached_msgs[-per_channel_limit:] if per_channel_limit > 0 else []:
            content = getattr(msg, 'content', '') or ''
            if not content or lowered_query not in content.lower():
                continue
            estimated_tokens = max(1, int(len(content) / 4))
            msg_id = int(getattr(msg, 'id', 0) or 0)
            candidates.append((msg_id, _format_msg_compact_with_channel(channel, msg), estimated_tokens))

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[int, str]] = []
    total_tokens = 0
    for msg_id, rendered, estimated_tokens in candidates:
        if max_results > 0 and len(selected) >= max_results:
            break
        if max_tokens > 0 and total_tokens + estimated_tokens > max_tokens:
            break
        selected.append((msg_id, rendered))
        total_tokens += estimated_tokens

    selected.sort(key=lambda item: item[0])
    result = [entry for _, entry in selected]
    _log_tool_call(
        ctx.context,
        f'search_messages_in_server(query={query!r}, max_results={max_results}, max_tokens={max_tokens}, per_channel_limit={per_channel_limit}) -> {len(result)} messages across {channels_scanned} channels'
    )
    return result


@function_tool
async def search_messages_by_user_across_server(
    ctx: RunContextWrapper[DiscordAgenticContext],
    username: str,
    query: str,
    max_results: int = 200,
    max_tokens: int = 20000,
    per_channel_limit: int = 500,
) -> list[str]:
    """Search messages from a specific username across all text channels in the guild."""
    member = _resolve_member_by_username(ctx.context, username, tool_name='search_messages_by_user_across_server')
    if member is None:
        return []

    if not query.strip():
        _log_tool_call(ctx.context, f'search_messages_by_user_across_server(username={username!r}, query=<empty>) -> 0 messages')
        return []

    lowered_query = query.lower()
    channels_scanned = 0
    fetch_limit = max(100, per_channel_limit)
    candidates: list[tuple[int, str, int]] = []

    for channel in ctx.context.guild.text_channels:
        channels_scanned += 1
        cached_msgs = await _get_cached_messages_for_channel(ctx.context, channel, fetch_if_missing=True, fetch_limit=fetch_limit)
        if not cached_msgs:
            continue

        channel_slice = cached_msgs[-per_channel_limit:] if per_channel_limit > 0 else []
        for msg in channel_slice:
            author_id = getattr(getattr(msg, 'author', None), 'id', None)
            if author_id != member.id:
                continue
            content = getattr(msg, 'content', '') or ''
            if not content or lowered_query not in content.lower():
                continue
            estimated_tokens = max(1, int(len(content) / 4))
            msg_id = int(getattr(msg, 'id', 0) or 0)
            candidates.append((msg_id, _format_msg_compact_with_channel(channel, msg), estimated_tokens))

    candidates.sort(key=lambda item: item[0], reverse=True)

    selected: list[tuple[int, str]] = []
    total_tokens = 0
    for msg_id, rendered, estimated_tokens in candidates:
        if max_results > 0 and len(selected) >= max_results:
            break
        if max_tokens > 0 and total_tokens + estimated_tokens > max_tokens:
            break
        selected.append((msg_id, rendered))
        total_tokens += estimated_tokens

    selected.sort(key=lambda item: item[0])
    result = [entry for _, entry in selected]
    _log_tool_call(
        ctx.context,
        f'search_messages_by_user_across_server(username={member.name!r}, query={query!r}, max_results={max_results}, max_tokens={max_tokens}, per_channel_limit={per_channel_limit}) -> {len(result)} messages across {channels_scanned} channels'
    )
    return result

class IgnoredUsersGuardrailOutput(BaseModel):
    blocked: bool
    reason: str | None = None

ignored_users_guardrail_agent = Agent(
    name="Ignored Users Guardrail Agent",
    instructions="Check if the user is asking about or for any users that are in the ignored users list for this guild, or if the input contains any information about any ignored users. If so, block the request. The list of ignored users can be obtained from the context." \
    " If the input does not mention or relate to any ignored users, allow the request. " \
    " Do not block the request if it is just a general question that may also apply to non-ignored users.",
    model=AGENTIC_GUARDRAIL_MODEL,
    output_type=IgnoredUsersGuardrailOutput,
    tools=[get_ignored_users],
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium"), tool_choice="get_ignored_users")
)

@input_guardrail
async def ignored_users_input_guardrail(
       ctx: RunContextWrapper[DiscordAgenticContext], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(ignored_users_guardrail_agent, input, context=ctx.context, run_config=OUR_RUN_CONFIG)
    if result.final_output.blocked:
        print(f"Input guardrail blocked input due to ignored users: {result.final_output.reason}")
    return GuardrailFunctionOutput(output_info=result.final_output, tripwire_triggered=result.final_output.blocked)
agent = Agent(
    name="Summarizer Agent",
    instructions="You are a helpful assistant that summarizes primarily technical Discord conversations. Use the functions provided to get and summarize the messages, " \
    "while ignoring any messages from or conversations with users in the ignored users list. A user specificed summarization prompt may be provided to guide the summary. " \
    "Functions are also provided to get more information regarding the server, such as server name, and available channels. If you are not provided a channel argument, " \
    "look up what channels are available, and determine the ones that are most relevant to the user prompt. \n" \
    "The tools you have access to are as follows:\n" \
    "- get_ignored_users(): Get the list of ignored users in the guild. You should use this to filter out any messages from these users when summarizing.\n" \
    "- get_channel_list(): Get the list of text channels in the guild. Use this to determine which channels to summarize from.\n" \
    "- get_user_information(username): Get information about a specific user by username. Use this when a summary depends on who a user is.\n" \
    "- fetch_recent_messages_in_channel(channel_id, limit=300): Fetch the newest messages from a channel. Use this first when you need to inspect a channel.\n" \
    "- fetch_messages_older_than(channel_id, before_message_id, limit=300): Fetch older messages before a known message ID. Use this repeatedly to page backwards through longer history when needed.\n" \
    "- fetch_messages_by_user(channel_id, username, limit=300): Fetch cached or recently loaded messages from a specific username in a channel.\n" \
    "- search_messages_by_user(channel_id, username, query, max_results=100, max_tokens=12000): Search cached or recently loaded messages from a specific username.\n" \
    "- fetch_messages_by_user_across_server(username, per_channel_limit=300, max_total=2000): Fetch messages from a specific username across all text channels.\n" \
    "- search_messages_by_user_across_server(username, query, max_results=200, max_tokens=20000, per_channel_limit=500): Search messages from a specific username across all text channels for a query/topic.\n" \
    "- search_messages_in_server(query, max_results=200, max_tokens=20000, per_channel_limit=500): Search messages across all text channels in the server for a topic/query.\n" \
    "- get_messages_in_channel(channel_id, limit=200, skip=0): Get a slice of messages from the cached/loaded channel history.\n" \
    "- search_messages_in_channel(channel_id, query, max_results=100, max_tokens=12000): Search cached/loaded messages in a specific channel by simple substring matching.\n" \
    "- get_context_around_message(channel_id, message_id, radius=10, max_tokens=4000): Get the context around a specific message by ID in a specific channel by ID.\n" \
    "- get_server_information(): Get basic information about the server.\n" \
    "When summarizing, ensure that you do not include any information from ignored users. Prefer usernames over user IDs when selecting user-focused tools. Do not assume the answer is limited to the invocation channel: default to checking multiple relevant channels unless the user explicitly asks for a single channel only. " \
    "If a channel appears empty from cache, fetch recent messages from it first, then page older messages if needed. For direct questions about a topic or a non-ignored person, expand search depth by paging through older messages (multiple pages) and checking other likely channels before concluding. Prefer server-wide tools for cross-channel topic/person requests and prefer exact message retrieval before searching. Always aim to provide a concise and informative summary that captures the key points of the conversation.",
    model=AGENTIC_SUMMARIZER_MODEL,
    input_guardrails=[ignored_users_input_guardrail],
    tools=[get_ignored_users, get_channel_list, get_user_information, fetch_recent_messages_in_channel, fetch_messages_older_than, fetch_messages_by_user, search_messages_by_user, fetch_messages_by_user_across_server, search_messages_by_user_across_server, search_messages_in_server, get_messages_in_channel, search_messages_in_channel, get_context_around_message, get_server_information],
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium"))
)

def create_agentic_context(guild: discord.Guild) -> DiscordAgenticContext:
    return DiscordAgenticContext(guild=guild)


async def create_agentic_summary(interaction: discord.Interaction, summarize_prompt: str, footer_text: str, source_channel: discord.TextChannel | None = None, count_msgs: int | None = None):
    """Run the agentic summarizer and update the Discord interaction with progress and final embed.

    This mirrors the UX in `summarizer.llm.run_llm`: show a 'reasoning' title while running,
    periodically update the message (best-effort), then replace with final summary.

    Assumptions and graceful handling:
    - Runner.run may be sync or async; we handle both.
    - result.final_output may be a plain string, a pydantic model with `summary`, or another object; we coerce to text.
    - The agent library may not stream tokens; if it does, this function will still work but won't stream token-by-token.
    """

    # Build agentic context and input
    guild = getattr(interaction, 'guild', None)
    if guild is None:
        err_embed = discord.Embed(title='Agentic summarizer error')
        err_embed.description = 'This command can only run inside a server.'
        err_embed.set_footer(text=footer_text)
        report_text = _render_markdown_report(
            title='Agentic summary report',
            footer_text=footer_text,
            prompt=summarize_prompt,
            guild=None,
            source_channel=source_channel,
            status_lines=['Rejected: command was used outside a server.'],
            tool_log=[],
            reasoning_messages=[],
            observed_tool_calls=[],
            exact_tool_calls=[],
            final_summary='This command can only run inside a server.',
            warnings=['Missing guild context.'],
        )
        await _edit_or_followup(
            interaction,
            embed=err_embed,
            content='',
            file=discord.File(fp=io.BytesIO(report_text.encode('utf-8')), filename='agentic_summary_report.md')
        )
        return

    agentic_ctx = create_agentic_context(guild)
    agentic_ctx.message_limit = count_msgs
    if source_channel is not None:
        agentic_ctx.source_channel_id = source_channel.id
        agentic_ctx.source_channel_name = source_channel.name

    status_lines: list[str] = []
    tool_log: list[str] = []
    reasoning_log: list[str] = []
    reasoning_messages: list[str] = []
    observed_tool_calls: list[str] = []
    stream_event_log: list[str] = []
    live_reasoning_current = ''
    warnings: list[str] = []
    phase = 'initialising'

    def _push_status(line: str) -> None:
        status_lines.append(line)
        if len(status_lines) > 12:
            del status_lines[0 : len(status_lines) - 12]

    def _push_tool_log(entry: str) -> None:
        tool_log.append(entry)
        if len(tool_log) > 120:
            del tool_log[0 : len(tool_log) - 120]

    def _push_reasoning_log(entry: str) -> None:
        cleaned = _normalize_reasoning_text(entry)
        if not cleaned:
            return
        fingerprint = _reasoning_fingerprint(cleaned)
        recent_fingerprints = {_reasoning_fingerprint(line) for line in reasoning_log[-6:]}
        if fingerprint in recent_fingerprints:
            return
        reasoning_log.append(cleaned)
        if len(reasoning_log) > 120:
            del reasoning_log[0 : len(reasoning_log) - 120]

    def _set_live_reasoning(entry: str) -> None:
        nonlocal live_reasoning_current
        merged = _merge_reasoning_text(live_reasoning_current, entry)
        if not merged:
            return
        if _reasoning_fingerprint(merged) == _reasoning_fingerprint(live_reasoning_current):
            return
        live_reasoning_current = merged

    def _push_reasoning_message(entry: str) -> None:
        cleaned = _normalize_reasoning_text(entry)
        if not cleaned:
            return
        fingerprint = _reasoning_fingerprint(cleaned)
        recent_fingerprints = {_reasoning_fingerprint(line) for line in reasoning_messages[-10:]}
        if fingerprint in recent_fingerprints:
            return
        reasoning_messages.append(cleaned)
        if len(reasoning_messages) > 200:
            del reasoning_messages[0 : len(reasoning_messages) - 200]

    def _push_observed_tool_call(entry: str) -> None:
        cleaned = entry.strip()
        if not cleaned:
            return
        observed_tool_calls.append(cleaned)
        if len(observed_tool_calls) > 200:
            del observed_tool_calls[0 : len(observed_tool_calls) - 200]

    def _push_stream_event(entry: str) -> None:
        cleaned = entry.strip()
        if not cleaned:
            return
        stream_event_log.append(cleaned)
        if len(stream_event_log) > 200:
            del stream_event_log[0 : len(stream_event_log) - 200]

    def _title() -> str:
        if phase == 'initialising':
            return 'Agentic summarizer is starting'
        if phase == 'streaming':
            return 'Agentic summarizer is reasoning'
        if phase == 'finalising':
            return 'Agentic summarizer is writing the report'
        return 'Summary'

    last_stream_event_at = datetime.now()
    stream_event_count = 0

    def _build_stream_preview(started_at: datetime) -> str:
        elapsed_seconds = max(0, int((datetime.now() - started_at).total_seconds()))
        since_last_event_seconds = max(0, int((datetime.now() - last_stream_event_at).total_seconds()))
        preview_lines: list[str] = []
        if status_lines:
            preview_lines.append('**Status**')
            for line in status_lines[-4:]:
                preview_lines.append(f'• {_truncate_text(line, 220)}')

        if reasoning_log:
            if preview_lines:
                preview_lines.append('')
            preview_lines.append('**Live reasoning**')
            for line in reasoning_log[-3:]:
                preview_lines.append(f'• {_truncate_text(line, 220)}')
        elif live_reasoning_current:
            if preview_lines:
                preview_lines.append('')
            preview_lines.append('**Live reasoning**')
            reasoning_lines = [line.strip() for line in live_reasoning_current.split('\n') if line.strip()]
            for line in reasoning_lines[-5:]:
                if line.strip():
                    preview_lines.append(f'• {_truncate_text(line.strip(), 220)}')

        if preview_lines:
            preview_lines.append('')
        if since_last_event_seconds >= 8:
            preview_lines.append(f'Still working… no new stream event for {since_last_event_seconds}s')
        preview_lines.append(f'Stream events observed: {stream_event_count}')
        preview_lines.append(f'Last stream event: {since_last_event_seconds}s ago')
        preview_lines.append(f'Elapsed: {elapsed_seconds}s')
        return '\n'.join(preview_lines)

    # Prepare embed similarly to the non-agentic flow
    _push_status('Initialising agent context and guardrails.')
    embed = _build_embed(_title(), footer_text, '\n'.join(status_lines))

    # Initial post so the user sees something right away. Best-effort edit; caller should have deferred earlier.
    await _edit_or_followup(interaction, embed=embed, content='')

    # Run the agent in streaming mode using Runner.run_streamed so we can surface
    # semantic events (RunItemStreamEvent) to the user as they occur.
    prompt = summarize_prompt
    if source_channel is not None:
        prompt = (
            f"Channel {source_channel.name} (ID: {source_channel.id}): {summarize_prompt}\n"
            f"For any channel-scoped tool call, prefer this exact channel ID {source_channel.id} unless explicitly asked to use another channel. "
            f"Still check other relevant channels by default when the question is broader than a single channel, especially for topic or person-specific questions."
        )
    else:
        prompt = f"Guild {guild.name if guild is not None else 'unknown'}: {summarize_prompt}"
    if count_msgs is not None:
        prompt += f"\nLimit the search to the most relevant {count_msgs} cached message(s) unless more context is required."
    _push_status('If the cache is empty, the tools will fall back to Discord history.')
    try:
        phase = 'streaming'
        _push_status('Submitting prompt to the agent and waiting for tool calls.')
        embed = _build_embed(_title(), footer_text, '\n'.join(status_lines))
        await _edit_or_followup(interaction, embed=embed, content='')
        streamed_result = Runner.run_streamed(agent, input=prompt, context=agentic_ctx, run_config=OUR_RUN_CONFIG, max_turns=50)
    except Exception as e:
        err_embed = discord.Embed(title='Agentic summarizer error')
        err_embed.description = f'An error occurred while running the agentic summarizer: {e}'
        err_embed.set_footer(text=footer_text)
        warnings.append(f'Runtime error while starting the run: {e}')
        report_text = _render_markdown_report(
            title='Agentic summary report',
            footer_text=footer_text,
            prompt=prompt,
            guild=guild,
            source_channel=source_channel,
            status_lines=status_lines,
            tool_log=tool_log,
            reasoning_messages=reasoning_messages,
            observed_tool_calls=observed_tool_calls,
            exact_tool_calls=list(agentic_ctx.tool_calls),
            final_summary=f'Error while starting the run: {e}',
            warnings=warnings,
        )
        await _edit_or_followup(
            interaction,
            embed=err_embed,
            content='',
            file=discord.File(fp=io.BytesIO(report_text.encode('utf-8')), filename='agentic_summary_report.md')
        )
        return

    final_text = ''
    last_edit = None
    stream_started_at = datetime.now()
    stream_finished = False

    async def _heartbeat_updater() -> None:
        nonlocal last_edit
        while not stream_finished:
            try:
                embed = _build_embed(_title(), footer_text, _build_stream_preview(stream_started_at))
                await _edit_or_followup(interaction, embed=embed, content='')
                last_edit = datetime.now()
            except Exception:
                pass
            await asyncio.sleep(3)

    heartbeat_task = asyncio.create_task(_heartbeat_updater())

    # Stream semantic events and update the embed periodically
    try:
        async for event in streamed_result.stream_events():
            event_type = getattr(event, 'type', 'unknown')
            name = getattr(event, 'name', '')
            item = getattr(event, 'item', None)
            print(f'Event: {event_type} / {name} / {type(item)}')
            stream_event_count += 1
            last_stream_event_at = datetime.now()
            event_label = f'{event_type}:{name}' if name else str(event_type)
            _push_stream_event(event_label)

            event_reasoning_delta = _extract_reasoning_delta_from_event(event)
            if event_reasoning_delta:
                _set_live_reasoning(event_reasoning_delta)
                _push_reasoning_log(live_reasoning_current)

            if event_type == 'run_item_stream_event':
                if name == 'message_output_created':
                    try:
                        text = ItemHelpers.text_message_output(cast(Any, item))
                    except Exception:
                        text = _coerce_text(item)
                    if text:
                        final_text += text
                        _push_status('Assistant message received.')
                elif name in ('tool_output', 'tool_called'):
                    tool_name, tool_args = _extract_tool_call_name_and_args(item)
                    tool_output_text = _coerce_text(_get_attr_or_key(item, 'output', None))
                    observed_entry = f'{name}: {tool_name}'
                    if tool_args:
                        observed_entry += f' args={_truncate_text(tool_args, 260)}'
                    if tool_output_text:
                        observed_entry += f' output={_truncate_text(tool_output_text, 260)}'
                    _push_observed_tool_call(observed_entry)
                    _push_tool_log(observed_entry)
                    _push_status(f'Tool used: {tool_name}')
                elif name == 'reasoning_item_created':
                    reasoning_text = _extract_reasoning_text(item)
                    if reasoning_text:
                        _set_live_reasoning(reasoning_text)
                        _push_reasoning_message(live_reasoning_current)
                        _push_reasoning_log(live_reasoning_current)
                    _push_status('Reasoning step recorded.')
                else:
                    _push_tool_log(f'{event_type}:{name}')

            preview = _build_stream_preview(stream_started_at)
            embed = _build_embed(_title(), footer_text, preview)

            if last_edit is None or (datetime.now() - last_edit).total_seconds() > 2:
                try:
                    await _edit_or_followup(interaction, embed=embed, content='')
                    last_edit = datetime.now()
                except Exception:
                    pass

        # After streaming completes, retrieve the final output
        result = streamed_result
    except Exception as e:
        err_embed = discord.Embed(title='Agentic summarizer error')
        err_embed.description = f'An error occurred while running the agentic summarizer: {e}'
        err_embed.set_footer(text=footer_text)
        warnings.append(f'Runtime error while streaming the run: {e}')
        report_text = _render_markdown_report(
            title='Agentic summary report',
            footer_text=footer_text,
            prompt=prompt,
            guild=guild,
            source_channel=source_channel,
            status_lines=status_lines,
            tool_log=tool_log,
            reasoning_messages=reasoning_messages,
            observed_tool_calls=observed_tool_calls,
            exact_tool_calls=list(agentic_ctx.tool_calls),
            final_summary=f'Error while streaming the run: {e}',
            warnings=warnings,
        )
        await _edit_or_followup(
            interaction,
            embed=err_embed,
            content='',
            file=discord.File(fp=io.BytesIO(report_text.encode('utf-8')), filename='agentic_summary_report.md')
        )
        return
    finally:
        stream_finished = True
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    # Prepare final embed
    phase = 'finalising'
    _push_status('Collecting the final output and writing the markdown report.')

    final_output = getattr(result, 'final_output', None)
    if final_output is not None:
        extracted = _coerce_text(final_output).strip()
        if extracted:
            final_text = extracted
    final_text = final_text.strip()
    if not final_text:
        warnings.append('The agent returned no textual final output.')
        final_text = 'Agent returned no content.'

    report_text = _render_markdown_report(
        title='Agentic summary report',
        footer_text=footer_text,
        prompt=prompt,
        guild=guild,
        source_channel=source_channel,
        status_lines=status_lines,
        tool_log=tool_log,
        reasoning_messages=reasoning_messages,
        observed_tool_calls=observed_tool_calls,
        exact_tool_calls=list(agentic_ctx.tool_calls),
        final_summary=final_text,
        warnings=warnings,
    )

    final_embed = _build_embed('Summary', footer_text, final_text if len(final_text) < 3900 else 'The full summary is attached in the markdown report.')
    report_file = discord.File(fp=io.BytesIO(report_text.encode('utf-8')), filename='agentic_summary_report.md')
    await _edit_or_followup(interaction, embed=final_embed, content='', file=report_file)
