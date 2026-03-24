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
from datetime import datetime
from pprint import pprint
from collections import defaultdict

class OurModelProvider(ModelProvider):
    def get_model(self, model_name: str):
        return OpenAIChatCompletionsModel(model=model_name, openai_client=ai_client)

set_tracing_disabled(True)

OUR_RUN_CONFIG = RunConfig(
    model_provider=OurModelProvider()
)

@dataclass
class DiscordAgenticContext:
    guild: discord.Guild
    message_snapshots: dict[int, list[discord.Message | cache.CachedDiscordMessage]] = field(default_factory=lambda: defaultdict(list))  # channel_id -> list of messages

    @property
    async def ignored_users(self) -> list[str]:
        ignored_users = await cache.list_ignored_users(self.guild.id)
        # Transform into list of strings with the format "Display name: username (user ID)"
        result: list[str] = []
        for user_id in ignored_users:
            member = self.guild.get_member(user_id['user_id'])
            if member is not None:
                result.append(f"{member.display_name}: {member.name} ({member.id})")
            else:
                result.append(f"Unknown User ({user_id['user_id']})")
        return result


async def _ignored_user_id_set(guild_id: int | None) -> set[int]:
    ignored_rows = await cache.list_ignored_users(guild_id)
    return {int(row['user_id']) for row in ignored_rows if row.get('user_id') is not None}

@function_tool
async def get_ignored_users(ctx: RunContextWrapper[DiscordAgenticContext]) -> list[str]:
    """Function tool to get the list of ignored users in the guild."""
    if ctx.context is None:
        return []
    ignored_users = await ctx.context.ignored_users
    return ignored_users

@function_tool
async def get_channel_list(ctx: RunContextWrapper[DiscordAgenticContext]) -> list[str]:
    """Function tool to get the list of text channels in the guild."""
    if ctx.context is None or ctx.context.guild is None:
        return []
    channels = ctx.context.guild.text_channels
    return [f"{channel.name} ({channel.id})" for channel in channels]

@function_tool
async def get_messages_in_channel(ctx: RunContextWrapper[DiscordAgenticContext], channel_id: int, limit: int = 50, skip: int = 0) -> list[str]:
    """Function tool to get the list of messages in a specific channel by ID."""
    print(f'get_messages_in_channel: channel_id={channel_id}, limit={limit}, skip={skip}')
    if ctx.context is None or ctx.context.guild is None:
        return []
    channel = ctx.context.guild.get_channel(channel_id)
    if channel is None or not isinstance(channel, discord.TextChannel):
        return []
    # Fetch messages from cache and save snapshot in context in case messages get added later, as they would throw off the limit/skip
    if ctx.context.message_snapshots is None:
        ctx.context.message_snapshots = {}
    if channel.id in ctx.context.message_snapshots:
        cached_msgs = ctx.context.message_snapshots[channel.id]
    else:
        cached_msgs = await cache.get_all_messages_in_channel_from_cache(channel)
        # Filter out all messages from ignored people
        ignored_user_ids = await _ignored_user_id_set(ctx.context.guild.id)
        cached_msgs = [m for m in cached_msgs if m.author.id not in ignored_user_ids]
        ctx.context.message_snapshots[channel.id] = cached_msgs

    if not cached_msgs:
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
        if msg.content is not None:
            result.append(f"{msg.author.name}: {msg.content}")
    print(f'get_messages_in_channel: returning {len(result)} messages')
    return result

@function_tool
async def search_messages_in_channel(ctx: RunContextWrapper[DiscordAgenticContext], channel_id: int, query: str, max_results: int = 20, max_tokens: int = 1000) -> list[str]:
    """Function tool to search messages in a specific channel by ID using simple substring matching."""
    print(f'search_messages_in_channel: channel_id={channel_id}, query="{query}", max_results={max_results}, max_tokens={max_tokens}')
    if ctx.context is None or ctx.context.guild is None:
        return []
    channel = ctx.context.guild.get_channel(channel_id)
    if channel is None or not isinstance(channel, discord.TextChannel):
        return []
    # Fetch messages from cache and save snapshot in context in case messages get added later, as they would throw off the limit/skip
    if ctx.context.message_snapshots is None:
        ctx.context.message_snapshots = {}
    if channel.id in ctx.context.message_snapshots:
        cached_msgs = ctx.context.message_snapshots[channel.id]
    else:
        cached_msgs = await cache.get_all_messages_in_channel_from_cache(channel)
        # Filter out all messages from ignored people
        ignored_user_ids = await _ignored_user_id_set(ctx.context.guild.id)
        cached_msgs = [m for m in cached_msgs if m.author.id not in ignored_user_ids]
        ctx.context.message_snapshots[channel.id] = cached_msgs

    if not cached_msgs:
        print('search_messages_in_channel: no cached messages')
        return []
    # Sort by ID (chronological)
    cached_msgs.sort(key=lambda m: m.id)
    # Simple substring match for now; could be improved with embeddings or more advanced search
    matched_msgs: list[discord.Message | cache.CachedDiscordMessage] = []
    total_tokens = 0
    for msg in cached_msgs:
        if msg.content is not None and query.lower() in msg.content.lower():
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
        if msg.content is not None:
            result.append(f"{msg.author.name} (message ID: {msg.id}): {msg.content}")
    print(f'search_messages_in_channel: returning {len(result)} messages')
    return result

@function_tool
async def get_context_around_message(ctx: RunContextWrapper[DiscordAgenticContext], channel_id: int, message_id: int, radius: int = 5, max_tokens: int = 1000) -> list[str]:
    """Function tool to get the context around a specific message by ID in a specific channel by ID."""
    if ctx.context is None or ctx.context.guild is None:
        return []
    channel = ctx.context.guild.get_channel(channel_id)
    if channel is None or not isinstance(channel, discord.TextChannel):
        return []
    # Fetch messages from cache and save snapshot in context in case messages get added later, as they would throw off the limit/skip
    if ctx.context.message_snapshots is None:
        ctx.context.message_snapshots = {}
    if channel.id in ctx.context.message_snapshots:
        cached_msgs = ctx.context.message_snapshots[channel.id]
    else:
        cached_msgs = await cache.get_all_messages_in_channel_from_cache(channel)
        # Filter out all messages from ignored people
        ignored_user_ids = await _ignored_user_id_set(ctx.context.guild.id)
        cached_msgs = [m for m in cached_msgs if m.author.id not in ignored_user_ids]
        ctx.context.message_snapshots[channel.id] = cached_msgs

    if not cached_msgs:
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
        if msg.content is not None:
            estimated_tokens = max(1, int(len(msg.content) / 4))
            if total_tokens + estimated_tokens > max_tokens:
                break
            result.append(f"{msg.author.name} (message ID: {msg.id}): {msg.content}")
            total_tokens += estimated_tokens
    return result

@function_tool
async def get_server_information(ctx: RunContextWrapper[DiscordAgenticContext]) -> str:
    """Function tool to get basic information about the server."""
    if ctx.context is None or ctx.context.guild is None:
        return "No guild information available."
    guild = ctx.context.guild
    info = f"Server Name: {guild.name}\n"
    info += f"Server ID: {guild.id}\n"
    info += f"Member Count: {guild.member_count}\n"
    info += f"Owner: {guild.owner}\n"
    info += f"Created At: {guild.created_at}\n"
    return info

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

@output_guardrail
async def ignored_users_output_guardrail(
       ctx: RunContextWrapper[DiscordAgenticContext], agent: Agent, output: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(ignored_users_guardrail_agent, output, context=ctx.context, run_config=OUR_RUN_CONFIG)
    if result.final_output.blocked:
        print(f"Output guardrail blocked output due to ignored users: {result.final_output.reason}")
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
    "- get_messages_in_channel(channel_id, limit=50, skip=0): Get the list of messages in a specific channel by ID. You can specify how many messages to retrieve and how many to skip.\n" \
    "- search_messages_in_channel(channel_id, query, max_results=20, max_tokens=1000): Search messages in a specific channel by ID using simple substring matching. You can specify the maximum number of results and the maximum number of tokens to return.\n" \
    "- get_context_around_message(channel_id, message_id, radius=5, max_tokens=1000): Get the context around a specific message by ID in a specific channel by ID. You can specify the radius (number of messages before and after) and the maximum number of tokens to return.\n" \
    "- get_server_information(): Get basic information about the server.\n" \
    "When summarizing, ensure that you do not include any information from ignored users. If the user prompt is vague, use the server information and channel list to " \
    "determine the most relevant channels to summarize from. Always aim to provide a concise and informative summary that captures the key points of the conversation.",
    model=AGENTIC_SUMMARIZER_MODEL,
    input_guardrails=[ignored_users_input_guardrail],
    output_guardrails=[ignored_users_output_guardrail],
    tools=[get_ignored_users, get_channel_list, get_messages_in_channel, search_messages_in_channel, get_context_around_message, get_server_information],
    model_settings=ModelSettings(reasoning=Reasoning(effort="medium"))
)

def create_agentic_context(guild: discord.Guild) -> DiscordAgenticContext:
    return DiscordAgenticContext(guild=guild)


async def create_agentic_summary(interaction: discord.Interaction, summarize_prompt: str, footer_text: str, source_channel: discord.TextChannel | None = None):
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
    agentic_ctx = create_agentic_context(guild)

    # Prepare embed similarly to the non-agentic flow
    embed = discord.Embed(title='Agentic summarizer is reasoning...')
    embed.set_footer(text=footer_text)
    try:
        embed.add_field(name='LLM Info', value=f'Model: {AGENTIC_SUMMARIZER_MODEL}')
    except Exception:
        pass

    # Initial post so the user sees something right away. Best-effort edit; caller should have deferred earlier.
    try:
        embed.description = 'Running agentic summarizer...'
        await interaction.edit_original_response(embed=embed, content='')
    except Exception:
        # If edit fails, ignore — we'll try to send final result later
        pass

    # Helper to coerce chunk-like objects into text
    def _text_from_obj(obj) -> str:
        if obj is None:
            return ''
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            # common keys
            for k in ('content', 'text', 'summary'):
                if k in obj and obj[k] is not None:
                    return str(obj[k])
            return str(obj)
        # try attributes
        for attr in ('content', 'text', 'summary'):
            try:
                v = getattr(obj, attr, None)
            except Exception:
                v = None
            if v is not None:
                return str(v)
        return str(obj)

    # Run the agent in streaming mode using Runner.run_streamed so we can surface
    # semantic events (RunItemStreamEvent) to the user as they occur.
    prompt = summarize_prompt
    if source_channel is not None:
        prompt = f"Channel {source_channel.name}: {summarize_prompt}"
    else:
        prompt = f"Guild {guild.name if guild is not None else 'unknown'}: {summarize_prompt}"
    try:
        streamed_result = Runner.run_streamed(agent, input=prompt, context=agentic_ctx, run_config=OUR_RUN_CONFIG, max_turns=50)
    except Exception as e:
        err_embed = discord.Embed(title='Agentic summarizer error')
        err_embed.description = f'An error occurred while running the agentic summarizer: {e}'
        err_embed.set_footer(text=footer_text)
        try:
            await interaction.edit_original_response(embed=err_embed, content='')
        except Exception:
            await interaction.followup.send(embed=err_embed)
        return

    summary = ''
    # We hide the reasoning and tool usages afterwards.
    only_text_summary = ''
    last_edit = None

    # Stream semantic events and update the embed periodically
    try:
        async for event in streamed_result.stream_events():
            if event.type == 'run_item_stream_event':
                # pprint(event)
                name = event.name
                item = event.item

                print(f'Event: {name} / {type(item)}')

                # For message outputs and tool outputs, append textual content
                if name == 'message_output_created':
                    # item is a RunItem representing a new assistant message
                    text = ItemHelpers.text_message_output(item)
                    if text:
                        summary += text
                        only_text_summary += text
                elif name == 'tool_output' or name == 'tool_called':
                    # pprint(event)
                    # Tool outputs may contain text we want to surface
                    text = item.output if hasattr(item, 'output') else None
                    if text:
                        # summary += f"\n[tool] {text}\n"
                        summary += f"[Used tool {getattr(item, 'type', 'unknown')}]\n"
                        print(f'[tool] {text}\n')
                elif name == 'reasoning_item_created':
                    # reasoning items may contain summaries or text blocks
                    text = ItemHelpers.text_message_output(item)
                    if text:
                        # summary += f"\n[reasoning] {text}\n"
                        summary += f"[Reasoning step]\n"
                        print(f'[reasoning] {text}\n')

                # Update embed description and edit periodically
                if len(summary) > 4096:
                    embed.description = 'Response is getting too long, please wait for it to be done...'
                else:
                    embed.description = summary

                if last_edit is None or (datetime.now() - last_edit).total_seconds() > 2:
                    try:
                        await interaction.edit_original_response(embed=embed, content='')
                        last_edit = datetime.now()
                    except Exception:
                        # Ignore transient edit failures
                        pass

            # We ignore other event types for live updates

        # After streaming completes, retrieve the final output
        result = streamed_result
    except Exception as e:
        err_embed = discord.Embed(title='Agentic summarizer error')
        err_embed.description = f'An error occurred while running the agentic summarizer: {e}'
        err_embed.set_footer(text=footer_text)
        try:
            await interaction.edit_original_response(embed=err_embed, content='')
        except Exception:
            await interaction.followup.send(embed=err_embed)
        return

    # Prepare final embed
    final_embed = discord.Embed(title='Summary')
    final_embed.set_footer(text=footer_text)
    try:
        final_embed.add_field(name='LLM Info', value=f'Model: {AGENTIC_SUMMARIZER_MODEL}')
    except Exception:
        pass

    if only_text_summary is None:
        final_embed.description = 'Agent returned no content.'
        try:
            await interaction.edit_original_response(embed=final_embed, content='')
        except Exception:
            await interaction.followup.send(embed=final_embed)
        return

    # If too long for embed, attach as a file
    if len(only_text_summary) >= 4096:
        f = io.StringIO(only_text_summary)
        try:
            await interaction.edit_original_response(embed=final_embed, attachments=[discord.File(fp=f, filename='agentic_summary.txt')])
        except Exception:
            # fallback to followup
            await interaction.followup.send(embed=final_embed, file=discord.File(fp=f, filename='agentic_summary.txt'))
    else:
        final_embed.description = only_text_summary
        try:
            await interaction.edit_original_response(embed=final_embed, content='')
        except Exception:
            await interaction.followup.send(embed=final_embed)
