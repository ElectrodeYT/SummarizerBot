import os
from openai import AsyncOpenAI
from pprint import pprint

import discord
from discord import app_commands
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from summarizer.llm import create_summary, create_topic_summary
from summarizer.embeddings import create_search_embeddings
import summarizer.cache as cache
from summarizer.cache import add_ignored_user, remove_ignored_user, list_ignored_users

discord_token = os.environ['DISCORD_TOKEN']
default_summary_prompt = 'Summarize the conversation(s). If there are several conversations, summarize them individually.'

OPENAPI_TOKEN = os.environ['OPENAI_API_KEY']

ai_client = AsyncOpenAI(
    base_url=os.environ.get('OPENAI_API_BASE'),
    api_key=OPENAPI_TOKEN
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

    async def on_message(self, message: discord.Message) -> None:
        # Ignore messages without a channel (safety) or from webhooks
        try:
            if message.channel is None:
                return
        except Exception:
            return

        # Ignore ephemeral messages
        if message.flags.ephemeral:
            return

        # Best-effort: commit message to cache. Do not block on failures.
        try:
            await cache.commit_single_message_to_cache(message)
        except Exception:
            # swallow exceptions from caching to avoid disrupting bot operation
            return

    async def on_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        # Best-effort: update cached message content if present in DB
        try:
            await cache.update_message_in_cache(after)
        except Exception:
            # swallow exceptions to avoid disrupting bot runtime
            return

    async def on_raw_message_delete(self, payload) -> None:
        # Handle deletions where only the message ID is available (e.g., messages older than cache or partial state).
        try:
            # payload has attribute message_id
            msg_id = getattr(payload, 'message_id', None)
            if msg_id is None:
                return
            await cache.delete_message_from_cache(msg_id)
        except Exception:
            # best-effort: don't let exceptions propagate
            return

    async def on_raw_bulk_message_delete(self, payload) -> None:
        # Best-effort: handle raw bulk delete events where only message IDs are provided
        try:
            msg_ids = getattr(payload, 'message_ids', None)
            if not msg_ids:
                return
            for mid in msg_ids:
                try:
                    await cache.delete_message_from_cache(mid)
                except Exception:
                    continue
        except Exception:
            return


intents = discord.Intents.default()
intents.message_content = True
intents.members = True
client = DiscordClient(intents=intents)


@client.tree.command()
async def summarize(interaction: discord.Interaction, count_msgs: int | None = None, channel: discord.TextChannel = None,
                    summarize_prompt: str = default_summary_prompt) -> None:
    """
    Summarize messages in a channel.

    Args:
        interaction (discord.Interaction): The interaction object.
        count_msgs (int | None): Number of messages to summarize. If None, summarize all cached messages for the channel.
        channel (discord.TextChannel | None): The channel to summarize messages from. If None, use the channel where the command was invoked.
        summarize_prompt (str): The prompt to use for summarization.
    """
    # If count_msgs is omitted, we will summarize all messages currently cached for this channel.
    await interaction.response.send_message('Doing stuff, might take a (long) while...')

    try:
        if channel is None:
            channel = interaction.channel

        await interaction.edit_original_response(
            content='Doing stuff, might take a (long) while... (compiling messages)')

        # Get the messages. If no count was provided, default to all cached messages for this channel.
        if count_msgs is None:
            discord_messages = await cache.get_all_messages_in_channel_from_cache(channel)
        else:
            discord_messages = await cache.get_messages(channel=channel, limit=count_msgs)
        footer_text = f'Summary of the last {len(discord_messages)} messages in {channel.name} | Summary prompt: {summarize_prompt}'

        await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (Firing up AI)')
        await create_summary(interaction, discord_messages, summarize_prompt, footer_text, source_channel=channel)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.command(description='Add a user to the ignore list for this guild')
async def ignore_user(interaction: discord.Interaction, user: discord.User) -> None:
    await interaction.response.send_message('Updating ignore list...', ephemeral=True)
    try:
        # Only allow adding other users if the invoking user is a server admin. Users may ignore themselves.
        def _is_invoker_admin():
            guild = getattr(interaction, 'guild', None)
            if guild is None:
                return False
            try:
                member = guild.get_member(interaction.user.id)
            except Exception:
                member = None
            if member is None:
                return False
            try:
                if getattr(guild, 'owner_id', None) == member.id:
                    return True
                perms = getattr(member, 'guild_permissions', None)
                if perms is not None and (perms.administrator or perms.manage_guild):
                    return True
            except Exception:
                pass
            return False

        guild = getattr(interaction, 'guild', None)
        guild_id = getattr(guild, 'id', None)
        # If target is not the invoker, enforce admin-only
        if user.id != interaction.user.id and not _is_invoker_admin():
            await interaction.edit_original_response(content="You don't have permission to ignore other users; you may only ignore yourself.")
            return

        await cache.add_ignored_user(user.id, guild_id, getattr(user, 'name', None))
        await interaction.edit_original_response(content=f'User {user.name} ({user.id}) added to ignore list for this server.')
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.command(description='Remove a user from the ignore list for this guild')
async def unignore_user(interaction: discord.Interaction, user: discord.User) -> None:
    await interaction.response.send_message('Updating ignore list...', ephemeral=True)
    try:
        # Only allow removing other users if the invoking user is a server admin. Users may unignore themselves.
        def _is_invoker_admin():
            guild = getattr(interaction, 'guild', None)
            if guild is None:
                return False
            try:
                member = guild.get_member(interaction.user.id)
            except Exception:
                member = None
            if member is None:
                return False
            try:
                if getattr(guild, 'owner_id', None) == member.id:
                    return True
                perms = getattr(member, 'guild_permissions', None)
                if perms is not None and (perms.administrator or perms.manage_guild):
                    return True
            except Exception:
                pass
            return False

        guild = getattr(interaction, 'guild', None)
        guild_id = getattr(guild, 'id', None)
        if user.id != interaction.user.id and not _is_invoker_admin():
            await interaction.edit_original_response(content="You don't have permission to unignore other users; you may only unignore yourself.")
            return

        removed = await cache.remove_ignored_user(user.id, guild_id)
        if removed:
            await interaction.edit_original_response(content=f'User {user.name} ({user.id}) removed from ignore list for this server.')
        else:
            await interaction.edit_original_response(content=f'User {user.name} ({user.id}) was not in the ignore list for this server.')
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.command(description='List users on the ignore list for this guild (and global entries)')
async def list_ignored(interaction: discord.Interaction) -> None:
    await interaction.response.send_message('Fetching ignore list...', ephemeral=True)
    try:
        # Only server admins may view the server ignore list
        guild = getattr(interaction, 'guild', None)
        guild_id = getattr(guild, 'id', None)

        def _is_invoker_admin():
            if guild is None:
                return False
            try:
                member = guild.get_member(interaction.user.id)
            except Exception:
                member = None
            if member is None:
                return False
            try:
                if getattr(guild, 'owner_id', None) == member.id:
                    return True
                perms = getattr(member, 'guild_permissions', None)
                if perms is not None and (perms.administrator or perms.manage_guild):
                    return True
            except Exception:
                pass
            return False

        if not _is_invoker_admin():
            await interaction.edit_original_response(content="You don't have permission to view the server's ignore list.")
            return

        rows = await cache.list_ignored_users(guild_id)
        if not rows:
            await interaction.edit_original_response(content='No ignored users for this server.')
            return
        lines = []
        for r in rows:
            g = r.get('guild_id')
            scope = 'Global' if g is None else 'Server'
            name = r.get('name') or 'Unknown'
            lines.append(f"{name} ({r.get('user_id')}) — {scope}")
        await interaction.edit_original_response(content='\n'.join(lines))
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.command(description='Summarize for topic')
async def summarize_topic(interaction: discord.Interaction, topic: str, count_msgs: int | None = None,
                          channel: discord.TextChannel = None) -> None:
    # If count_msgs is omitted, summarize all cached messages for the channel by default.
    await interaction.response.send_message('Doing stuff, might take a (long) while...')

    try:
        if channel is None:
            channel = interaction.channel

        await interaction.edit_original_response(
            content='Doing stuff, might take a (long) while... (compiling messages)')

        # Get the messages. If no count was provided, default to all cached messages for this channel.
        if count_msgs is None:
            discord_messages = await cache.get_all_messages_in_channel_from_cache(channel)
        else:
            discord_messages = await cache.get_messages(channel=channel, limit=count_msgs)

        footer_text = f'Summary of the last {len(discord_messages)} messages in {channel.name} | Topic: {topic}'

        await interaction.edit_original_response(content='Doing stuff, might take a (long) while... (Firing up AI)')
        await create_topic_summary(interaction, discord_messages, topic, footer_text)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.command(description='Pre-fetch older messages and reconcile channel links in the DB')
async def reconcile_messages(interaction: discord.Interaction, count_msgs: int = 1000,
                             channel: discord.TextChannel = None) -> None:
    await interaction.response.send_message('Starting reconciliation, this may take a while...', ephemeral=True)

    try:
        if channel is None:
            channel = interaction.channel

        # We'll update the original response periodically to show progress
        last_reported = {'processed': 0}

        async def progress_cb(processed, total, type="Fetched"):
            # Only edit when notable progress is made
            if processed - last_reported['processed'] >= 25:
                try:
                    await interaction.edit_original_response(content=f'{type} {processed}/{total} messages...')
                    last_reported['processed'] = processed
                except Exception:
                    pass

        processed = await cache.reconcile_channel_links(channel, limit=count_msgs, progress_callback=progress_cb)

        await interaction.edit_original_response(content=f'Reconciled {processed} messages for channel {channel.name}')
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.command(description='Fill gaps between earliest and latest cached messages for a channel')
async def reconcile_gaps(interaction: discord.Interaction, channel: discord.TextChannel = None) -> None:
    await interaction.response.send_message('Starting gap reconciliation, this may take a while...', ephemeral=True)

    try:
        if channel is None:
            channel = interaction.channel

        last_reported = {'processed': 0}

        async def progress_cb(processed, total, type='fetched_gap'):
            # Update periodically
            if processed - last_reported['processed'] >= 50:
                try:
                    await interaction.edit_original_response(content=f'Fetched {processed} messages...')
                    last_reported['processed'] = processed
                except Exception:
                    pass

        processed = await cache.reconcile_fill_gaps_between_cached_bounds(channel, progress_callback=progress_cb)

        await interaction.edit_original_response(content=f'Filled {processed} missing messages for channel {channel.name}')
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.command(description='Show cache status for a channel')
async def cache_status(interaction: discord.Interaction, channel: discord.TextChannel = None) -> None:
    await interaction.response.send_message('Fetching cache status...', ephemeral=True)

    try:
        if channel is None:
            channel = interaction.channel

        stats = await cache.get_channel_cache_stats(channel)

        embed = discord.Embed(title=f'Cache status for #{channel.name}')
        embed.add_field(name='Total cached messages', value=str(stats['total']), inline=True)
        embed.add_field(name='Heads (no previous)', value=str(stats['heads']), inline=True)
        embed.add_field(name='Missing previous', value=str(stats['missing_previous']), inline=True)
        embed.add_field(name='Missing next', value=str(stats['missing_next']), inline=True)
        if stats['earliest_id'] is not None:
            embed.add_field(name='Earliest ID', value=str(stats['earliest_id']), inline=False)
            embed.add_field(name='Earliest preview', value=stats['earliest_preview'] or 'N/A', inline=False)
        if stats['latest_id'] is not None:
            embed.add_field(name='Latest ID', value=str(stats['latest_id']), inline=False)
            embed.add_field(name='Latest preview', value=stats['latest_preview'] or 'N/A', inline=False)

        await interaction.edit_original_response(content='', embed=embed)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise


@client.tree.command(description='Search cached messages using embeddings (does NOT hit Discord API)')
async def search_cache(interaction: discord.Interaction, query: str, channel: discord.TextChannel = None,
                       top_k: int = 5) -> None:
    await interaction.response.send_message('Searching cache...', ephemeral=True)

    try:
        if channel is None:
            channel = interaction.channel

        model = 'bge-multilingual-gemma2'

        # 1) Build grouped search embeddings (concatenate consecutive messages by same author)
        # progress callback: update the interaction at most once every 3 seconds
        last_reported = {'t': 0, 'processed': 0}

        async def search_progress_cb(stage: str, processed: int, total: int):
            now = time.time()
            # update at most once every 3 seconds, or when finished
            if now - last_reported['t'] >= 3 or processed == total:
                try:
                    await interaction.edit_original_response(content=f'{stage.replace("_", " ").capitalize()}: {processed}/{total}...')
                    last_reported['t'] = now
                    last_reported['processed'] = processed
                except Exception:
                    pass

        grouped = await create_search_embeddings(channel, model=model, limit=5000, progress_callback=search_progress_cb)
        if not grouped:
            await interaction.edit_original_response(content='No cached embeddings found for this channel.')
            return

        # 2) Ensure we have an embedding for the query
        question_embedding_resp = await ai_client.embeddings.create(input=query, model=model)
        q_emb = question_embedding_resp.data[0].embedding

        # 3) Compute cosine similarities
        message_embs = [item.embedding for item in grouped if item.embedding is not None]
        if not message_embs:
            await interaction.edit_original_response(content='No embeddings available to search.')
            return

        # Compute cosine similarities in batches so we can report progress
        q_vec = np.array(q_emb).reshape(1, -1)
        n = len(message_embs)
        batch_size = 256
        sims_parts = []
        for i in range(0, n, batch_size):
            batch = np.stack(message_embs[i:i + batch_size])
            part = cosine_similarity(batch, q_vec).reshape(-1)
            sims_parts.append(part)
            # report similarity progress
            try:
                await search_progress_cb('similarity', min(i + batch_size, n), n)
            except Exception:
                pass

        sims = np.concatenate(sims_parts)

        # 4) Pick top_k indices
        top_idx = sims.argsort()[::-1]

        # 5) For each result, fetch nearby context and build a snippet
        # Build initial result list with context message ids
        results = []
        seen_ids = set()
        for idx in top_idx:
            if len(results) >= top_k:
                break

            item = grouped[int(idx)]
            score = float(sims[int(idx)])
            # pick the most recent message id in the grouped block as representative
            rep_id = item.message_ids[-1] if item.message_ids else None
            context = []
            if rep_id is not None:
                context = await cache.get_context_for_message(channel, rep_id, before=3, after=3)

            # Collect context ids for deduplication
            context_ids = set([c.get('id') for c in context if c.get('id') is not None])

            if context_ids & seen_ids:
                # overlap with an accepted, higher-quality result -> skip
                continue

            seen_ids |= context_ids

            # Build a snippet with usernames and highlight the central message
            snippet_lines = []
            for c in context:
                try:
                    content = c.get('content') or ''
                    short = content[:200]
                    author_obj = await cache.fetch_author_from_cache(c.get('author_id'))
                    author_name = author_obj.name if author_obj is not None else 'Unknown'
                    if rep_id is not None and c.get('id') == rep_id:
                        # bold the central message to make it stand out
                        line = f"**{author_name}: {short}**"
                    else:
                        line = f"{author_name}: {short}"
                    snippet_lines.append(line)
                except Exception:
                    # best-effort: skip problematic context rows
                    continue

            snippet = '\n'.join(snippet_lines)
            results.append((score, item, snippet))

        # 6) Build embed to return (compact)
        embed = discord.Embed(title=f'Search results for "{query}" (top {top_k})')
        for score, item, snippet in results:
            top_id = item.message_ids[0] if item.message_ids else None
            # Build a Discord message jump link to the top (earliest) message in the block
            jump_link = None
            try:
                guild_id = channel.guild.id
                jump_link = f'https://discord.com/channels/{guild_id}/{channel.id}/{top_id}' if top_id is not None else None
            except Exception:
                jump_link = None

            title = f"Score: {score:.4f}"
            snippet_text = (snippet or '')
            if snippet_text and len(snippet_text) > 1000:
                snippet_text = snippet_text[:1000] + '...'

            if jump_link:
                value = f"[Jump to messages]({jump_link})\n\n{snippet_text}"
            else:
                value = snippet_text or ''

            embed.add_field(name=title, value=value, inline=False)

        await interaction.edit_original_response(content='', embed=embed)
    except Exception as e:
        await interaction.edit_original_response(content=f'Caught exception: {e}')
        raise

async def uwuify_impl(interaction: discord.Interaction, message: discord.Message, ephemeral: bool):
    await interaction.response.send_message('Doing stuff, might take a while...', ephemeral=ephemeral)

    model = 'llama-3.3-70b-instruct'
    temperature = 1.26
    top_p = 0.81
    max_tokens = 4096
    presence_penalty = 0

    messages = [{'role': 'system', 'content': 'UwU-ify the users message, and echo it, without saying anything else. '
                                              'Use emoticons too.'},
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
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def uwuify(interaction: discord.Interaction, message: discord.Message):
    await uwuify_impl(interaction=interaction, message=message, ephemeral=False)


@client.tree.context_menu(name='UwU-ify message (Keep Private)')
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def uwuify_ephemeral(interaction: discord.Interaction, message: discord.Message):
    await uwuify_impl(interaction=interaction, message=message, ephemeral=True)


async def zoomer_translator_impl(interaction: discord.Interaction, message: discord.Message, ephemeral: bool):
    await interaction.response.send_message('Doing stuff, might take a while...', ephemeral=ephemeral)

    model = 'deepseek-r1-distill-llama-70b'
    temperature = 0.70
    top_p = 0.95
    max_tokens = 4096
    presence_penalty = 0

    messages = [{'role': 'system', 'content': 'You are given messages that use many acronyms, phrases, and words '
                                              'associated with \"zoomers\" and zoomer culture. You should translate '
                                              'the messages into more conventional language and grammar, and then '
                                              'say the translated message, and nothing else.'},
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

@client.tree.context_menu(name='Zoomer Translator')
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def zoomer_translator(interaction: discord.Interaction, message: discord.Message):
    await zoomer_translator_impl(interaction=interaction, message=message, ephemeral=False)

@client.tree.context_menu(name='Zoomer Translator (Keep Private)')
@app_commands.allowed_contexts(guilds=True, dms=True, private_channels=True)
async def zoomer_translator_ephemeral(interaction: discord.Interaction, message: discord.Message):
    await zoomer_translator_impl(interaction=interaction, message=message, ephemeral=True)


def main() -> None:
    client.run(discord_token)


if __name__ == '__main__':
    main()
