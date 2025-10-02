import ast
import hashlib
import sqlite3
import discord
import os
import numpy as np

from pprint import pprint

os.makedirs('data', exist_ok=True)

db_con = sqlite3.connect('data/cache.db')
# Improve concurrency and read performance
try:
    db_con.execute('PRAGMA journal_mode = WAL')
    db_con.execute('PRAGMA synchronous = NORMAL')
    db_con.execute('PRAGMA temp_store = MEMORY')
except Exception:
    pass

db_con.execute('CREATE TABLE IF NOT EXISTS authors(id INT PRIMARY KEY, name TEXT)')
db_con.execute('CREATE TABLE IF NOT EXISTS messages(id PRIMARY KEY, content TEXT, author_id INT, channel_id INT,'
               'previous_message_id INT, next_message_id INT)')
db_con.execute('CREATE TABLE IF NOT EXISTS embeddings(hash INT, embedding BLOB, model TEXT, UNIQUE(hash, model))')

# Create helpful indices for faster lookups
try:
    db_con.execute('CREATE INDEX IF NOT EXISTS idx_messages_next ON messages(next_message_id)')
    db_con.execute('CREATE INDEX IF NOT EXISTS idx_messages_prev ON messages(previous_message_id)')
    db_con.execute('CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id)')
    db_con.execute('CREATE INDEX IF NOT EXISTS idx_messages_channel_next ON messages(channel_id, next_message_id)')
    db_con.execute('CREATE INDEX IF NOT EXISTS idx_messages_channel_prev ON messages(channel_id, previous_message_id)')
except Exception:
    pass

db_con.commit()


class CachedDiscordAuthor(discord.Object):
    name: str
    id: int

    def __init__(self, name: str, id: int):
        self.name = name
        self.id = id


class CachedDiscordMessage(discord.Object):
    content: str
    id: int
    before_id: int
    after_id: int
    author: CachedDiscordAuthor
    channel_id: int

    def __init__(self, content: str = '', id: int | None = None, before: int | None = None, after: int | None = None,
                 author: CachedDiscordAuthor | None = None, channel_id: int | None = None):
        self.content = content
        self.id = id
        self.before_id = before
        self.after_id = after
        self.author = author
        self.channel_id = channel_id

    @property
    def type(self):
        return 'cached'

    def __repr__(self):
        return super().__repr__() + f' (author={self.author.name if self.author else None}, content="{self.content[:30]}...")'


# For this, we expect the messages to be in oldest -> newest order.
async def commit_messages_to_cache(messages: list[CachedDiscordMessage], channel: discord.TextChannel):
    cur = db_con.cursor()
    for idx, message in enumerate(messages):
        # Set known prev/after message IDs
        if idx != 0:
            message.before_id = messages[idx - 1].id

        if idx != len(messages) - 1:
            message.after_id = messages[idx + 1].id

        # If prev or after are None, check if the message is in the database, and check if we have that saved there
        if message.before_id is None:
            cur.execute('SELECT previous_message_id FROM messages WHERE id = ?', (message.id,))
            result = cur.fetchone()
            print(f'before: {result} in cache for message {message.id} ({message.content})')
            if result is not None and result[0] is not None:
                message.before_id = int(result[0])

        if message.after_id is None:
            cur.execute('SELECT next_message_id FROM messages WHERE id = ?', (message.id,))
            result = cur.fetchone()
            print(f'after: {result} in cache for message {message.id} ({message.content})')
            # The result may be none if none was put in
            if result is not None and result[0] is not None:
                message.after_id = int(result[0])

        # Commit message and author (TODO: optimize this) to SQL cache
        cur.execute('INSERT OR REPLACE INTO messages(id, content, author_id, channel_id, previous_message_id,'
                    'next_message_id) VALUES (?, ?, ?, ?, ?, ?)', (message.id, message.content, message.author.id,
                                                                   message.channel_id, message.before_id,
                                                                   message.after_id))

        cur.execute('INSERT OR REPLACE INTO authors(id, name) VALUES (?, ?)', (message.author.id,
                                                                               message.author.name))
    cur.close()
    db_con.commit()


async def fetch_author_from_cache(author_id: int):
    cur = db_con.cursor()
    cur.execute('SELECT name FROM authors WHERE id = ?', (author_id,))
    result = cur.fetchone()
    cur.close()
    if result is not None:
        return CachedDiscordAuthor(id=author_id, name=result[0])
    else:
        return None


async def fetch_from_cache(limit: int, before: CachedDiscordMessage | discord.Message | None = None, progress_callback=None):
    cur = db_con.cursor()
    cached_messages = []

    if before is None:
        return cached_messages

    # Use a recursive CTE to fetch up to `limit` messages that precede `before.id` by following next_message_id links.
    cte_sql = '''
    WITH RECURSIVE chain(depth, id, content, author_id, channel_id, previous_message_id, next_message_id) AS (
      SELECT 0, id, content, author_id, channel_id, previous_message_id, next_message_id
      FROM messages WHERE next_message_id = ?
      UNION ALL
      SELECT chain.depth+1, m.id, m.content, m.author_id, m.channel_id, m.previous_message_id, m.next_message_id
      FROM messages m JOIN chain ON m.next_message_id = chain.id
    )
    SELECT id, content, author_id, channel_id, previous_message_id, next_message_id
    FROM chain
    ORDER BY depth ASC
    LIMIT ?;
    '''

    try:
        cur.execute(cte_sql, (before.id, limit))
        rows = cur.fetchall()
    except Exception:
        cur.close()
        return []

    for idx, row in enumerate(rows):
        message_id, content, author_id, channel_id, previous_message_id, next_message_id = row
        author = await fetch_author_from_cache(author_id)
        if author is None:
            # stop if author missing
            break

        cached_message = CachedDiscordMessage(id=message_id, content=content, author=author,
                                              channel_id=channel_id, before=previous_message_id,
                                              after=next_message_id)
        cached_messages.append(cached_message)

        # Periodic progress callback
        if progress_callback is not None and (idx + 1) % 50 == 0:
            try:
                await progress_callback(len(cached_messages), len(rows), "Fetched from cache")
            except Exception:
                pass

    if len(cached_messages):
        print(f'Fetched {len(cached_messages)} messages from cache')

    cur.close()
    return cached_messages


# Get all messages in a channel from cache. May or may not be ordered in any way!
async def get_all_messages_in_channel_from_cache(channel: discord.TextChannel):
    cur = db_con.cursor()
    cur.execute('SELECT id, content, author_id, channel_id, previous_message_id, next_message_id'
                ' FROM messages WHERE channel_id = ?', (channel.id,))
    rows = cur.fetchall()
    cur.close()

    cached_messages = []
    for row in rows:
        message_id, content, author_id, channel_id, previous_message_id, next_message_id = row
        author = await fetch_author_from_cache(author_id)
        if author is None:
            continue

        cached_message = CachedDiscordMessage(id=message_id, content=content, author=author,
                                              channel_id=channel_id, before=previous_message_id,
                                              after=next_message_id)
        cached_messages.append(cached_message)

    print(f'Fetched {len(cached_messages)} messages from cache for channel {channel.id}')
    return cached_messages


async def fetch_embedding_from_cache(message: CachedDiscordMessage | discord.Message | str, used_model: str):
    if type(message) is str:
        hash = hashlib.sha1(message.encode('utf-8')).hexdigest()
    else:
        hash = hashlib.sha1(message.content.encode('utf-8')).hexdigest()

    cur = db_con.cursor()
    cur.execute('SELECT embedding FROM embeddings WHERE hash = ? AND model = ?', (hash, used_model))
    row = cur.fetchone()
    cur.close()

    if row is None:
        return None

    embedding_data = row[0]
    # Support both legacy TEXT storage (Python literal) and new BLOB (float32 bytes)
    try:
        if isinstance(embedding_data, (bytes, bytearray)):
            emb = np.frombuffer(embedding_data, dtype=np.float32)
            return emb
        else:
            # legacy serialized Python literal
            embedding = ast.literal_eval(embedding_data)
            return np.asarray(embedding, dtype=np.float32)
    except Exception:
        return None


async def fetch_embeddings_for_hashes(hashes: list[str], used_model: str) -> dict:
    """Fetch embeddings for multiple hashes in a single query.

    Returns a mapping hash -> embedding (list) for found embeddings.
    """
    if not hashes:
        return {}

    cur = db_con.cursor()
    found = {}
    # Query in chunks to avoid too large IN lists
    chunk_size = 500
    for i in range(0, len(hashes), chunk_size):
        chunk = hashes[i:i + chunk_size]
        placeholders = ','.join('?' for _ in chunk)
        sql = f'SELECT hash, embedding FROM embeddings WHERE hash IN ({placeholders}) AND model = ?'
        params = chunk + [used_model]
        try:
            cur.execute(sql, params)
            for row in cur.fetchall():
                h_val, embedding_data = row
                try:
                    if isinstance(embedding_data, (bytes, bytearray)):
                        emb = np.frombuffer(embedding_data, dtype=np.float32)
                    else:
                        emb_list = ast.literal_eval(embedding_data)
                        emb = np.asarray(emb_list, dtype=np.float32)
                except Exception:
                    emb = None
                if emb is not None:
                    found[h_val] = emb
        except Exception:
            # ignore DB errors and continue
            continue

    cur.close()
    return found

async def commit_embedding_to_cache(message: CachedDiscordMessage | discord.Message | str, embedding: list[float], used_model: str):
    if type(message) is str:
        hash = hashlib.sha1(message.encode('utf-8')).hexdigest()
    else:
        hash = hashlib.sha1(message.content.encode('utf-8')).hexdigest()
    # Ensure numpy float32 representation for compact storage
    try:
        arr = np.asarray(embedding, dtype=np.float32)
        blob = arr.tobytes()
    except Exception:
        # Fallback: store as text representation if numpy fails for some reason
        blob = str(embedding).encode('utf-8')

    cur = db_con.cursor()
    cur.execute('INSERT OR REPLACE INTO embeddings(hash, embedding, model) VALUES (?, ?, ?)', (hash, blob, used_model))
    cur.close()
    db_con.commit()


def discord_author_to_cached_author(author: discord.User):
    return CachedDiscordAuthor(name=author.name, id=author.id)


def discord_messages_to_cached_messages(discord_messages: list[discord.Message]):
    messages = []
    for discord_message in discord_messages:
        messages.append(CachedDiscordMessage(content=discord_message.content, id=discord_message.id,
                                                   author=discord_author_to_cached_author(discord_message.author),
                                                   channel_id=discord_message.channel.id))

    return messages

async def get_messages_from_discord(channel: discord.TextChannel, limit=50, before=None, after=None):
    print(f'Fetching {limit} messages from discord directly (limit={limit}, before={before}, after={after})')
    return [message async for message in channel.history(limit=limit, before=before, after=after)]

async def get_messages(channel: discord.TextChannel, limit=50, before=None, after=None, progress_callback=None):
    messages = []

    while limit > 0:
        # TODO: handle after being true
        assert after is None

        # If before or after are both none, no requests have been made
        # Therefore, request the first batch of messages
        # Whenever calling to discord API, we max the limit to 50, as that is the maximum the discord API itself supports
        if before is None and after is None:
            discord_messages = await get_messages_from_discord(channel, limit=min(limit, 50))
            before = discord_messages[-1]
        elif before is not None:
            async def fetch_from_cache_progress_callback(fetched, total, phase):
                if progress_callback is not None:
                    try:
                        await progress_callback(len(messages) + fetched, len(messages) + limit, "Fetched")
                    except Exception:
                        # don't let progress failures stop fetching
                        pass

            # Check to see if we have a cache hit, and if we do, pull as many messages from there as we can (up to limit)
            cached_messages = await fetch_from_cache(limit, before, fetch_from_cache_progress_callback)
            if len(cached_messages):
                messages.extend(cached_messages)
                limit -= len(cached_messages)
                before = cached_messages[-1]
                continue
            else:
                # Messages not in cache, fetch them from discord
                discord_messages = await get_messages_from_discord(channel, min(limit, 50), before=before)
                if len(discord_messages) == 0:
                    break
                before = discord_messages[-1]

        # If we got here, we need to convert the discord messages to cached messages
        messages.extend(discord_messages_to_cached_messages(discord_messages))
        limit -= min(limit, 50)

        # Call progress callback if provided
        if progress_callback is not None:
            try:
                await progress_callback(len(messages), len(messages) + limit, "Fetched")
            except Exception:
                # don't let progress failures stop fetching
                pass

    if before is not None:
        # We fetched from newest, reverse array
        messages = messages[::-1]

    await commit_messages_to_cache(messages, channel)
    return messages


async def commit_single_message_to_cache(discord_message: discord.Message):
    """Commit a single incoming discord.Message to the cache while maintaining
    previous_message_id/next_message_id links.

    Behavior:
    - Determine the immediate previous message in the channel (via history).
    - Store the new message and set previous_message_id to the previous message's id (if any).
    - If the previous message is already in the DB, set its next_message_id to this message's id.
    - Always ensure the message's author is present in the authors table.
    """
    cur = db_con.cursor()

    # Ensure author is present
    author = discord_message.author
    try:
        cur.execute('INSERT OR REPLACE INTO authors(id, name) VALUES (?, ?)', (author.id, author.name))
    except Exception:
        # best-effort: ignore author insert failures
        pass

    # Try to find the immediate previous message in the channel
    previous_id = None
    try:
        # Use channel.history to get the last message before this one
        async for prev in discord_message.channel.history(limit=1, before=discord_message):
            # Only accept the previous id if that message already exists in our DB.
            prev_id_candidate = prev.id
            cur.execute('SELECT id FROM messages WHERE id = ?', (prev_id_candidate,))
            if cur.fetchone() is not None:
                previous_id = prev_id_candidate
            break
    except Exception:
        # If history access fails (permissions, etc.), leave previous_id as None
        previous_id = None

    # Insert or replace this message record. next_message_id is unknown at insert time.
    try:
        cur.execute(
            'INSERT OR REPLACE INTO messages(id, content, author_id, channel_id, previous_message_id, next_message_id) VALUES (?, ?, ?, ?, ?, ?)',
            (discord_message.id, discord_message.content, author.id, discord_message.channel.id, previous_id, None)
        )
        print(f'Committed message {discord_message.id} to cache with previous_id={previous_id} (content: {discord_message.content})')
    except Exception:
        # If insert fails for some reason, close and return
        cur.close()
        db_con.commit()
        return

    # If we know the previous message id, and it exists in DB, update its next_message_id
    if previous_id is not None:
        cur.execute('SELECT id FROM messages WHERE id = ?', (previous_id,))
        if cur.fetchone() is not None:
            try:
                cur.execute('UPDATE messages SET next_message_id = ? WHERE id = ?', (discord_message.id, previous_id))
            except Exception:
                # ignore update errors
                pass

    cur.close()
    db_con.commit()


async def update_message_in_cache(discord_message: discord.Message) -> bool:
    """If the message exists in the DB, update its content (and author name) and return True.
    If the message is not present, return False.
    """
    cur = db_con.cursor()
    try:
        cur.execute('SELECT id FROM messages WHERE id = ?', (discord_message.id,))
        if cur.fetchone() is None:
            cur.close()
            return False

        # Update author name in authors table as well
        try:
            cur.execute('INSERT OR REPLACE INTO authors(id, name) VALUES (?, ?)', (discord_message.author.id, discord_message.author.name))
        except Exception:
            pass

        # Update the message content
        cur.execute('UPDATE messages SET content = ? WHERE id = ?', (discord_message.content, discord_message.id))
        db_con.commit()
        cur.close()
        return True
    except Exception:
        try:
            cur.close()
        except Exception:
            pass
        return False


async def delete_message_from_cache(message: discord.Message | int) -> bool:
    """Remove a message from the cache and fix linked previous/next pointers.

    Accepts either a discord.Message or an integer message id. Returns True if a row was
    deleted, False otherwise.
    """
    # Normalize to id
    if isinstance(message, int):
        message_id = message
    else:
        try:
            message_id = int(message.id)
        except Exception:
            return False

    cur = db_con.cursor()
    try:
        print(f'[cache] delete_message_from_cache: message_id={message_id}')
        # Find the neighbouring pointers for this message
        cur.execute('SELECT previous_message_id, next_message_id FROM messages WHERE id = ?', (message_id,))
        row = cur.fetchone()
        if row is None:
            print(f'[cache] delete_message_from_cache: message_id={message_id} not found in DB')
            cur.close()
            return False

        prev_id, next_id = row
        print(f'[cache] delete_message_from_cache: prev_id={prev_id}, next_id={next_id}')

        # Update previous' next_message_id to skip this message
        if prev_id is not None:
            try:
                cur.execute('UPDATE messages SET next_message_id = ? WHERE id = ?', (next_id, prev_id))
                print(f'[cache] Updated prev ({prev_id}). next_message_id -> {next_id} (rows={cur.rowcount})')
            except Exception as e:
                print(f'[cache] Failed to update prev ({prev_id}): {e}')

        # Update next's previous_message_id to skip this message
        if next_id is not None:
            try:
                cur.execute('UPDATE messages SET previous_message_id = ? WHERE id = ?', (prev_id, next_id))
                print(f'[cache] Updated next ({next_id}). previous_message_id -> {prev_id} (rows={cur.rowcount})')
            except Exception as e:
                print(f'[cache] Failed to update next ({next_id}): {e}')

        # Delete the message row itself
        cur.execute('DELETE FROM messages WHERE id = ?', (message_id,))
        deleted = cur.rowcount
        db_con.commit()
        cur.close()
        print(f'[cache] delete_message_from_cache: deleted_rows={deleted}')
        return deleted > 0
    except Exception as e:
        try:
            cur.close()
        except Exception:
            pass
        print(f'[cache] delete_message_from_cache: exception: {e}')
        return False


async def reconcile_channel_links(channel: discord.TextChannel, limit: int | None = None,
                                   progress_callback=None) -> int:
    # Call get_messages to re-fetch messages and re-commit them to the cache, which will fix up links    
    messages = await get_messages(channel, limit=limit, before=None, after=None, progress_callback=progress_callback)
    return len(messages)

async def get_channel_cache_stats(channel: discord.TextChannel) -> dict:
    """Return basic statistics about the cached messages for a channel.

    Returns a dict with keys: total, missing_previous, missing_next, earliest_id,
    latest_id, earliest_preview, latest_preview, heads (number of messages with no previous).
    """
    cur = db_con.cursor()
    try:
        cur.execute('SELECT COUNT(*) FROM messages WHERE channel_id = ?', (channel.id,))
        total = cur.fetchone()[0]

        cur.execute('SELECT COUNT(*) FROM messages WHERE channel_id = ? AND previous_message_id IS NULL', (channel.id,))
        missing_previous = cur.fetchone()[0]

        cur.execute('SELECT COUNT(*) FROM messages WHERE channel_id = ? AND next_message_id IS NULL', (channel.id,))
        missing_next = cur.fetchone()[0]

        cur.execute('SELECT id, content FROM messages WHERE channel_id = ? ORDER BY id ASC LIMIT 1', (channel.id,))
        row = cur.fetchone()
        if row is None:
            earliest_id = None
            earliest_preview = None
        else:
            earliest_id = row[0]
            earliest_preview = (row[1][:200] + '...') if row[1] and len(row[1]) > 200 else row[1]

        cur.execute('SELECT id, content FROM messages WHERE channel_id = ? ORDER BY id DESC LIMIT 1', (channel.id,))
        row = cur.fetchone()
        if row is None:
            latest_id = None
            latest_preview = None
        else:
            latest_id = row[0]
            latest_preview = (row[1][:200] + '...') if row[1] and len(row[1]) > 200 else row[1]

        cur.execute('SELECT COUNT(*) FROM messages WHERE channel_id = ? AND previous_message_id IS NULL', (channel.id,))
        heads = cur.fetchone()[0]

        return {
            'total': total,
            'missing_previous': missing_previous,
            'missing_next': missing_next,
            'earliest_id': earliest_id,
            'latest_id': latest_id,
            'earliest_preview': earliest_preview,
            'latest_preview': latest_preview,
            'heads': heads,
        }
    finally:
        cur.close()


async def get_context_for_message(channel: discord.TextChannel, message_id: int, before: int = 3, after: int = 3):
    """Return a list of messages (dicts) around `message_id` from the cache, ordered oldest->newest.

    Each dict: id, content, author_id
    """
    cur = db_con.cursor()

    # Walk backwards via previous_message_id
    prev_ids = []
    cur_id = message_id
    for _ in range(before):
        cur.execute('SELECT previous_message_id FROM messages WHERE id = ?', (cur_id,))
        r = cur.fetchone()
        if r is None or r[0] is None:
            break
        cur_id = r[0]
        prev_ids.append(cur_id)

    prev_ids = prev_ids[::-1]  # oldest first

    # Walk forwards via next_message_id
    next_ids = []
    cur_id = message_id
    for _ in range(after):
        cur.execute('SELECT next_message_id FROM messages WHERE id = ?', (cur_id,))
        r = cur.fetchone()
        if r is None or r[0] is None:
            break
        cur_id = r[0]
        next_ids.append(cur_id)

    ids = prev_ids + [message_id] + next_ids
    if not ids:
        cur.close()
        return []

    placeholders = ','.join('?' for _ in ids)
    cur.execute(f'SELECT id, content, author_id FROM messages WHERE id IN ({placeholders})', ids)
    rows = cur.fetchall()
    # Order rows according to ids
    row_map = {r[0]: r for r in rows}
    ordered = []
    for id_ in ids:
        r = row_map.get(id_)
        if r is not None:
            ordered.append({'id': r[0], 'content': r[1], 'author_id': r[2]})

    cur.close()
    return ordered

