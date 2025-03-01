import ast
import hashlib
import sqlite3
import discord
import os

os.makedirs('data', exist_ok=True)

db_con = sqlite3.connect('data/cache.db')
db_con.execute('CREATE TABLE IF NOT EXISTS authors(id INT PRIMARY KEY, name TEXT)')
db_con.execute('CREATE TABLE IF NOT EXISTS messages(id PRIMARY KEY, content TEXT, author_id INT, channel_id INT,'
               'previous_message_id INT, next_message_id INT)')
db_con.execute('CREATE TABLE IF NOT EXISTS embeddings(hash INT PRIMARY KEY, embedding TEXT)')
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


# For this, we expect the messages to be in oldest -> newest order.
async def commit_messages_to_cache(messages: [CachedDiscordMessage], channel: discord.TextChannel):
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


async def fetch_from_cache(limit: int, before: CachedDiscordMessage | discord.Message | None = None):
    cur = db_con.cursor()
    cached_messages = []

    assert before is not None
    while limit > 0:
        # Check if we can get the message
        cur.execute('SELECT id, content, author_id, channel_id, previous_message_id, next_message_id'
                    ' FROM messages WHERE next_message_id = ?', (before.id,))
        result = cur.fetchone()

        if result is not None:
            # We have a cached result, add it to the cached messages list
            # Try to get the author
            message_id, content, author_id, channel_id, previous_message_id, next_message_id = result
            author = await fetch_author_from_cache(author_id)
            if author is None:
                # We can't fetch the author, break
                break

            cached_message = CachedDiscordMessage(id=message_id, content=content, author=author,
                                                  channel_id=channel_id, before=previous_message_id,
                                                  after=next_message_id)
            cached_messages.append(cached_message)
            before = cached_message
            limit -= 1
        else:
            # We have reached our limit
            break

    if len(cached_messages):
        print(f'Fetched {len(cached_messages)} messages from cache')

    return cached_messages


async def fetch_embedding_from_cache(message: CachedDiscordMessage | discord.Message):
    hash = hashlib.sha1(message.content.encode('utf-8')).hexdigest()

    cur = db_con.cursor()
    cur.execute('SELECT embedding FROM embeddings WHERE hash = ?', (hash,))
    embedding_str = cur.fetchone()
    cur.close()

    if embedding_str is None:
        return None

    embedding = ast.literal_eval(embedding_str)
    assert embedding is list[float]

    return embedding
