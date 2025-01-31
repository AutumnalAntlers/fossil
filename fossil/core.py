import datetime
import sqlite3
import html2text
import numpy as np

import requests

from fossil import config
import os
from pydantic import BaseModel
import openai



def create_database():
    if os.path.exists("fossil.db"):
        return

    with sqlite3.connect("fossil.db") as conn:
        c = conn.cursor()

        # Create the toots table if it doesn't exist
        c.execute('''
            CREATE TABLE IF NOT EXISTS toots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                author TEXT,
                url TEXT,
                created_at DATETIME,
                embedding BLOB,
                cluster TEXT  -- Added cluster column
            )
        ''')

        conn.commit()


class Toot(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    id: int | None = None
    content: str | None
    author: str | None
    url: str | None
    created_at: datetime.datetime
    embedding: np.ndarray | None = None
    cluster: str | None = None  # Added cluster property

    def save(self) -> bool:
        with sqlite3.connect("fossil.db") as conn:
            c = conn.cursor()

            # Check if the URL already exists
            c.execute('''
                SELECT COUNT(*) FROM toots WHERE url = ? and embedding is not null
            ''', (self.url,))

            result = c.fetchone()
            url_exists = result[0] > 0

            if url_exists:
                # URL already exists, handle accordingly
                return False

            c.execute('''
                DELETE FROM toots WHERE url = ?
            ''', (self.url,))

            c.execute('''
                INSERT INTO toots (content, author, url, created_at, embedding, cluster)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (self.content, self.author, self.url, self.created_at, self.embedding.tobytes() if self.embedding is not None else bytes(), self.cluster))

            conn.commit()
            return True

    @classmethod
    def get_toots_since(cls, since: datetime.datetime) -> list["Toot"]:
        with sqlite3.connect("fossil.db") as conn:
            c = conn.cursor()

            c.execute('''
                SELECT * FROM toots WHERE created_at >= ?
            ''', (since,))

            rows = c.fetchall()
            toots = []
            for row in rows:
                toot = cls(
                    id=row[0],
                    content=row[1],
                    author=row[2],
                    url=row[3],
                    created_at=row[4],
                    embedding=np.frombuffer(row[5]) if row[5] else None,
                    cluster=row[6]  # Added cluster property
                )
                toots.append(toot)

            return toots

    @staticmethod
    def get_latest_date() -> datetime.datetime:
        with sqlite3.connect("fossil.db") as conn:
            c = conn.cursor()

            c.execute('''
                SELECT MAX(created_at) FROM toots
            ''')

            result = c.fetchone()
            latest_date = result[0] if result[0] else None

            if isinstance(latest_date, str):
                latest_date = datetime.datetime.strptime(latest_date, "%Y-%m-%d %H:%M:%S")
            return latest_date

    @classmethod
    def from_json(cls, json_data):
        if json_data.get("reblog"):
            return cls.from_json(json_data["reblog"])
        return cls(
            content=json_data.get("content"),
            author=json_data.get("account", {}).get("acct"),
            url=json_data.get("url"),
            created_at=datetime.datetime.strptime(json_data.get("created_at"), "%Y-%m-%dT%H:%M:%S.%fZ")
        )


def get_toots_since(since: datetime.datetime):
    assert isinstance(since, datetime.datetime), type(since)
    create_database()
    download_timeline(since)
    return Toot.get_toots_since(since)


def download_timeline(since: datetime.datetime):
    create_database()

    last_date = Toot.get_latest_date() or since
    earliest_date = None
    buffer: list[Toot] = []
    last_id = ""
    while not earliest_date or earliest_date > last_date:
        response = requests.get(f"https://hachyderm.io/api/v1/timelines/home?limit=40&max_id={last_id}", headers=config.headers())
        if not response.ok:
            break
        json = response.json()
        if not json:
            break
        if len(json) > 1:
            last_id = json[-1]["id"]
        print(f"Got {len(json)} toots; earliest={earliest_date.isoformat() if earliest_date else None}, last_id={last_id}")
        for toot_dict in json:
            toot = Toot.from_json(toot_dict)
            earliest_date = toot.created_at if not earliest_date else min(earliest_date, toot.created_at)
            buffer.append(toot)

    # Set embeddings in batches of 100
    page_size = 50
    num_pages = len(buffer) // page_size
    for page in range(num_pages):
        start_index = page * page_size
        end_index = start_index + page_size
        page_toots = buffer[start_index:end_index]

        # Example: Call the _create_embeddings function
        _create_embeddings(page_toots)
        for toot in page_toots:
            toot.save()


def _create_embeddings(toots: list[Toot]):
    # Convert the list of toots to a single string
    client = openai.OpenAI(api_key=config.get_config()["OPENAI_KEY"])
    toots = [t for t in toots if t.content]

    # Call the OpenAI Text Embedding API to create embeddings
    response = client.embeddings.create(input=[html2text.html2text(t.content) for t in toots], model="text-embedding-ada-002")


    # Extract the embeddings from the API response
    embeddings = [np.array(embedding.embedding) for embedding in response.data]
    for i, toot in enumerate(toots):
        toot.embedding = embeddings[i]

    # Return the embeddings
    return toots


def time_ago(dt: datetime.datetime) -> str:
    current_time = datetime.datetime.utcnow()
    time_ago = current_time - dt

    # Convert the time difference to a readable string
    if time_ago < datetime.timedelta(minutes=1):
        time_ago_str = "just now"
    elif time_ago < datetime.timedelta(hours=1):
        minutes = int(time_ago.total_seconds() / 60)
        time_ago_str = f"{minutes} minutes ago"
    elif time_ago < datetime.timedelta(days=1):
        hours = int(time_ago.total_seconds() / 3600)
        time_ago_str = f"{hours} hours ago"
    else:
        days = time_ago.days
        time_ago_str = f"{days} days ago"

    return time_ago_str
