"""MongoDB connection helpers."""

import os
from typing import Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database


DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_MONGO_DB = "feak_tc"
DEFAULT_SERVER_SELECTION_TIMEOUT_MS = 5000


def get_mongo_client(
    uri: Optional[str] = None,
    server_selection_timeout_ms: int = DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
) -> MongoClient:
    """Create a MongoClient from arguments or environment variables."""

    load_dotenv()
    mongo_uri = uri or os.getenv("MONGO_URI") or DEFAULT_MONGO_URI
    return MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=server_selection_timeout_ms,
    )


def get_database(
    client: Optional[MongoClient] = None,
    db_name: Optional[str] = None,
    uri: Optional[str] = None,
) -> Database:
    """Return the configured MongoDB database."""

    load_dotenv()
    mongo_client = client or get_mongo_client(uri=uri)
    database_name = db_name or os.getenv("MONGO_DB") or DEFAULT_MONGO_DB
    return mongo_client[database_name]


def ping(client: MongoClient) -> None:
    """Raise if MongoDB is not reachable."""

    client.admin.command("ping")
