#!/usr/bin/env python
"""Ingest normalized AI-Hub essay JSON files into MongoDB."""

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.data.aihub_loader import load_aihub_records
from feak_tc.db.indexes import create_indexes
from feak_tc.db.mongo import DEFAULT_MONGO_DB, get_database, get_mongo_client, ping
from feak_tc.db.repositories import EssayRepository


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/raw", help="JSON file or directory to ingest")
    parser.add_argument("--limit", type=int, default=None, help="Maximum records to ingest")
    parser.add_argument("--mongo-uri", default=None, help="MongoDB URI; defaults to MONGO_URI")
    parser.add_argument("--mongo-db", default=None, help=f"MongoDB database; defaults to MONGO_DB or {DEFAULT_MONGO_DB}")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    parser.add_argument("--skip-indexes", action="store_true", help="Do not create MongoDB indexes")
    parser.add_argument("--dry-run", action="store_true", help="Normalize and print a summary without writing to MongoDB")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    env_path = Path(args.env_file)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    input_path = Path(args.input)
    records = load_aihub_records(input_path, limit=args.limit, missing_ok=True)
    if not records:
        print(f"No JSON records found under: {input_path}")
        return 0

    print(f"Loaded {len(records)} normalized records from {input_path}")
    if args.dry_run:
        sample = records[0].model_dump(mode="json")
        print(f"Sample essay_id: {sample['essay_id']}")
        print(f"Sample text length: {len(sample['text'])}")
        return 0

    client = get_mongo_client(uri=args.mongo_uri)
    ping(client)
    db = get_database(client=client, db_name=args.mongo_db)

    if not args.skip_indexes:
        created = create_indexes(db)
        print(f"Created indexes for {len(created)} collections")

    result = EssayRepository(db).upsert_many(records)
    print(
        "Upserted essays: "
        f"matched={result['matched_count']} "
        f"modified={result['modified_count']} "
        f"upserted={result['upserted_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
