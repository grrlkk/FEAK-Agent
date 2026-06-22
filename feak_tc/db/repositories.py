"""Repository helpers for MongoDB persistence."""

from typing import Dict, Iterable, List

from pymongo import UpdateOne

from feak_tc.schemas.essay import EssayRecord


class EssayRepository:
    """Persistence adapter for normalized essays."""

    def __init__(self, db, collection_name: str = "essays") -> None:
        self.collection = db[collection_name]

    def upsert_one(self, record: EssayRecord):
        document = record.to_mongo_document()
        return self.collection.update_one(
            {"essay_id": document["essay_id"]},
            {"$set": document},
            upsert=True,
        )

    def upsert_many(self, records: Iterable[EssayRecord]) -> Dict[str, int]:
        operations: List[UpdateOne] = []
        for record in records:
            document = record.to_mongo_document()
            operations.append(
                UpdateOne(
                    {"essay_id": document["essay_id"]},
                    {"$set": document},
                    upsert=True,
                )
            )

        if not operations:
            return {
                "matched_count": 0,
                "modified_count": 0,
                "upserted_count": 0,
            }

        result = self.collection.bulk_write(operations, ordered=False)
        return {
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_count": len(result.upserted_ids),
        }
