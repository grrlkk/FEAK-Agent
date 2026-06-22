"""MongoDB index definitions for FEAK-TC collections."""

from typing import Dict, List

from pymongo import ASCENDING


INDEX_SPECS = {
    "essays": [
        ([("essay_id", ASCENDING)], {"unique": True}),
        ([("raw_path", ASCENDING)], {}),
        ([("topic", ASCENDING), ("grade", ASCENDING)], {}),
    ],
    "feak_observations": [
        ([("text_hash", ASCENDING)], {"unique": True}),
        ([("essay_id", ASCENDING)], {}),
    ],
    "expert_action_labels": [
        ([("essay_id", ASCENDING)], {}),
        ([("target_rubric", ASCENDING), ("action_type", ASCENDING)], {}),
    ],
    "agent_runs": [
        ([("run_id", ASCENDING)], {"unique": True}),
        ([("essay_id", ASCENDING), ("status", ASCENDING)], {}),
    ],
    "agent_steps": [
        ([("run_id", ASCENDING), ("step_index", ASCENDING)], {"unique": True}),
    ],
    "candidates": [
        ([("run_id", ASCENDING), ("step_index", ASCENDING), ("candidate_id", ASCENDING)], {"unique": True}),
    ],
    "transition_pairs": [
        ([("pair_id", ASCENDING)], {"unique": True}),
        ([("essay_id", ASCENDING)], {}),
    ],
    "tvm_training_examples": [
        ([("example_id", ASCENDING)], {"unique": True}),
        ([("split", ASCENDING)], {}),
    ],
}


def create_indexes(db) -> Dict[str, List[str]]:
    """Create indexes for all FEAK-TC collections and return index names."""

    created: Dict[str, List[str]] = {}
    for collection_name, specs in INDEX_SPECS.items():
        collection = db[collection_name]
        created[collection_name] = [
            collection.create_index(keys, **options)
            for keys, options in specs
        ]
    return created
