import json

import pytest

from scripts.run_corruption_chains import _read_record_ids, _select_records


def test_select_records_applies_offset_and_limit_before_sharding():
    rows = [{"record_id": f"r{i}"} for i in range(10)]

    first = _select_records(rows, offset=3, limit=5, shard=0, num_shards=2)
    second = _select_records(rows, offset=3, limit=5, shard=1, num_shards=2)

    assert [row["record_id"] for row in first] == ["r3", "r5", "r7"]
    assert [row["record_id"] for row in second] == ["r4", "r6"]
    assert {row["record_id"] for row in first + second} == {
        "r3",
        "r4",
        "r5",
        "r6",
        "r7",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"offset": -1, "limit": 1, "shard": 0, "num_shards": 1}, "offset"),
        ({"offset": 0, "limit": 0, "shard": 0, "num_shards": 1}, "limit"),
        ({"offset": 0, "limit": 1, "shard": 1, "num_shards": 1}, "shard"),
    ],
)
def test_select_records_rejects_invalid_windows(kwargs, message):
    with pytest.raises(SystemExit, match=message):
        _select_records([], **kwargs)


def test_read_record_ids_accepts_source_or_chain_jsonl(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(
            json.dumps({"record_id": record_id}) + "\n"
            for record_id in ("r2", "r1", "r2")
        ),
        encoding="utf-8",
    )

    assert _read_record_ids([str(source)]) == {"r1", "r2"}
