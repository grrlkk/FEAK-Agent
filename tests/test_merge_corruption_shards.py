import json
import sys

from scripts import merge_corruption_shards


def test_merge_can_exclude_previously_selected_records(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    shard = tmp_path / "shard.jsonl"
    previous = tmp_path / "previous.jsonl"
    output = tmp_path / "output.jsonl"
    _write_jsonl(source, [{"record_id": f"r{i}"} for i in range(5)])
    _write_jsonl(
        shard,
        [
            {"record_id": "r3", "status": "ok"},
            {"record_id": "r1", "status": "ok"},
            {"record_id": "r4", "status": "partial"},
            {"record_id": "r2", "status": "ok"},
        ],
    )
    _write_jsonl(previous, [{"record_id": "r1"}])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_corruption_shards.py",
            "--source",
            str(source),
            "--shards",
            str(shard),
            "--output",
            str(output),
            "--status",
            "ok",
            "--exclude-records-from",
            str(previous),
            "--limit",
            "2",
        ],
    )

    assert merge_corruption_shards.main() == 0
    assert [row["record_id"] for row in _read_jsonl(output)] == ["r2", "r3"]


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
