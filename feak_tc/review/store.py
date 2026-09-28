"""Frozen public packets and isolated, versioned human ratings. No model calls."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import secrets
import sqlite3
from typing import Literal, Optional
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

CRITERIA = ("goal_achievement", "necessity", "preservation", "global_benefit")
SCHEMA_VERSION = "pilot-human-review-v1"


class ConflictError(ValueError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Rating(StrictModel):
    label: Optional[Literal["PASS", "FAIL", "UNCERTAIN"]]
    reason: str = Field(max_length=3000)


class Ratings(StrictModel):
    goal_achievement: Rating
    necessity: Rating
    preservation: Rating
    global_benefit: Rating


class ReviewUpdate(StrictModel):
    expected_version: int = Field(ge=0)
    status: Literal["draft", "submitted"]
    ratings: Ratings
    notes: str = Field(max_length=5000)

    @model_validator(mode="after")
    def complete_submission(self):
        if self.status == "submitted":
            if any(getattr(self.ratings, key).label is None or
                   not getattr(self.ratings, key).reason.strip() for key in CRITERIA):
                raise ValueError("네 기준의 판정과 근거를 모두 작성해 주세요.")
        return self


def now():
    return datetime.now(timezone.utc).isoformat()


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_private(path, value):
    with path.open("x", encoding="utf-8") as file:
        path.chmod(0o600)
        file.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: 비어 있지 않은 문자열이 필요합니다.")
    return value


def prepare_study(run_dir, study_dir, *, raters=2, seed=20260929, title="글 수정 평가"):
    """Include every produced candidate, regardless of the original RV decision."""
    if not 1 <= raters <= 50:
        raise ValueError("평가자 수는 1–50이어야 합니다.")
    title = text(title, "title")
    run_dir, study_dir = Path(run_dir).resolve(), Path(study_dir).resolve()
    sources = {name: (run_dir / name).read_bytes() for name in ("inputs.jsonl", "trajectories.jsonl")}
    inputs = {}
    for line in sources["inputs.jsonl"].decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = text(row.get("sample_id"), "sample_id")
        if sid in inputs:
            raise ValueError("중복 sample_id가 있습니다.")
        inputs[sid] = text(row.get("writing_prompt"), "writing_prompt")
    packets, mapping, excluded = [], [], []
    for number, line in enumerate(sources["trajectories.jsonl"].decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("candidate") is None:
            excluded.append({"source_line": number, "reason": "candidate_not_produced"})
            continue
        sid = row.get("sample_id")
        if sid not in inputs:
            raise ValueError(f"후보 {number}: 문항에 대응하는 sample_id가 없습니다.")
        plan = row.get("plan")
        if not isinstance(plan, dict):
            raise ValueError(f"후보 {number}: 수정 계획이 없습니다.")
        preserve = plan.get("must_preserve")
        if not isinstance(preserve, list) or not preserve:
            raise ValueError(f"후보 {number}: 보존 조건이 없습니다.")
        # Allowlist construction: never copy a trajectory or plan dictionary into the browser payload.
        packet = {"case_id": uuid.uuid4().hex, "writing_prompt": inputs[sid],
                  "before": text(row.get("text_before"), "text_before"),
                  "after": text(row.get("candidate"), "candidate"),
                  "goal": text(plan.get("goal"), "goal"),
                  "must_preserve": [text(v, "must_preserve") for v in preserve]}
        packets.append(packet)
        mapping.append({"case_id": packet["case_id"], "source_line": number,
                        "sample_id": sid, "iteration": row.get("iteration"), "attempt": row.get("attempt")})
    if not packets:
        raise ValueError("평가 가능한 수정 후보가 없습니다.")
    study_dir.mkdir(parents=True, exist_ok=False)
    study_dir.chmod(0o700)
    database = study_dir / "review.sqlite3"
    con = sqlite3.connect(database)
    database.chmod(0o600)
    credentials = []
    study_id = uuid.uuid4().hex
    try:
        con.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE cases (case_id TEXT PRIMARY KEY, packet TEXT NOT NULL);
            CREATE TABLE raters (rater_id TEXT PRIMARY KEY, label TEXT NOT NULL, token_hash TEXT NOT NULL UNIQUE);
            CREATE TABLE assignments (
                rater_id TEXT REFERENCES raters, case_id TEXT REFERENCES cases, position INTEGER NOT NULL,
                PRIMARY KEY (rater_id, case_id), UNIQUE (rater_id, position));
            CREATE TABLE reviews (
                rater_id TEXT, case_id TEXT, version INTEGER NOT NULL, status TEXT NOT NULL,
                ratings TEXT NOT NULL, notes TEXT NOT NULL, updated_at TEXT NOT NULL,
                PRIMARY KEY (rater_id, case_id), FOREIGN KEY (rater_id, case_id) REFERENCES assignments);
            CREATE TABLE history (
                id INTEGER PRIMARY KEY, rater_id TEXT NOT NULL, case_id TEXT NOT NULL,
                version INTEGER NOT NULL, status TEXT NOT NULL, ratings TEXT NOT NULL,
                notes TEXT NOT NULL, updated_at TEXT NOT NULL,
                UNIQUE (rater_id, case_id, version),
                FOREIGN KEY (rater_id, case_id) REFERENCES assignments);
        """)
        with con:
            for key, value in {"schema_version": SCHEMA_VERSION, "study_id": study_id,
                               "title": title, "created_at": now()}.items():
                con.execute("INSERT INTO meta VALUES (?,?)", (key, value))
            con.executemany("INSERT INTO cases VALUES (?,?)", [(p["case_id"], encode(p)) for p in packets])
            for index in range(1, raters + 1):
                rid, token = f"R{index}", secrets.token_urlsafe(32)
                label = f"평가자 {index:02d}"
                con.execute("INSERT INTO raters VALUES (?,?,?)", (rid, label, digest_bytes(token.encode())))
                order = [p["case_id"] for p in packets]
                random.Random(seed + index).shuffle(order)
                con.executemany("INSERT INTO assignments VALUES (?,?,?)",
                                [(rid, case_id, pos) for pos, case_id in enumerate(order)])
                credentials.append({"rater_id": rid, "label": label, "token": token})
    finally:
        con.close()
    manifest = {"schema_version": SCHEMA_VERSION, "study_id": study_id, "title": title,
                "created_at": now(), "seed": seed, "raters": raters, "cases": len(packets),
                "source_dir": str(run_dir), "source_sha256": {k: digest_bytes(v) for k, v in sources.items()},
                "packet_sha256": digest_bytes(encode(packets).encode()), "excluded": excluded,
                "selection": "all_produced_candidates_without_decision_filter", "mapping": mapping}
    write_private(study_dir / "manifest.private.json", manifest)
    write_private(study_dir / "reviewers.private.json", {"study_id": study_id, "reviewers": credentials})
    return {"study_id": study_id, "cases": len(packets), "raters": raters, "excluded": len(excluded)}


def empty_review():
    return {"version": 0, "status": "empty", "ratings": {k: {"label": None, "reason": ""} for k in CRITERIA},
            "notes": "", "updated_at": None}


def public_review(row):
    if row is None:
        return empty_review()
    return {"version": row["version"], "status": row["status"], "ratings": json.loads(row["ratings"]),
            "notes": row["notes"], "updated_at": row["updated_at"]}


class ReviewStore:
    def __init__(self, study_dir):
        self.directory = Path(study_dir).resolve()
        self.database = self.directory / "review.sqlite3"
        if not self.database.is_file() or not (self.directory / "manifest.private.json").is_file():
            raise ValueError("먼저 prepare 명령으로 평가 자료를 만들어 주세요.")
        with self.connect() as con:
            self.meta = dict(con.execute("SELECT key,value FROM meta"))
            if self.meta.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("지원하지 않는 평가 자료 버전입니다.")

    @contextmanager
    def connect(self):
        con = sqlite3.connect(str(self.database), timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        try:
            yield con
        finally:
            con.close()

    def authenticate(self, token):
        if not isinstance(token, str) or not 20 <= len(token) <= 200:
            return None
        with self.connect() as con:
            row = con.execute("SELECT rater_id,label FROM raters WHERE token_hash=?",
                              (digest_bytes(token.encode()),)).fetchone()
            return dict(row) if row else None

    def session(self, rater):
        with self.connect() as con:
            rows = con.execute("""SELECT a.case_id, a.position, COALESCE(r.status,'empty') status
                FROM assignments a LEFT JOIN reviews r USING (rater_id,case_id)
                WHERE a.rater_id=? ORDER BY a.position""", (rater,)).fetchall()
        return {"study_id": self.meta["study_id"], "title": self.meta["title"],
                "cases": [dict(r) for r in rows], "total": len(rows),
                "completed": sum(r["status"] == "submitted" for r in rows)}

    def case(self, rater, case_id):
        with self.connect() as con:
            row = con.execute("""SELECT c.packet FROM cases c JOIN assignments a USING (case_id)
                                 WHERE a.rater_id=? AND c.case_id=?""", (rater, case_id)).fetchone()
            if row is None:
                raise KeyError("평가 항목을 찾을 수 없습니다.")
            review = con.execute("SELECT * FROM reviews WHERE rater_id=? AND case_id=?", (rater, case_id)).fetchone()
        return {"case": json.loads(row["packet"]), "review": public_review(review)}

    def save(self, rater, case_id, update):
        if not isinstance(update, ReviewUpdate):
            update = ReviewUpdate.model_validate(update)
        with self.connect() as con:
            with con:
                con.execute("BEGIN IMMEDIATE")
                if not con.execute("SELECT 1 FROM assignments WHERE rater_id=? AND case_id=?",
                                   (rater, case_id)).fetchone():
                    raise KeyError("평가 항목을 찾을 수 없습니다.")
                previous = con.execute("SELECT * FROM reviews WHERE rater_id=? AND case_id=?", (rater, case_id)).fetchone()
                version = previous["version"] if previous else 0
                if version != update.expected_version:
                    raise ConflictError("다른 창에서 저장된 평가가 있습니다. 현재 메모를 복사한 뒤 다시 불러와 주세요.")
                values = (rater, case_id, version + 1, update.status,
                          encode(update.ratings.model_dump()), update.notes, now())
                con.execute("INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?,?)", values)
                con.execute("INSERT INTO history (rater_id,case_id,version,status,ratings,notes,updated_at) VALUES (?,?,?,?,?,?,?)", values)
                saved = con.execute("SELECT * FROM reviews WHERE rater_id=? AND case_id=?", (rater, case_id)).fetchone()
        return public_review(saved)

    def export_rater(self, rater):
        with self.connect() as con:
            rows = con.execute("""SELECT r.* FROM reviews r JOIN assignments a USING (rater_id,case_id)
                                  WHERE r.rater_id=? ORDER BY a.position""", (rater,)).fetchall()
        return [{"study_id": self.meta["study_id"], "rater_id": rater, "case_id": r["case_id"],
                 **public_review(r)} for r in rows]

    def export_all(self, output_dir):
        """Owner-only CLI export. This is never an HTTP route."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        output_dir.chmod(0o700)
        counts = {}
        # A single read transaction keeps current ratings and history mutually consistent.
        with self.connect() as con:
            con.execute("BEGIN")
            raters = con.execute("SELECT rater_id FROM raters ORDER BY rater_id").fetchall()
            for entry in raters:
                rid = entry["rater_id"]
                rows = con.execute("SELECT * FROM reviews WHERE rater_id=? ORDER BY case_id", (rid,)).fetchall()
                data = [{"study_id": self.meta["study_id"], "rater_id": rid, "case_id": r["case_id"], **public_review(r)} for r in rows]
                path = output_dir / f"{rid}.jsonl"
                path.write_text("".join(encode(r) + "\n" for r in data), encoding="utf-8")
                path.chmod(0o600)
                counts[rid] = {"submitted": sum(r["status"] == "submitted" for r in data), "saved": len(data)}
            history = [{**dict(r), "ratings": json.loads(r["ratings"])} for r in con.execute("SELECT * FROM history ORDER BY id")]
            write_private(output_dir / "history.private.json", history)
        write_private(output_dir / "manifest.private.json", json.loads((self.directory / "manifest.private.json").read_text()))
        write_private(output_dir / "export.json", {"study_id": self.meta["study_id"], "exported_at": now(), "raters": counts})
        return counts
