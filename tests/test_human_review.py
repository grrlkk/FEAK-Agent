"""Blinding, durability, and reviewer isolation are research-data contracts."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pydantic import ValidationError
import pytest

from feak_tc.review.server import make_server
from feak_tc.review.store import CRITERIA, ConflictError, ReviewStore, prepare_study


@pytest.fixture
def source(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "inputs.jsonl").write_text(json.dumps({"sample_id": "SECRET_SAMPLE_ID", "writing_prompt": "공정한 역할 분담을 설명하시오."}) + "\n")
    rows = []
    for i, decision in enumerate(["ACCEPT", "REJECT", "ERROR"]):
        rows.append({"sample_id": "SECRET_SAMPLE_ID", "iteration": i + 1, "attempt": i,
                     "text_before": "모둠원은 함께 일한다. 역할 역할을 나눈다.",
                     "candidate": f"모둠원은 함께 일한다. 역할을 나눈다. {i}",
                     "plan": {"goal": "반복 표현을 간결하게 정리한다.", "must_preserve": ["함께 일한다는 주장"],
                              "problem": "SECRET_PROBLEM", "target_rubric": "SECRET_RUBRIC"},
                     "rv": {"decision": decision, "reason": "SECRET_RV_REASON"},
                     "acceptance_decision": decision, "scores_before": {"secret_score": 918273},
                     "expected_label": "SECRET_EXPECTED_LABEL", "model": "SECRET_MODEL_NAME"})
    rows.append({"sample_id": "SECRET_SAMPLE_ID", "candidate": None})
    (directory / "trajectories.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return directory


@pytest.fixture
def study(source, tmp_path):
    path = tmp_path / "study"
    result = prepare_study(source, path, raters=2, seed=7)
    assert result["cases"] == 3 and result["excluded"] == 1
    credentials = json.loads((path / "reviewers.private.json").read_text())["reviewers"]
    return ReviewStore(path), credentials


def update(version=0, status="submitted"):
    return {"expected_version": version, "status": status,
            "ratings": {key: {"label": "PASS", "reason": f"{key}의 근거"} for key in CRITERIA}, "notes": "메모"}


def case_id(store, rater="R1"):
    return store.session(rater)["cases"][0]["case_id"]


def test_public_packet_allowlist_and_all_decisions_included(study):
    store, credentials = study
    all_text = json.dumps(store.session("R1"))
    for item in store.session("R1")["cases"]:
        response = store.case("R1", item["case_id"])
        assert set(response["case"]) == {"case_id", "writing_prompt", "before", "after", "goal", "must_preserve"}
        assert all(r["label"] is None for r in response["review"]["ratings"].values())
        all_text += json.dumps(response)
    assert "SECRET_" not in all_text and "918273" not in all_text
    assert store.authenticate(credentials[0]["token"])["rater_id"] == "R1"
    assert store.authenticate("invalid-token") is None
    assert store.authenticate(credentials[1]["token"])["rater_id"] == "R2"


def test_frozen_source_resume_and_no_overwrite(study, source):
    store, _ = study
    cid = case_id(store)
    before = store.case("R1", cid)
    with pytest.raises(FileExistsError):
        prepare_study(source, store.directory)
    (source / "trajectories.jsonl").write_text("changed after freeze")
    restarted = ReviewStore(store.directory)
    assert restarted.case("R1", cid) == before
    assert restarted.session("R1") == store.session("R1")


def test_per_rater_drafts_submissions_and_history_survive_restart(study, tmp_path):
    store, _ = study
    cid = case_id(store)
    draft = update(status="draft")
    draft["ratings"]["preservation"] = {"label": None, "reason": "작성 중"}
    assert store.save("R1", cid, draft)["version"] == 1
    assert store.session("R1")["completed"] == 0
    assert store.case("R2", cid)["review"]["version"] == 0
    restarted = ReviewStore(store.directory)
    assert restarted.case("R1", cid)["review"]["ratings"]["preservation"]["reason"] == "작성 중"
    assert restarted.save("R1", cid, update(1))["status"] == "submitted"
    assert restarted.session("R1")["completed"] == 1
    assert restarted.session("R2")["completed"] == 0
    assert restarted.export_rater("R2") == []
    exported = tmp_path / "export"
    counts = restarted.export_all(exported)
    assert counts["R1"] == {"saved": 1, "submitted": 1}
    assert (exported / "R2.jsonl").read_text() == ""
    history = json.loads((exported / "history.private.json").read_text())
    assert [r["version"] for r in history] == [1, 2]
    assert history[0]["ratings"]["preservation"]["label"] is None
    assert "token" not in json.dumps(json.loads((exported / "manifest.private.json").read_text()))


@pytest.mark.parametrize("mutation", ["empty_label", "blank_reason", "extra_field", "bad_label", "rater_spoof"])
def test_invalid_submission_never_changes_saved_data(study, mutation):
    store, _ = study
    data = update()
    if mutation == "empty_label": data["ratings"]["necessity"]["label"] = None
    if mutation == "blank_reason": data["ratings"]["necessity"]["reason"] = "  "
    if mutation == "extra_field": data["scores"] = [9]
    if mutation == "bad_label": data["ratings"]["necessity"]["label"] = "GOOD"
    if mutation == "rater_spoof": data["rater_id"] = "R2"
    with pytest.raises(ValidationError):
        store.save("R1", case_id(store), data)
    assert store.export_rater("R1") == []


def test_concurrent_tabs_cannot_silently_overwrite(study):
    store, _ = study
    cid = case_id(store)
    def save(note):
        data = update(); data["notes"] = note
        try:
            return store.save("R1", cid, data)
        except ConflictError:
            return "conflict"
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = list(executor.map(save, ["first tab", "second tab"]))
    assert result.count("conflict") == 1
    assert store.case("R1", cid)["review"]["version"] == 1
    with pytest.raises(KeyError):
        store.save("R1", "0" * 32, update())


def test_prepare_rejects_invalid_packets_before_creating_study(source, tmp_path):
    row = json.loads((source / "trajectories.jsonl").read_text().splitlines()[0])
    row["plan"]["must_preserve"] = [""]
    (source / "trajectories.jsonl").write_text(json.dumps(row))
    target = tmp_path / "invalid"
    with pytest.raises(ValueError):
        prepare_study(source, target)
    assert not target.exists()


@pytest.fixture
def http(study):
    store, credentials = study
    server = make_server(store, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    def request(path, *, rater=None, method="GET", data=None, origin=None):
        headers = {}
        if rater is not None: headers["Authorization"] = "Bearer " + credentials[rater]["token"]
        if data is not None: headers["Content-Type"] = "application/json"
        if origin: headers["Origin"] = origin
        req = Request(base + path, method=method, headers=headers,
                      data=json.dumps(data).encode() if data is not None else None)
        try:
            with urlopen(req, timeout=5) as response:
                return response.status, response.read(), response.headers
        except HTTPError as error:
            return error.code, error.read(), error.headers
    yield request, store
    server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_http_blinding_is_not_only_css_and_no_private_download_routes(http):
    request, store = http
    assert request("/api/session")[0] == 401
    for path in ["/", "/app.js", "/styles.css"]:
        code, content, headers = request(path)
        assert code == 200 and content
        assert headers["Cache-Control"] == "no-store"
    for path in ["/manifest.private.json", "/reviewers.private.json", "/review.sqlite3", "/api/raters", "/../store.py"]:
        assert request(path, rater=0)[0] == 404
    code, content, _ = request("/api/session", rater=0)
    assert code == 200 and b"SECRET_" not in content and b"token" not in content
    cid = case_id(store)
    code, content, _ = request(f"/api/cases/{cid}", rater=0)
    assert code == 200 and b"SECRET_" not in content
    assert request(f"/api/cases/{cid}/review", rater=0, method="PUT", data=update(), origin="https://other.invalid")[0] == 403


def test_http_save_conflict_and_export_is_only_own_work(http):
    request, store = http
    cid = case_id(store)
    endpoint = f"/api/cases/{cid}/review"
    assert request(endpoint, rater=0, method="PUT", data=update())[0] == 200
    assert request(endpoint, rater=0, method="PUT", data=update())[0] == 409
    assert request(endpoint, rater=1, method="PUT", data={**update(), "rater_id": "R1"})[0] == 400
    assert request("/api/export", rater=1)[1] == b""
    code, body, headers = request("/api/export", rater=0)
    assert code == 200
    assert json.loads(body)["rater_id"] == "R1"
    assert b"SECRET_" not in body and b"token" not in body
    assert "attachment" in headers["Content-Disposition"]
