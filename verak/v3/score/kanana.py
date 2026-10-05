"""Greedy first-line scores and raw teacher-forced digit expectations.

Digit renormalization follows essay_scoring_llm.soft_sc.DigitDistributionHelper,
restricted to the nine single-digit IDs. Generation scores, top-k, 01/001
tokens, and feedback parsing are deliberately absent from this backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import re
import sqlite3

from ..common import file_sha, pair_key, read_json, sha_text

SYSTEM_PROMPT = (
    "에세이 채점기. 8개 루브릭(과제충실성, 설명명료성, 설명구체성, 설명적절성, "
    "문장연결성, 글통일성, 어휘적절성, 어법적절성)별 1-9점 채점한다. "
    "공백으로 구분한 정수 8개를 한 줄로만 출력한다. 피드백이나 설명은 출력하지 않는다."
)
DIGIT_IDS = tuple(range(16, 25))
SCORE_LINE = re.compile(r"[ \t]*([1-9])(?:[ \t]+([1-9])){7}[ \t]*\Z")
SCORER_VERSION = "phase1-raw-teacher-forced-digits-v1"


class InputTooLong(ValueError):
    def __init__(self, tokens, limit):
        self.tokens, self.limit = tokens, limit
        super().__init__(f"Input has {tokens} tokens, exceeding {limit}; not truncated")


class ScoreParseError(ValueError):
    pass


@dataclass(frozen=True)
class ScoreResult:
    expected: list[float]
    integers: list[int]
    mean: float
    genre: str
    cache_hit: bool
    input_tokens: int
    score_line: str
    cache_key: str

    def to_dict(self):
        return asdict(self)


def parse_first_line(line):
    if not SCORE_LINE.fullmatch(line):
        raise ScoreParseError("Expected exactly eight single-digit scores on the first line")
    return [int(part) for part in line.split()]


def expected_from_digit_logits(logits):
    """Raw nine-way softmax, using float64 CPU arithmetic; no clipping/warping."""
    import torch
    if logits.shape != (8, 9) or not torch.isfinite(logits).all():
        raise ValueError("Expected finite raw logits with shape (8, 9)")
    probabilities = torch.softmax(logits.detach().to(device="cpu", dtype=torch.float64), dim=-1)
    values = torch.arange(1, 10, dtype=torch.float64)
    expected = (probabilities * values).sum(-1).tolist()
    integers = (probabilities.argmax(-1) + 1).tolist()
    return expected, integers


def teacher_positions(prefix_length, line_ids, digit_ids=DIGIT_IDS):
    positions = [prefix_length + i - 1 for i, token in enumerate(line_ids) if token in digit_ids]
    if len(positions) != 8 or min(positions) < 0:
        raise ScoreParseError("Score line must contain exactly eight canonical digit tokens")
    return positions


class ScoreCache:
    def __init__(self, path, fingerprint):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        previous = self.db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
        if previous and previous[0] != fingerprint:
            self.db.close()
            raise ValueError("Score cache belongs to a different model or scoring configuration")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('fingerprint', ?)", (fingerprint,))
        self.db.execute("CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.commit()

    def get(self, key):
        row = self.db.execute("SELECT value FROM scores WHERE key=?", (key,)).fetchone()
        return ScoreResult(**json.loads(row[0])) if row else None

    def put(self, result):
        payload = json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)
        self.db.execute("INSERT OR REPLACE INTO scores VALUES (?, ?)", (result.cache_key, payload))
        self.db.commit()

    def close(self):
        self.db.close()


class KananaScorer:
    def __init__(self, config, *, model=None, tokenizer=None, cache_enabled=None):
        self.config = config
        self.limit = config["scorer"]["max_input_tokens"]
        self.max_new_tokens = config["scorer"]["max_new_tokens"]
        paths = config["paths"]
        self.genres = read_json(paths["metadata"] / "genres.json")["questions"]
        self.model, self.tokenizer = model, tokenizer
        if (model is None) != (tokenizer is None):
            raise ValueError("Supply model and tokenizer together")
        if self.model is None:
            from .loading import load_frozen_model
            self.model, self.tokenizer = load_frozen_model(paths["policy_base"], paths["scorer_adapter"],
                gpu=config["scorer"]["gpu"], seed=config["scorer"]["seed"],
                load_in_4bit=config["scorer"]["load_in_4bit"])
        for digit, expected_id in enumerate(DIGIT_IDS, 1):
            if self.tokenizer.encode(str(digit), add_special_tokens=False) != [expected_id]:
                raise ValueError("Scorer tokenizer does not match the approved digit IDs")
        self.model.eval()
        fingerprints = {"version": SCORER_VERSION, "system": SYSTEM_PROMPT,
            "settings": config["scorer"], "base": str(paths["policy_base"]),
            "adapter": str(paths["scorer_adapter"])}
        for name, path in (("adapter_weights", paths["scorer_adapter"] / "adapter_model.safetensors"),
                           ("adapter_config", paths["scorer_adapter"] / "adapter_config.json"),
                           ("base_config", paths["policy_base"] / "config.json")):
            if path.is_file():
                fingerprints[name] = file_sha(path)
        self.fingerprint = sha_text(json.dumps(fingerprints, sort_keys=True, ensure_ascii=False))
        enabled = config["scorer"].get("cache", True) if cache_enabled is None else cache_enabled
        self.cache = ScoreCache(paths["output"] / "score_cache.sqlite", self.fingerprint) if enabled else None

    def prepare_input(self, question, text):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"질문: {question}\n에세이: {text}"}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=False,
                                 padding=False, add_special_tokens=False)
        length = int(encoded["input_ids"].shape[1])
        if length > self.limit:
            raise InputTooLong(length, self.limit)
        return encoded

    def input_tokens(self, question, text):
        return int(self.prepare_input(question, text)["input_ids"].shape[1])

    def score(self, question: str, text: str, *, use_cache=True) -> ScoreResult:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList
        if not question.strip() or not text.strip():
            raise ValueError("Question and essay must be nonblank")
        qhash = sha_text(question)
        if qhash not in self.genres:
            raise ValueError("Question requires a cached genre label before scoring")
        genre, key = self.genres[qhash]["genre"], pair_key(question, text)
        encoded = self.prepare_input(question, text)
        length = int(encoded["input_ids"].shape[1])
        if self.cache and use_cache:
            hit = self.cache.get(key)
            if hit is not None:
                return replace(hit, cache_hit=True, genre=genre)
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        tokenizer = self.tokenizer

        class FirstNewline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                stopped = ["\n" in tokenizer.decode(row[length:], skip_special_tokens=True)
                           for row in input_ids]
                return torch.tensor(stopped, device=input_ids.device, dtype=torch.bool)

        with torch.inference_mode():
            generated = self.model.generate(**encoded, max_new_tokens=self.max_new_tokens,
                do_sample=False, num_beams=1, num_return_sequences=1, temperature=None,
                top_p=None, top_k=None, stopping_criteria=StoppingCriteriaList([FirstNewline()]),
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            new_ids = generated[0, length:].tolist()
            line = tokenizer.decode(new_ids, skip_special_tokens=True).split("\n", 1)[0].rstrip("\r")
            parse_first_line(line)
            line_ids = tokenizer.encode(line, add_special_tokens=False)
            positions = teacher_positions(length, line_ids)
            continuation = torch.tensor([line_ids], dtype=encoded["input_ids"].dtype,
                                        device=self.model.device)
            ids = torch.cat([encoded["input_ids"], continuation], dim=1)
            # Position p-1 predicts digit p. No generate() scores enter the expectation.
            output = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            selected = output.logits[0, positions][:, list(DIGIT_IDS)]
            expected, integers = expected_from_digit_logits(selected)
        result = ScoreResult(expected, integers, sum(expected) / 8, genre, False, length, line, key)
        if self.cache:
            self.cache.put(result)
        return result

    def close(self):
        if self.cache:
            self.cache.close()
