import csv
import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from filelock import FileLock


ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"event", "technical", "narrative"}
LABELS = {"preserved", "contradicted", "missing"}
WORDS = re.compile(r"[^\W_]+(?:[’'-][^\W_]+)*")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def word_count(text):
    return len(WORDS.findall(text))


def load_design(config_path, dataset_path, passage_ids, schedule_ids):
    config_bytes = config_path.read_bytes()
    dataset_bytes = dataset_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    passages = {}
    for line in dataset_bytes.decode("utf-8").splitlines():
        passage = json.loads(line)
        passage_id = passage["id"]
        if passage_id in passages:
            raise ValueError(f"Duplicate passage ID {passage_id}")
        claims = passage["claims"]
        if not 3 <= len(claims) <= 4 or len({c["id"] for c in claims}) != len(claims):
            raise ValueError(f"{passage_id} must have three or four uniquely identified claims")
        if not all(claim["text"].strip() for claim in claims):
            raise ValueError(f"{passage_id} has an empty claim")
        if not 45 <= word_count(passage["text"]) <= 80:
            raise ValueError(f"{passage_id} must contain 45 to 80 words")
        passages[passage_id] = passage

    split = config["split"]
    passages = {key: p for key, p in passages.items() if p["split"] == split}
    counts = Counter(p["group"] for p in passages.values())
    if counts != {group: config["passages_per_group"] for group in GROUPS}:
        raise ValueError(f"Unexpected group sizes in the {split} split: {dict(counts)}")
    annotations = config["annotations"]
    if set(annotations["early_passages"]) - passages.keys():
        raise ValueError("An early annotation passage is outside the configured split")

    schedules = config["schedules"]
    for model in [*config["models"].values(), config["encoder"]]:
        if not re.fullmatch(r"[0-9a-f]{40}", model["revision"]):
            raise ValueError(f"Give {model['id']} a full model revision hash")
    if not config["prompt"].strip() or config["max_new_tokens"] < 1:
        raise ValueError("The prompt and generation token cap are required")
    for name, schedule in schedules.items():
        models = schedule["models"]
        if not models or set(models) - config["models"].keys() or schedule["rounds"] < 1:
            raise ValueError(f"Invalid model sequence or round count in {name}")
        if set(schedule.get("passages", ())) - passages.keys():
            raise ValueError(f"Schedule {name} refers to a passage outside the configured split")

    unknown_passages = set(passage_ids) - passages.keys()
    unknown_schedules = set(schedule_ids) - schedules.keys()
    if unknown_passages or unknown_schedules:
        raise ValueError(f"Unknown passages {sorted(unknown_passages)} or schedules {sorted(unknown_schedules)}")
    selected = {key: p for key, p in passages.items() if not passage_ids or key in passage_ids}
    chains = {}
    for name, schedule in schedules.items():
        if schedule_ids and name not in schedule_ids:
            continue
        eligible = schedule.get("passages", passages)
        matches = [key for key in selected if key in eligible]
        if matches:
            chains[name] = matches
    if not chains:
        raise ValueError("The selected passages and schedules have no chains in common")

    return {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "config": config,
        "passages": selected,
        "chains": chains,
        "expected_chains": sum(map(len, chains.values())),
        "expected_rewrites": sum(len(ids) * schedules[name]["rounds"] for name, ids in chains.items()),
    }


def expected_rounds(design):
    for schedule, passage_ids in design["chains"].items():
        last_round = design["config"]["schedules"][schedule]["rounds"]
        for round_number in range(last_round + 1):
            for passage_id in passage_ids:
                yield schedule, passage_id, round_number


def scheduled_model(config, schedule, round_number):
    models = config["schedules"][schedule]["models"]
    return models[(round_number - 1) % len(models)]


def run_lock(run_dir):
    return FileLock(run_dir / ".run.lock", timeout=0)


def read_generations(run_dir, design, repair_tail=False):
    path = run_dir / "generations.jsonl"
    if not path.exists():
        return {}

    records = {}
    expected = set(expected_rounds(design))
    with path.open("r+b" if repair_tail else "rb") as stream:
        while line := stream.readline():
            terminated = line.endswith(b"\n")
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if not terminated and repair_tail:
                    stream.truncate(stream.tell() - len(line))
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                raise ValueError(f"Invalid JSON in {path} after record {len(records)}") from error

            key = record["schedule"], record["passage_id"], record["round"]
            if key not in expected or key in records:
                raise ValueError(f"Unexpected or duplicate generation record {key}")
            schedule, passage_id, round_number = key
            if round_number == 0:
                original = design["passages"][passage_id]["text"]
                if record["status"] != "original" or record["text"] != original:
                    raise ValueError(f"Incorrect round zero for {key}")
            else:
                previous = records.get((schedule, passage_id, round_number - 1))
                if previous is None or previous["status"] not in {"original", "valid"}:
                    raise ValueError(f"Invalid or missing predecessor for {key}")
                model_key = scheduled_model(design["config"], schedule, round_number)
                model = design["config"]["models"][model_key]
                if (record["input_text"] != previous["text"] or record["model_id"] != model["id"]
                        or record["model_revision"] != model["revision"]):
                    raise ValueError(f"Saved input or model disagrees with the chain for {key}")
                if record["status"] not in {"valid", "invalid"}:
                    raise ValueError(f"Unexpected generation status for {key}")
            records[key] = record
            if not terminated and repair_tail:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
    return records


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, document):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path, columns, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
