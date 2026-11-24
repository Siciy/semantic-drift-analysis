import argparse
import csv
import hashlib
import json
from pathlib import Path

from experiment import expected_rounds, read_generations, run_lock, word_count, write_csv, write_json


SCORE_COLUMNS = (
    "split", "passage_id", "group", "schedule", "schedule_label", "round", "final_round",
    "retention_checkpoint", "status", "stop_reason", "model_id", "unchanged", "commentary", "truncated",
    "input_tokens", "output_tokens", "generation_seconds", "model_load_seconds", "warmup",
    "word_count", "original_word_count", "length_ratio", "encoder_tokens", "original_encoder_tokens",
    "similarity_status", "similarity",
)
ANNOTATION_COLUMNS = ("annotation_id", "rewrite", "claim", "label", "notes")
KEY_COLUMNS = ("annotation_id", "passage_id", "schedule", "round", "claim_id")


class Encoder:
    def __init__(self, spec, device):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "cpu":
            torch.set_num_threads(min(8, torch.get_num_threads()))
        self.model = SentenceTransformer(
            spec["id"], revision=spec["revision"], device=device,
            model_kwargs={"use_safetensors": True},
        )
        self.limit = spec["max_tokens"]
        if self.model.max_seq_length != self.limit:
            raise ValueError(f"Configured encoder limit {self.limit} disagrees with the model")

    def encode(self, texts):
        texts = list(dict.fromkeys(texts))
        lengths = {}
        for start in range(0, len(texts), 64):
            batch = texts[start:start + 64]
            encoded = self.model.tokenizer(
                batch, add_special_tokens=True, padding=False, truncation=False, verbose=False,
            )
            lengths.update((text, len(ids)) for text, ids in zip(batch, encoded["input_ids"], strict=True))
        eligible = [text for text in texts if lengths[text] <= self.limit]
        if not eligible:
            return lengths, {}
        vectors = self.model.encode(eligible, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return lengths, dict(zip(eligible, vectors, strict=True))


def score_rounds(design, generations, encoder):
    config = design["config"]
    originals = [p["text"] for p in design["passages"].values()]
    valid = [record["text"] for record in generations.values() if record["status"] == "valid"]
    lengths, vectors = encoder.encode([*originals, *valid])
    annotations = config["annotations"]
    broken_chains = set()

    for key in expected_rounds(design):
        schedule, passage_id, round_number = key
        passage = design["passages"][passage_id]
        original = passage["text"]
        spec = config["schedules"][schedule]
        record = generations.get(key, {})
        chain = schedule, passage_id
        status = record.get("status", "blocked" if chain in broken_chains else "pending")
        if status == "invalid":
            broken_chains.add(chain)
        text = record["text"] if status in {"original", "valid"} else None
        original_words = word_count(original)
        words = word_count(text) if text is not None else None
        flags = record.get("flags", {})
        sampled_checkpoint = (
            passage_id in annotations["early_passages"]
            and (round_number in annotations["early_rounds"] or round_number == spec["rounds"])
        )
        similarity = None
        similarity_status = "no_valid_text"
        if text is not None:
            if original not in vectors or text not in vectors:
                similarity_status = "encoder_limit"
            else:
                cosine = float(vectors[original] @ vectors[text])
                similarity = 1.0 if text == original else max(-1.0, min(1.0, cosine))
                similarity_status = "scored"

        yield {
            "split": config["split"], "passage_id": passage_id, "group": passage["group"],
            "schedule": schedule, "schedule_label": spec["label"], "round": round_number,
            "final_round": spec["rounds"], "retention_checkpoint": sampled_checkpoint,
            "status": status, "stop_reason": record.get("stop_reason"), "model_id": record.get("model_id"),
            "unchanged": flags.get("unchanged"), "commentary": flags.get("commentary"),
            "truncated": flags.get("truncated"), "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"), "generation_seconds": record.get("generation_seconds"),
            "model_load_seconds": record.get("model_load_seconds"), "warmup": record.get("warmup"),
            "word_count": words, "original_word_count": original_words,
            "length_ratio": words / original_words if words is not None else None,
            "encoder_tokens": lengths.get(text), "original_encoder_tokens": lengths[original],
            "similarity_status": similarity_status, "similarity": similarity,
        }


def annotation_rows(design, generations):
    config = design["config"]
    rules = config["annotations"]
    rows = {}
    for (schedule, passage_id, round_number), record in generations.items():
        if record["status"] != "valid":
            continue
        final = round_number == config["schedules"][schedule]["rounds"]
        early = round_number in rules["early_rounds"] and passage_id in rules["early_passages"]
        if not (final or early):
            continue
        for claim in design["passages"][passage_id]["claims"]:
            key = schedule, passage_id, round_number, claim["id"], record["text"], claim["text"]
            annotation_id = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
            blinded = {"annotation_id": annotation_id, "rewrite": record["text"], "claim": claim["text"]}
            lookup = {"annotation_id": annotation_id, "passage_id": passage_id,
                      "schedule": schedule, "round": round_number, "claim_id": claim["id"]}
            rows[annotation_id] = blinded, lookup
    return rows


def save_annotations(run_dir, rows):
    path = run_dir / "annotations.csv"
    existing = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                annotation_id = row["annotation_id"]
                if annotation_id in existing or annotation_id not in rows:
                    raise ValueError(f"Existing annotation ID is duplicate or does not belong to this run: {annotation_id}")
                existing[annotation_id] = row

    blinded_rows = []
    key_rows = []
    for annotation_id, (blinded, lookup) in sorted(rows.items()):
        review = existing.get(annotation_id, {})
        blinded_rows.append({**blinded, "label": review.get("label", ""), "notes": review.get("notes", "")})
        key_rows.append(lookup)
    write_csv(path, ANNOTATION_COLUMNS, blinded_rows)
    write_csv(run_dir / "annotation_key.csv", KEY_COLUMNS, key_rows)


def main():
    parser = argparse.ArgumentParser(description="Score a saved experiment and prepare blinded annotation rows.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run_dir = args.run_dir

    with run_lock(run_dir):
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        design = manifest["design"]
        generations = read_generations(run_dir, design)
        if not generations:
            raise ValueError("No generations have been saved for this run")
        encoder = Encoder(design["config"]["encoder"], args.device)
        write_csv(run_dir / "scores.csv", SCORE_COLUMNS, score_rounds(design, generations, encoder))
        save_annotations(run_dir, annotation_rows(design, generations))
        write_json(run_dir / "scoring.json", {"encoder": design["config"]["encoder"], "device": args.device})
    print(f"Wrote scores and annotation files in {run_dir}")


if __name__ == "__main__":
    main()
