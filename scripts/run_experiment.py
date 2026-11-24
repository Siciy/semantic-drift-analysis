import argparse
import json
import os
import platform
import re
import subprocess
from contextlib import nullcontext
from importlib import metadata
from pathlib import Path
from time import perf_counter

from tqdm import tqdm

from experiment import (
    ROOT, append_jsonl, expected_rounds, load_design, read_generations, run_lock,
    scheduled_model, utc_now, write_json,
)


COMMENTARY_START = re.compile(
    r"^(?:sure|certainly|of course)[!,.]\s*(?:$|\n|here\b|i(?:'|’)ve\b|i can\b|"
    r"this\b|the rewritten\b)|"
    r"^(?:here(?:'s|’s| is) (?:a |the |my |your )?(?:requested )?"
    r"(?:rewrit(?:e|ten)|paraphras|rephras|revised (?:passage|version|text))|"
    r"(?:(?:rewritten|paraphrased|rephrased|revised) (?:passage|version|paragraph|text)|"
    r"rewrite|paraphrase)\s*:|i (?:cannot|can['’]t|am unable to) (?:rewrite|paraphrase)\b)",
    re.IGNORECASE,
)
COMMENTARY_END = re.compile(
    r"^\s*(?:let me know if you(?:'d|’d| would| want| need)|"
    r"i hope (?:this|that) helps|(?:this|the) (?:rewritten version|rewrite|paraphrase) "
    r"(?:retains|preserves|maintains|keeps))\b",
    re.IGNORECASE | re.MULTILINE,
)


def output_flags(text, input_text, truncated):
    commentary_text = re.sub(r"^[ \t]*(?:\*\*|__|#{1,6}[ \t]+)", "", text, flags=re.MULTILINE)
    return {
        "truncated": truncated,
        "empty": not text,
        "commentary": bool(COMMENTARY_START.search(commentary_text) or COMMENTARY_END.search(commentary_text)),
        "unchanged": text == input_text,
    }


class Rewriter:
    def __init__(self, device_name, vulkan=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.torch = torch
        self.tokenizer_class = AutoTokenizer
        self.model_class = AutoModelForCausalLM
        self.generation_config_class = GenerationConfig
        self.vulkan = vulkan
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA mode requires a CUDA-enabled PyTorch installation and GPU")
        self.device = torch.device(device_name if vulkan is None else "cpu")
        self.dtype = torch.float32
        if vulkan is not None:
            self.dtype = torch.float16
        elif device_name == "cuda":
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch.set_num_threads(min(8, os.cpu_count() or 1))
        self.active = None
        self.model = None
        self.tokenizer = None

    def runtime(self):
        torch = self.torch
        runtime = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "cpu_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name() if self.device.type == "cuda" else None,
            "cuda": torch.version.cuda,
            "packages": {package: metadata.version(package) for package in (
                "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors",
                "sentence-transformers", "pandas", "matplotlib", "PyYAML", "filelock",
            )},
        }
        if self.vulkan is not None:
            runtime.update(self.vulkan.runtime(), device="vulkan")
        return runtime

    def load(self, spec):
        model_key = spec["id"], spec["revision"]
        if self.active == model_key and (self.vulkan is None or self.vulkan.is_running()):
            return

        self.active = None
        self.tokenizer = None
        self.model = None
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

        start = perf_counter()
        if self.vulkan is not None:
            from vulkan import snapshot_path

            tokenizer = self.tokenizer_class.from_pretrained(snapshot_path(spec), local_files_only=True)
            self.vulkan.load(spec)
        else:
            tokenizer = self.tokenizer_class.from_pretrained(spec["id"], revision=spec["revision"])
            self.model = self.model_class.from_pretrained(
                spec["id"], revision=spec["revision"], dtype=self.dtype, use_safetensors=True,
            ).to(self.device).eval()
        self.tokenizer = tokenizer
        self.active = model_key
        self.load_seconds = perf_counter() - start
        self.warmed_up = False
        device = self.vulkan.gpu if self.vulkan is not None else self.device
        tqdm.write(f"Loaded {spec['id']} on {device} in {self.load_seconds:.1f}s")

    def rewrite(self, spec, prompt, passage, max_tokens):
        self.load(spec)
        tokenizer = self.tokenizer
        messages = [{"role": "user", "content": f"{prompt}\n\n{passage}"}]
        rendered_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        start = perf_counter()
        if self.vulkan is not None:
            input_ids = tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]
            input_tokens = len(input_ids)
            completion = self.vulkan.complete(input_ids, max_tokens)
            continuation = completion["tokens"]
            termination = "eos" if completion["stop_type"] == "eos" else "token_cap"
            raw_output = completion["content"]
        else:
            input_tokens, continuation, termination = self.generate_transformers(rendered_prompt, max_tokens)
            raw_output = tokenizer.decode(continuation, clean_up_tokenization_spaces=False)
        generation_seconds = perf_counter() - start

        text = tokenizer.decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        record = {
            "messages": messages,
            "rendered_prompt": rendered_prompt,
            "input_tokens": input_tokens,
            "output_tokens": len(continuation),
            "raw_output": raw_output,
            "output_token_ids": continuation,
            "text": text,
            "termination": termination,
            "flags": output_flags(text, passage, termination == "token_cap"),
            "generation_seconds": generation_seconds,
            "model_load_seconds": None if self.warmed_up else self.load_seconds,
            "warmup": not self.warmed_up,
        }
        self.warmed_up = True
        return record

    def generate_transformers(self, rendered_prompt, max_tokens):
        tokenizer = self.tokenizer
        inputs = tokenizer(rendered_prompt, add_special_tokens=False, return_tensors="pt").to(self.device)
        defaults = self.model.generation_config
        eos = defaults.eos_token_id if defaults.eos_token_id is not None else tokenizer.eos_token_id
        generation_config = self.generation_config_class(
            do_sample=False, max_new_tokens=max_tokens, eos_token_id=eos,
            bos_token_id=defaults.bos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        input_tokens = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, generation_config=generation_config)
        continuation = result[0, input_tokens:].tolist()
        eos_ids = eos if isinstance(eos, list) else [eos]
        ended = bool(continuation and continuation[-1] in eos_ids)
        termination = "eos" if ended else "token_cap" if len(continuation) >= max_tokens else "other"
        return input_tokens, continuation, termination


def code_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def generate_round(run_dir, config, rewriter, key, input_text):
    schedule, passage_id, round_number = key
    model_key = scheduled_model(config, schedule, round_number)
    spec = config["models"][model_key]
    record = {
        "schedule": schedule, "passage_id": passage_id, "round": round_number,
        "input_text": input_text, "model_id": spec["id"], "model_revision": spec["revision"],
    }
    for attempt in (1, 2):
        try:
            response = rewriter.rewrite(spec, config["prompt"], input_text, config["max_new_tokens"])
            break
        except Exception as error:
            append_jsonl(run_dir / "errors.jsonl", {
                "schedule": schedule, "passage_id": passage_id, "round": round_number,
                "model_id": spec["id"], "attempt": attempt, "time": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            })
            if attempt == 2:
                raise RuntimeError(f"Generation failed twice for {key}. Retry the same run to resume.") from error
            tqdm.write(f"Retrying {key} after {type(error).__name__}")

    flags = response["flags"]
    reasons = [reason for reason in ("truncated", "empty", "commentary") if flags[reason]]
    return {
        **record, **response, "time": utc_now(), "attempt": attempt,
        "status": "invalid" if reasons else "valid",
        "stop_reason": ",".join(reasons) if reasons else None,
    }


def run_experiment(run_dir, design, rewriter, max_round):
    run_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(run_dir):
        path = run_dir / "manifest.json"
        runtime = rewriter.runtime()
        commit = code_commit()
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest["design"] != design or manifest["runtime"] != runtime or manifest["code_commit"] != commit:
                raise ValueError("This run used a different design, code commit, or runtime. Use its original setup or a new run ID.")
        else:
            if (run_dir / "generations.jsonl").exists():
                raise ValueError("Existing generations have no manifest")
            write_json(path, {
                "schema_version": 1, "run_id": run_dir.name, "created_at": utc_now(),
                "code_commit": commit, "design": design, "runtime": runtime,
                "decoding": "greedy",
            })

        config = design["config"]
        records = read_generations(run_dir, design, repair_tail=True)
        planned = [key for key in expected_rounds(design) if max_round is None or key[2] <= max_round]
        rewrites = [key for key in planned if key[2] > 0]
        valid = sum(key in records and records[key]["status"] == "valid" for key in rewrites)
        invalid = sum(key in records and records[key]["status"] == "invalid" for key in rewrites)
        added = 0
        skipped = 0
        with tqdm(total=len(rewrites), initial=valid + invalid, desc="Rewrites", unit="rewrite",
                  ascii=True, dynamic_ncols=True, disable=None,
                  bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}") as progress:
            for key in planned:
                schedule, passage_id, round_number = key
                if key in records:
                    continue
                if round_number == 0:
                    record = {
                        "schedule": schedule, "passage_id": passage_id, "round": 0,
                        "status": "original", "text": design["passages"][passage_id]["text"], "time": utc_now(),
                    }
                    append_jsonl(run_dir / "generations.jsonl", record)
                    records[key] = record
                    continue

                previous = records.get((schedule, passage_id, round_number - 1))
                if previous is None or previous["status"] == "invalid":
                    skipped += 1
                else:
                    schedule_name = config["schedules"][schedule]["label"]
                    progress.set_postfix_str(f"{schedule_name}, {passage_id}, round {round_number}")
                    record = generate_round(run_dir, config, rewriter, key, previous["text"])
                    append_jsonl(run_dir / "generations.jsonl", record)
                    records[key] = record
                    added += 1
                    valid += record["status"] == "valid"
                    invalid += record["status"] == "invalid"
                    if record["stop_reason"]:
                        tqdm.write(f"Stopped {schedule} {passage_id} at round {round_number}: {record['stop_reason']}")
                progress.set_postfix_str(f"{valid} valid, {invalid} invalid, {skipped} skipped", refresh=False)
                progress.update()

        print(f"{run_dir}: {added} new rewrites, {valid} valid, {invalid} invalid, {skipped} skipped "
              f"of {len(rewrites)} planned", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run or resume the AI Telephone Game experiment.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pilot.yaml")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "passages.jsonl")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--schedule", action="append", default=[])
    parser.add_argument("--passage", action="append", default=[])
    parser.add_argument("--max-round", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "vulkan"), default="auto")
    parser.add_argument("--vulkan-gpu", help="Match a Vulkan GPU by name when multiple GPUs are present")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_id):
        parser.error("--run-id must be a single name with letters, digits, underscores, or hyphens")
    if args.max_round is not None and args.max_round < 1:
        parser.error("--max-round must be positive")
    if args.vulkan_gpu and args.device != "vulkan":
        parser.error("--vulkan-gpu requires --device vulkan")
    design = load_design(args.config, args.dataset, args.passage, args.schedule)
    run_dir = args.runs_dir / args.run_id
    vulkan = None
    if args.device == "vulkan":
        from vulkan import VulkanBackend

        os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
        model_keys = dict.fromkeys(
            model for schedule in design["chains"] for model in design["config"]["schedules"][schedule]["models"]
        )
        specs = [design["config"]["models"][model] for model in model_keys]
        vulkan = VulkanBackend(args.vulkan_gpu, specs, run_dir)
    with vulkan or nullcontext():
        run_experiment(run_dir, design, Rewriter(args.device, vulkan), args.max_round)


if __name__ == "__main__":
    main()
