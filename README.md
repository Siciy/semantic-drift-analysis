# AI Telephone Game

This experiment measures how meaning, length, and specific details change when a language model repeatedly rewrites a passage. Round 0 is the original. Each later round receives only the previous round's passage and the same rewrite instructions. The [experiment plan](experiment-plan.md) describes the study design.

## Setup

Use Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/). From this directory, install the locked dependencies with `uv sync --locked`. The first model call downloads the pinned weights to the Hugging Face cache. Setting `HF_HOME` to a local cache directory lets you control where those weights are stored.

Generation selects CUDA if PyTorch can use it and otherwise uses the CPU. Pass `--device cpu` for the smoke test or `--device cuda` to require a CUDA-enabled GPU. The same scripts run on Windows and on a course or hosted Linux GPU. Use the same environment, design, and code commit when resuming a run.

## Windows Radeon pilot

On a fresh environment, run `uv sync --locked --extra vulkan` first. The setup script downloads the pinned llama.cpp Windows Vulkan build and converts the two pinned Hugging Face models to FP16 GGUF. It stores the files in `.llama-cpp/` and reuses them on subsequent runs. Setup requires an internet connection for the first download.

```powershell
$env:HF_HOME = Join-Path $PWD '.hf-cache'
.\.venv\Scripts\python.exe scripts/prepare_vulkan.py
.\.venv\Scripts\python.exe scripts/run_experiment.py --config configs/pilot.yaml --run-id pilot-radeon-local --device vulkan --vulkan-gpu 'RX 6700S'
.\.venv\Scripts\python.exe scripts/score_results.py runs/pilot-radeon-local
.\.venv\Scripts\python.exe scripts/plot_results.py runs/pilot-radeon-local
```

The progress bar shows the current schedule and round, elapsed time, estimated time remaining, and counts of valid, invalid, and skipped rewrites. Skipped rounds belong to chains stopped by an invalid reply. Rerun the same command to resume from saved results. Runs using the same Vulkan device wait for exclusive use of the GPU. `vulkan-server.log` records which GPU loaded each model and confirms layer offload. Use the same backend and precision for all four main-study schedules.

## Two-round smoke test

```sh
uv run --locked python scripts/run_experiment.py --run-id smoke --schedule qwen_only --passage pilot-01 --max-round 2 --device cpu
uv run --locked python scripts/score_results.py runs/smoke
uv run --locked python scripts/plot_results.py runs/smoke
```

The saved [CPU smoke test](examples/cpu-smoke/) can be replotted with `uv run --locked python scripts/plot_results.py examples/cpu-smoke`. Later rounds appear as pending because the smoke test ends at round 2 of the pilot's five-round Qwen schedule.

## Pilot and main run

The pilot contains six passages and 60 single-model rewrites. The additional two-round alternating chain uses `pilot-01` to check the Qwen-to-SmolLM2 handoff. The main run contains 24 different passages across event notices, technical explanations, and narratives. It schedules 96 chains and 1,536 rewrites across Qwen only, SmolLM2 only, and both orders of alternation.

```sh
uv run --locked python scripts/run_experiment.py --config configs/pilot.yaml --run-id pilot --device cuda
uv run --locked python scripts/score_results.py runs/pilot
uv run --locked python scripts/plot_results.py runs/pilot
```

Review the pilot's raw responses, truncation rate, and warmed-up generation times for each model. The starting cap is 256 generated tokens. Once the pilot establishes a suitable cap, record it in `configs/main.yaml` and freeze the main settings before generation. Estimate runtime using warmed-up generation times, model loading, and evaluation. For the main run, use `configs/main.yaml`, change `--run-id` to `main`, and run the scoring and plotting commands with `runs/main`.

Repeat the original generation command to resume. `--max-round` limits an invocation and can be increased when resuming. Use a new run ID to change the prompt, model revisions, dataset, schedule selection, or runtime. `--passage` and `--schedule` can each be repeated to narrow a run. Runs are stored in `runs/` by default. Preserve the entire run directory for any reported results.

## Generation records and scores

`manifest.json` stores the experimental design, original passages and claims, input file hashes, model revisions, decoding strategy, code commit when available, packages, and hardware. `generations.jsonl` stores the original and every saved reply, including the input passage, rendered chat prompt, raw continuation, scoreable text, generation time, and validity flags. A technical failure is retried under the same passage, schedule, and round. `errors.jsonl` records retry attempts. After two technical failures the command exits so the same run can be resumed once the error is resolved.

A reply that is empty, contains obvious assistant commentary, or reaches the token cap without emitting an end token stops that chain. The runner saves the raw reply and stop reason. It uses fixed phrase patterns for obvious commentary, so the pilot review should check their behavior. A valid reply that repeats the input is counted. The log retains stop reasons, incomplete chains, and exact output for inspection.

`scores.csv` contains one row per planned chain and round. A round is reached when it has a saved record, including the terminal invalid reply. Subsequent rounds of that chain are blocked. Unattempted or interrupted rounds with no saved reply are pending. Metrics use valid passages only, with round 0 as the baseline. The word-count rule counts letters and digits and treats internal hyphens and apostrophes as part of a word. Similarity is cosine similarity to the original using all-MiniLM-L6-v2. The encoder check counts special tokens as part of its 256-token budget. A passage above that budget retains its length score and receives a blank similarity score.

## Human annotations

Share `annotations.csv` with reviewers. It contains an opaque ID, the rewrite, one original claim, and the label and notes columns. Keep `annotation_key.csv` with the study materials to join labels to the original passage, schedule, round, and claim IDs. Leave unreviewed labels blank. Keep the CSV headers and IDs intact.

- Use `preserved` when the rewrite supports the claim, including relevant qualifiers.
- Use `contradicted` when it gives conflicting information, such as a different start time.
- Use `missing` when it omits the claim or part of the information needed to establish it.

For the main study, review all valid round-16 outputs and the valid outputs at rounds 1 and 8 for the six passages specified in the main config. Give two teammates the same subset to label independently with the schedule hidden. Resolve disagreements and combine the labels into an agreed CSV. Run `plot_results.py` with `--annotations path/to/agreed.csv` to use that file. Retention is calculated only for rewrites with every claim reviewed.

## Reports

- `round_summary.csv` and `group_round_summary.csv` give the number of expected, reached, valid, and scored passages and the mean metrics by round. Check these counts when comparing curves, since early termination changes which passages contribute to later averages.
- `timing_summary.csv` gives the truncation rate and warmed-up generation times by model. The first generation after each model load is recorded as warm-up.
- `paired_final_summary.csv` gives the difference between alternating and single-model means at round 16, calculated across passages with all four outcomes for that metric. It reports the paired sample size for every metric. The comparison measures alternating schedules against single-model schedules under this prompt.
- `claim_retention.csv` and `retention_summary.csv` summarize labels and retention at the review checkpoints. The retention plot uses the same six-passage subset at all three main-study checkpoints.
- `figures/` contains the similarity and length plots, plus a retention plot once complete human labels are available.
