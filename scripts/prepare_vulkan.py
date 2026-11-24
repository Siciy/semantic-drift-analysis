import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import yaml
from filelock import FileLock

from experiment import ROOT, write_json
from vulkan import LLAMA_RELEASE, SERVER_EXE, VULKAN_DIR, file_sha256, model_path, snapshot_path, tokenizer_hashes


BINARY_ARCHIVE = f"llama-{LLAMA_RELEASE}-bin-win-vulkan-x64.zip"
SOURCE_ARCHIVE = f"llama-{LLAMA_RELEASE}-source.zip"
BINARY_SHA256 = "4a0732dd3b68ea1d7346fde885253a315c0c5eb7279da897d936892e0f709108"
SOURCE_SHA256 = "fbec8d865a8e55a453442cbe524c436f71bd72eb2380532795e8df0ca66f62db"
CONVERTER = VULKAN_DIR / f"llama.cpp-{LLAMA_RELEASE}" / "convert_hf_to_gguf.py"


def extract_release(name, url, expected_hash, target):
    archive = VULKAN_DIR / name
    if not archive.is_file() or file_sha256(archive) != expected_hash:
        print(f"Downloading {name}", flush=True)
        partial = archive.with_suffix(".part")
        with urlopen(url, timeout=60) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if file_sha256(partial) != expected_hash:
            raise ValueError(f"Checksum mismatch for {name}")
        partial.replace(archive)
    with ZipFile(archive) as files:
        files.extractall(target)


def prepare_model(spec):
    from huggingface_hub import snapshot_download

    target = model_path(spec)
    stamp = target.with_suffix(".json")
    provenance = {"model": spec, "converter": LLAMA_RELEASE, "precision": "f16"}
    local_snapshot = snapshot_path(spec)
    if target.is_file() and stamp.is_file() and (local_snapshot / "tokenizer_config.json").is_file():
        saved = json.loads(stamp.read_text(encoding="utf-8"))
        if saved == {**provenance, "sha256": file_sha256(target), "tokenizer_sha256": tokenizer_hashes(spec)}:
            print(f"Ready: {spec['id']}", flush=True)
            return

    print(f"Preparing {spec['id']} at {spec['revision']}", flush=True)
    snapshot = snapshot_download(
        spec["id"], revision=spec["revision"], local_dir=local_snapshot,
        allow_patterns=["config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
                        "vocab.json", "merges.txt", "*.safetensors", "*.safetensors.index.json", "*.jinja"],
    )
    temporary = target.with_suffix(".part.gguf")
    subprocess.run(
        [sys.executable, str(CONVERTER), snapshot, "--outfile", str(temporary), "--outtype", "f16"], check=True,
    )
    provenance["sha256"] = file_sha256(temporary)
    provenance["tokenizer_sha256"] = tokenizer_hashes(spec)
    temporary.replace(target)
    write_json(stamp, provenance)
    print(f"Ready: {spec['id']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare the pinned Windows Vulkan backend and FP16 models.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "pilot.yaml")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This setup script installs the Windows x64 Vulkan release")
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "65")
    config = yaml.safe_load(args.config.read_bytes())
    VULKAN_DIR.mkdir(exist_ok=True)
    with FileLock(VULKAN_DIR / ".prepare.lock"):
        if not SERVER_EXE.is_file():
            extract_release(
                BINARY_ARCHIVE,
                f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/{BINARY_ARCHIVE}",
                BINARY_SHA256, SERVER_EXE.parent,
            )
        if not CONVERTER.is_file():
            extract_release(
                SOURCE_ARCHIVE, f"https://codeload.github.com/ggml-org/llama.cpp/zip/refs/tags/{LLAMA_RELEASE}",
                SOURCE_SHA256, VULKAN_DIR,
            )
        (VULKAN_DIR / "models").mkdir(exist_ok=True)
        for spec in config["models"].values():
            prepare_model(spec)


if __name__ == "__main__":
    main()
