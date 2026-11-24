import hashlib
import json
import re
import secrets
import socket
import subprocess
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

from filelock import FileLock, Timeout
from tqdm import tqdm

from experiment import ROOT


LLAMA_RELEASE = "b12676"
VULKAN_DIR = ROOT / ".llama-cpp"
SERVER_EXE = VULKAN_DIR / f"bin-{LLAMA_RELEASE}" / "llama-server.exe"
CONTEXT_TOKENS = 2048
DEVICE_LINE = re.compile(r"^\s*(Vulkan\d+):\s+(.+?)\s+\(\d+ MiB, .+\)\s*$", re.MULTILINE)
OFFLOAD_LINE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
WINDOW_FLAGS = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def model_path(spec):
    model_name = spec["id"].replace("/", "--")
    return VULKAN_DIR / "models" / f"{model_name}--{spec['revision']}--f16.gguf"


def snapshot_path(spec):
    model_name = spec["id"].replace("/", "--")
    return VULKAN_DIR / "snapshots" / f"{model_name}--{spec['revision']}"


def tokenizer_hashes(spec):
    snapshot = snapshot_path(spec)
    return {name: file_sha256(snapshot / name) for name in ("tokenizer.json", "tokenizer_config.json")}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class VulkanBackend:
    def __init__(self, gpu, specs, run_dir):
        if not SERVER_EXE.is_file():
            raise FileNotFoundError("Run scripts/prepare_vulkan.py to install the Windows Vulkan backend")
        self.executable = SERVER_EXE
        self.log_path = run_dir / "vulkan-server.log"
        self.process = None
        self.opener = build_opener(ProxyHandler({}))
        self.version = self.server_output("--version").splitlines()[0]
        available = DEVICE_LINE.findall(self.server_output("--list-devices"))
        matches = [(device, name) for device, name in available if not gpu or gpu.casefold() in name.casefold()]
        if len(matches) != 1:
            names = ", ".join(name for _, name in available) or "none"
            raise ValueError(f"Choose one Vulkan device with --vulkan-gpu. Detected devices: {names}")
        self.device, self.gpu = matches[0]
        self.gpu_lock = FileLock(VULKAN_DIR / f".{self.device}.lock")
        self.models = {}
        self.tokenizers = {}
        for spec in specs:
            path = model_path(spec)
            stamp = path.with_suffix(".json")
            if not path.is_file() or not stamp.is_file() or not snapshot_path(spec).is_dir():
                raise FileNotFoundError(f"Missing {spec['id']} GGUF. Run scripts/prepare_vulkan.py first.")
            provenance = json.loads(stamp.read_text(encoding="utf-8"))
            if (provenance["model"] != spec or provenance["converter"] != LLAMA_RELEASE
                    or provenance["precision"] != "f16" or provenance["sha256"] != file_sha256(path)
                    or provenance["tokenizer_sha256"] != tokenizer_hashes(spec)):
                raise ValueError(f"The prepared GGUF disagrees with its recorded revision or checksum: {path}")
            self.models[(spec["id"], spec["revision"])] = provenance["sha256"]
            self.tokenizers[spec["id"]] = provenance["tokenizer_sha256"]

    def server_output(self, option):
        result = subprocess.run(
            [str(self.executable), option], cwd=self.executable.parent, capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=True, timeout=30, creationflags=WINDOW_FLAGS,
        )
        return (result.stdout + result.stderr).strip()

    def runtime(self):
        return {
            "backend": "llama.cpp/vulkan", "gpu": self.gpu, "vulkan_device": self.device,
            "llama_cpp_version": self.version, "context_tokens": CONTEXT_TOKENS,
            "gguf_sha256": {name: digest for (name, _), digest in self.models.items()},
            "tokenizer_sha256": self.tokenizers,
        }

    def __enter__(self):
        try:
            self.gpu_lock.acquire(timeout=0)
        except Timeout:
            tqdm.write(f"Waiting for {self.gpu} to finish another experiment run.")
            self.gpu_lock.acquire()
        return self

    def __exit__(self, *_):
        try:
            self.close()
        finally:
            self.gpu_lock.release()

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def close(self):
        if self.is_running():
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None

    def request(self, route, timeout, body=None):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        request = Request(f"{self.url}{route}", data=data, headers=headers)
        with self.opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def load(self, spec):
        self.close()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with socket.socket() as port_reservation:
            port_reservation.bind(("127.0.0.1", 0))
            port = port_reservation.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.api_key = secrets.token_urlsafe(24)
        command = [
            str(self.executable), "--model", str(model_path(spec)), "--host", "127.0.0.1", "--port", str(port),
            "--device", self.device, "--gpu-layers", "999", "--fit", "off", "--ctx-size", str(CONTEXT_TOKENS),
            "--parallel", "1", "--no-context-shift", "--no-webui", "--verbosity", "4", "--api-key", self.api_key,
        ]
        with self.log_path.open("ab") as log:
            start = log.tell()
            self.process = subprocess.Popen(
                command, cwd=self.executable.parent, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, creationflags=WINDOW_FLAGS,
            )
        deadline = monotonic() + 150
        while self.is_running() and monotonic() < deadline:
            try:
                if self.request("/health", timeout=2).get("status") == "ok":
                    break
            except (URLError, TimeoutError):
                pass
            sleep(0.25)
        else:
            self.close()
            raise RuntimeError(f"Vulkan model failed to start. See {self.log_path}")

        with self.log_path.open("rb") as log:
            log.seek(start)
            offload = OFFLOAD_LINE.search(log.read().decode("utf-8", errors="replace"))
        if not offload or offload[1] == "0" or offload[1] != offload[2]:
            self.close()
            raise RuntimeError(f"llama.cpp did not confirm full Vulkan layer offload. See {self.log_path}")

    def complete(self, input_ids, max_tokens):
        if len(input_ids) + max_tokens > CONTEXT_TOKENS:
            raise ValueError("The prompt and generation cap exceed the Vulkan context window")
        try:
            response = self.request("/completion", timeout=120, body={
                "prompt": input_ids, "n_predict": max_tokens, "temperature": 0.0,
                "samplers": ["temperature"], "cache_prompt": False, "return_tokens": True,
            })
        except (URLError, TimeoutError):
            self.close()
            raise
        if response["truncated"] or response["stop_type"] not in {"eos", "limit"}:
            raise RuntimeError(f"Unexpected Vulkan termination {response['stop_type']}. See {self.log_path}")
        return response
