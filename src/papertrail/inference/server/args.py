"""Launch the C++ `llama-server` (local binary or Docker) for any GGUF model.

Why this wraps the C++ server and not `llama_cpp.Llama`: the verified Nemotron
vision path uses the `ghcr.io/ggml-org/llama.cpp:server-cuda` server, whose flags
(`--mmproj`, `--jinja`, `--chat-template-kwargs`, `--fit`, ...) do not exist on the
Python `Llama.__init__` signature. So the real source of truth for "every possible
arg" is `llama-server --help`, which this module parses and propagates into an
argparse CLI. Every server flag becomes a `--long-flag` here, defaults are set for
Nemotron, and the result is launched locally (`nohup`-style detach) or via
`docker run -d`.

Help text is fetched from (in order): a forced `--refresh-help`, the local
`llama-server` binary, `docker run --rm <image> --help`, or the bundled snapshot
`_llama_server_help.txt`. The snapshot makes the CLI usable offline; refresh it on
your machine to match your exact build:

    uv run python -m papertrail.inference.server.args --refresh-help --print

Examples:
    # Print the docker command (default backend), Nemotron defaults:
    uv run python -m papertrail.inference.server.args --print
    # Launch detached in Docker:
    uv run python -m papertrail.inference.server.args --run --detach
    # Local binary, thinking off, override a parsed flag:
    uv run python -m papertrail.inference.server.args --backend local --run --detach \\
        --chat-template-kwargs '{"enable_thinking": false}' --ctx-size 16384
    # Emit LLAMA_ARG_* env for your compose file:
    uv run python -m papertrail.inference.server.args --print-env
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# --- Nemotron defaults --------------------------------------------------------

DEFAULT_IMAGE = os.environ.get("PAPERTRAIL_LLAMA_IMAGE", "ghcr.io/ggml-org/llama.cpp:server-cuda")
DEFAULT_BINARY = os.environ.get("PAPERTRAIL_LLAMA_BINARY", "llama-server")
# Model location. The convention matches `hf download <repo> --local-dir <root>/<repo>`:
#   <models-root>/<model-repo>/<files>
# Point PAPERTRAIL_MODELS_ROOT at your HF local-dir and everything resolves to your
# already-downloaded files (filenames are auto-discovered, not hardcoded).
DEFAULT_MODELS_ROOT = os.environ.get("PAPERTRAIL_MODELS_ROOT", "./models")
DEFAULT_MODEL_REPO = os.environ.get(
    "PAPERTRAIL_MODEL_REPO", "NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF"
)
DEFAULT_QUANT = os.environ.get("PAPERTRAIL_QUANT", "UD-Q4_K_XL")
FALLBACK_MMPROJ = "mmproj-BF16.gguf"

# Defaults applied to parsed flags by their argparse dest (only if the flag exists
# in the parsed help — never invent a flag this build does not support).
NEMOTRON_DEFAULTS: dict[str, object] = {
    # NOTE: deliberately NO --n-gpu-layers default. Pinning it disables `--fit on`'s
    # ability to back off ("n_gpu_layers already set by user, abort"). Leaving it
    # unset lets fit offload as many layers as fit in free VRAM and park the rest on
    # CPU — exactly how Unsloth Studio loads this model. Pass --n-gpu-layers 99
    # yourself once you have ~23 GiB of VRAM free (headless / apps closed).
    "ctx_size": "16384",
    "flash_attn": "on",
    "jinja": True,
    "no_context_shift": True,
    "host": "0.0.0.0",
    "port": "8080",
    "alias": "nemotron",
    "parallel": "1",
    "fit": "on",
}

# Info-only flags we never want as serving options (and `--help` collides with
# argparse's own).
_SKIP = {"--help", "--usage", "--version", "--license", "--completion-bash", "--cache-list"}

_HELP_CACHE = Path(__file__).with_name("_llama_server_help.txt")
_ENV_RE = re.compile(r"\(env:\s*(LLAMA_ARG_\w+)\)")
_DIVIDER_RE = re.compile(r"^\s*-{3,}")
_OPT_RE = re.compile(
    r"^(?P<indent>\s*)(?P<flags>--?[A-Za-z0-9][\w-]*(?:,\s*--?[\w-]+)*)(?P<rest>.*)$"
)


# --- flag model ---------------------------------------------------------------


@dataclass
class Flag:
    dest: str
    options: list[str]  # long option strings used on the CLI
    is_bool: bool
    metavar: str = ""
    help: str = ""
    env: str | None = None
    _lines: list[str] = field(default_factory=list)


def _is_metavar(chunk: str) -> bool:
    """A token after the flags is a metavar (=> the flag takes a value).

    llama.cpp descriptions start lowercase ("size of...", "set ..."); metavars
    are uppercase (N, FNAME, HOST) or bracketed enums ([on|off]).
    """
    if not chunk:
        return False
    return chunk[0] == "[" or chunk[0].isupper()


def parse_help(text: str) -> list[Flag]:
    """Parse `llama-server --help` into Flag specs (one per long option for bools)."""
    flags: list[Flag] = []
    cur: Flag | None = None
    seen: set[str] = set()

    def emit(longs: list[str], is_bool: bool, metavar: str, desc: str, line: str) -> Flag | None:
        primary = longs[0]
        dest = primary.lstrip("-").replace("-", "_")
        if dest in seen:
            return None
        seen.add(dest)
        f = Flag(dest=dest, options=longs, is_bool=is_bool, metavar=metavar, help=desc)
        m = _ENV_RE.search(line)
        if m:
            f.env = m.group(1)
        f._lines.append(line)
        flags.append(f)
        return f

    for line in text.splitlines():
        if _DIVIDER_RE.match(line):
            cur = None
            continue
        m = _OPT_RE.match(line)
        if m and len(m.group("indent")) <= 7:
            tokens = [t.strip() for t in m.group("flags").split(",")]
            longs = [t for t in tokens if t.startswith("--") and t not in _SKIP]
            if not longs:  # short-only or skipped (e.g. --help): ignore for serving
                cur = None
                continue
            rest = m.group("rest").strip()
            metavar, desc = "", ""
            if rest:
                parts = re.split(r"\s{2,}", rest, maxsplit=1)
                if _is_metavar(parts[0]):
                    metavar = parts[0]
                    desc = parts[1] if len(parts) > 1 else ""
                else:
                    desc = rest
            is_bool = not metavar
            if is_bool and len(longs) > 1:
                # Paired toggles like `--jinja, --no-jinja`: expose each separately.
                cur = None
                for lg in longs:
                    emit([lg], True, "", desc, line)
            else:
                cur = emit(longs, is_bool, metavar, desc, line)
        elif cur is not None:
            cur._lines.append(line)
            if cur.env is None:
                em = _ENV_RE.search(line)
                if em:
                    cur.env = em.group(1)
            extra = line.strip()
            if extra and not extra.startswith("(env:"):
                cur.help = (cur.help + " " + extra).strip()
    return flags


# --- help acquisition ---------------------------------------------------------


def fetch_help(image: str, binary: str, refresh: bool) -> str:
    """Return `llama-server --help`, caching to the bundled snapshot."""
    if not refresh and _HELP_CACHE.exists():
        return _HELP_CACHE.read_text(encoding="utf-8")

    text = ""
    from shutil import which

    if which(binary):
        text = subprocess.run([binary, "--help"], capture_output=True, text=True).stdout
    if not text and which("docker"):
        text = subprocess.run(
            ["docker", "run", "--rm", image, "--help"], capture_output=True, text=True
        ).stdout
    if text:
        _HELP_CACHE.write_text(text, encoding="utf-8")
        return text
    if _HELP_CACHE.exists():
        return _HELP_CACHE.read_text(encoding="utf-8")
    raise RuntimeError("could not obtain llama-server --help: no local binary, no docker, no cache")


# --- parser assembly ----------------------------------------------------------


def build_parser(flags: list[Flag]) -> tuple[argparse.ArgumentParser, set[str]]:
    p = argparse.ArgumentParser(
        prog="papertrail-llama",
        description="Launch llama-server (Nemotron defaults) locally or via Docker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    w = p.add_argument_group("launcher (not passed to llama-server)")
    w.add_argument("--backend", choices=["docker", "local"], default="docker")
    w.add_argument("--image", default=DEFAULT_IMAGE, help="docker image for the server")
    w.add_argument("--binary", default=DEFAULT_BINARY, help="local llama-server binary")
    w.add_argument(
        "--models-root",
        default=DEFAULT_MODELS_ROOT,
        help="host HF local-dir (mounted to /models in docker); env PAPERTRAIL_MODELS_ROOT",
    )
    w.add_argument("--model-repo", default=DEFAULT_MODEL_REPO, help="repo subdir under models-root")
    w.add_argument(
        "--quant", default=DEFAULT_QUANT, help="preferred quant tag when auto-discovering"
    )
    w.add_argument("--container-name", default="papertrail-llama")
    w.add_argument("--docker-gpus", default="all", help="value for docker --gpus")
    w.add_argument("--log-dir", default="./logs", help="where detached local logs go")
    w.add_argument("--detach", action="store_true", help="run in background / docker -d")
    w.add_argument("--run", action="store_true", help="actually launch (default: just print)")
    w.add_argument("--print-env", action="store_true", help="print LLAMA_ARG_* env and exit")
    w.add_argument(
        "--print", action="store_true", help="print the command (default when --run absent)"
    )
    w.add_argument("--refresh-help", action="store_true", help="re-read llama-server --help")

    g = p.add_argument_group("llama-server flags (parsed from --help)")
    valid: set[str] = set()
    for f in flags:
        kwargs: dict = {"dest": f.dest, "default": None, "help": f.help}
        if f.is_bool:
            kwargs["action"] = "store_true"
        else:
            kwargs["metavar"] = f.metavar or f.dest.upper()
        try:
            g.add_argument(*f.options, **kwargs)
            valid.add(f.dest)
        except argparse.ArgumentError:
            continue  # conflicts with a launcher option; reach it via passthrough
    return p, valid


# --- translation: namespace -> server flags / env -----------------------------


def server_flags(ns: argparse.Namespace, flags: list[Flag]) -> list[str]:
    out: list[str] = []
    for f in flags:
        val = getattr(ns, f.dest, None)
        if val is None or val is False:
            continue
        if f.is_bool:
            out.append(f.options[0])
        else:
            out += [f.options[0], str(val)]
    return out


def to_env(ns: argparse.Namespace, flags: list[Flag]) -> dict[str, str]:
    env: dict[str, str] = {}
    for f in flags:
        val = getattr(ns, f.dest, None)
        if val is None or val is False or not f.env:
            continue
        env[f.env] = "1" if val is True else str(val)
    return env


def discover_model_files(repo_dir: Path, quant: str) -> tuple[str | None, str | None]:
    """Find (model, mmproj) gguf filenames in repo_dir; (None, None) if absent."""
    if not repo_dir.is_dir():
        return None, None
    ggufs = sorted(p.name for p in repo_dir.glob("*.gguf"))
    mmprojs = [n for n in ggufs if "mmproj" in n.lower()]
    models = [n for n in ggufs if "mmproj" not in n.lower()]
    model = None
    if models:
        preferred = [n for n in models if quant and quant.lower() in n.lower()]
        model = (preferred or models)[0]
    return model, (mmprojs[0] if mmprojs else None)


def resolve_model_paths(ns: argparse.Namespace, valid: set[str]) -> list[str]:
    """Fill ns.model / ns.mmproj from <models-root>/<model-repo>, auto-discovering
    filenames. Returns the HOST paths that must exist before a real launch."""
    repo_dir = Path(ns.models_root) / ns.model_repo
    disc_model, disc_mmproj = discover_model_files(repo_dir, ns.quant)
    must_exist: list[str] = []

    def place(filename: str) -> str:
        if ns.backend == "docker":
            return f"/models/{ns.model_repo}/{filename}"
        return str(repo_dir / filename)

    if "model" in valid and getattr(ns, "model", None) is None:
        fn = disc_model or f"{ns.model_repo.removesuffix('-GGUF')}-{ns.quant}.gguf"
        ns.model = place(fn)
        must_exist.append(str(repo_dir / fn))
    if "mmproj" in valid and getattr(ns, "mmproj", None) is None:
        fn = disc_mmproj or FALLBACK_MMPROJ
        ns.mmproj = place(fn)
        must_exist.append(str(repo_dir / fn))
    return must_exist


def build_argv(ns: argparse.Namespace, sflags: list[str]) -> list[str]:
    port = str(getattr(ns, "port", None) or "8080")
    if ns.backend == "local":
        return [ns.binary, *sflags]
    mount = str(Path(ns.models_root).resolve())
    run_mode = ["-d", "--restart", "unless-stopped"] if ns.detach else ["--rm"]
    return [
        "docker",
        "run",
        *run_mode,
        "--name",
        ns.container_name,
        "--gpus",
        ns.docker_gpus,
        "-p",
        f"{port}:{port}",
        "-v",
        f"{mount}:/models",
        ns.image,
        *sflags,
    ]
