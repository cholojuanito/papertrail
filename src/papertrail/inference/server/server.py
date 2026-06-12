import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .args import (
    DEFAULT_BINARY,
    DEFAULT_IMAGE,
    NEMOTRON_DEFAULTS,
    build_argv,
    build_parser,
    fetch_help,
    parse_help,
    resolve_model_paths,
    server_flags,
    to_env,
)


def _launch(ns: argparse.Namespace, argv: list[str]) -> int:
    if ns.backend == "local" and ns.detach:
        log_dir = Path(ns.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log = (log_dir / "llama-server.log").open("ab")
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (log_dir / "llama-server.pid").write_text(str(proc.pid))
        print(f"started pid={proc.pid}, logging to {log_dir / 'llama-server.log'}")
        return 0
    return subprocess.run(argv).returncode


# --- entrypoint ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    # Pre-parse only the options that affect help acquisition.
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--refresh-help", action="store_true")
    pre.add_argument("--image", default=DEFAULT_IMAGE)
    pre.add_argument("--binary", default=DEFAULT_BINARY)
    pre_ns, _ = pre.parse_known_args(argv)

    flags = parse_help(fetch_help(pre_ns.image, pre_ns.binary, pre_ns.refresh_help))
    parser, valid = build_parser(flags)
    for dest, value in NEMOTRON_DEFAULTS.items():
        if dest in valid:
            parser.set_defaults(**{dest: value})
    ns = parser.parse_args(argv)

    # Explicit env overrides win; otherwise resolve from <models-root>/<model-repo>.
    if os.environ.get("PAPERTRAIL_MODEL") and "model" in valid and ns.model is None:
        ns.model = os.environ["PAPERTRAIL_MODEL"]
    if os.environ.get("PAPERTRAIL_MMPROJ") and "mmproj" in valid and ns.mmproj is None:
        ns.mmproj = os.environ["PAPERTRAIL_MMPROJ"]
    must_exist = resolve_model_paths(ns, valid)

    if ns.print_env:
        for k, v in to_env(ns, flags).items():
            print(f"{k}={v}")
        return 0

    sflags = server_flags(ns, flags)
    argv_out = build_argv(ns, sflags)

    if ns.run:
        missing = [p for p in must_exist if not Path(p).exists()]
        if missing:
            print("error: model file(s) not found on host:", file=sys.stderr)
            for p in missing:
                print(f"  {p}", file=sys.stderr)
            print(
                "\nFix by either setting PAPERTRAIL_MODELS_ROOT to your HF local-dir, or:\n"
                f"  hf download unsloth/{ns.model_repo} \\\n"
                f"    --local-dir {Path(ns.models_root) / ns.model_repo} \\\n"
                f'    --include "*{ns.quant}*" --include "*mmproj*"',
                file=sys.stderr,
            )
            return 2
        return _launch(ns, argv_out)
    print(shlex.join(argv_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
