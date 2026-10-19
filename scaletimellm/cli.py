"""Command-line interface for ScaleTimeLLM.

Subcommands are added incrementally as the project is built.  Keeping a single
entry point makes every artifact reproducible from a short shell command.
"""

from __future__ import annotations

import argparse
import json
import sys

from scaletimellm import __version__
from scaletimellm.backends.registry import get_backend


def _cmd_version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _cmd_check_env(args: argparse.Namespace) -> int:
    import subprocess

    command = [sys.executable, "scripts/check_env.py", "--config", args.config]
    if args.load:
        command.append("--load")
    return subprocess.call(command)


def _cmd_demo(_: argparse.Namespace) -> int:
    backend = get_backend("mock")
    generations = backend.generate(["2+2?"], n=2, temperature=0.8, max_new_tokens=16)
    print(json.dumps([g.text for g in generations[0]], indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scaletimellm", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    version = sub.add_parser("version", help="print the package version")
    version.set_defaults(func=_cmd_version)

    check = sub.add_parser("check-env", help="report backend availability")
    check.add_argument("--config", default="configs/base.yaml")
    check.add_argument("--load", action="store_true")
    check.set_defaults(func=_cmd_check_env)

    demo = sub.add_parser("demo", help="run a seeded mock generation")
    demo.set_defaults(func=_cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
