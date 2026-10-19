"""Command-line interface for ScaleTimeLLM.

Subcommands are added incrementally as the project is built.  Keeping a single
entry point makes every artifact reproducible from a short shell command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from scaletimellm import __version__
from scaletimellm.backends.registry import get_backend
from scaletimellm.config import load_config
from scaletimellm.tasks.registry import get_task


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


def _cmd_inspect_task(args: argparse.Namespace) -> int:
    if args.offline:
        os.environ["SCALETIMELLM_OFFLINE"] = "1"
    task = get_task(args.task)
    try:
        problems = task.load(args.n, seed=args.seed)
    except Exception as error:
        print(f"task load failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    payload = []
    for problem in problems:
        payload.append(
            {
                "id": problem.id,
                "source": problem.metadata.get("source", task.name),
                "prompt": task.prompt(problem),
                "gold": problem.gold,
                "difficulty": problem.difficulty,
            }
        )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from scaletimellm.experiments.run import run_experiment

    config = load_config(args.config)
    summary = run_experiment(
        config,
        strategy_name=args.strategy,
        budget=args.budget,
        task_name=args.task,
        seed=args.seed,
        n_problems=args.n_problems,
        output_dir=args.output_dir,
    )
    return 0 if summary["records"] > 0 else 1


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

    inspect = sub.add_parser("inspect-task", help="print loaded prompts and gold answers")
    inspect.add_argument("--task", default="synthetic")
    inspect.add_argument("--n", type=int, default=3)
    inspect.add_argument("--seed", type=int, default=0)
    inspect.add_argument("--offline", action="store_true")
    inspect.set_defaults(func=_cmd_inspect_task)

    run = sub.add_parser("run", help="run one strategy/budget combination")
    run.add_argument("--config", default="configs/smoke.yaml")
    run.add_argument("--strategy", default=None)
    run.add_argument("--budget", type=int, default=None)
    run.add_argument("--task", default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--n-problems", type=int, default=None)
    run.add_argument("--output-dir", default=None)
    run.set_defaults(func=_cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
