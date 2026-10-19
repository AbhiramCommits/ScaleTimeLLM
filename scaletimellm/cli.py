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


def _cmd_train_verifier(args: argparse.Namespace) -> int:
    from scaletimellm.verifier.train import train_verifiers

    config = load_config(args.config)
    result = train_verifiers(
        config,
        n_train_problems=args.train_problems,
        n_eval_problems=args.eval_problems,
        k=args.k,
        epochs=args.epochs,
        seed=args.seed,
        artifact_dir=args.artifact_dir,
        results_path=args.results,
        train_trl=not args.no_trl,
        reuse_data=args.reuse_data,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    from scaletimellm.experiments.sweep import run_sweep

    config = load_config(args.config)
    summary = run_sweep(
        config, output_dir=args.output_dir, resume=not args.no_resume
    )
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "cell_summaries"},
            indent=2,
        )
    )
    return 0


def _cmd_bench_serve(args: argparse.Namespace) -> int:
    from scaletimellm.serving.bench import benchmark_serving

    config = load_config(args.config)
    payload = benchmark_serving(
        config,
        strategy_name=args.strategy,
        budget=args.budget,
        n_problems=args.n_problems,
        seed=args.seed,
        output_path=args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from scaletimellm.analysis.report import write_report

    summary = write_report(args.results_dir, n_boot=args.bootstrap)
    print(
        json.dumps(
            {
                "summary": "results/summary.json",
                "report": "results/REPORT.md",
                "best_per_task": summary["best_per_task"],
                "equal_flops": summary["equal_flops"],
            },
            indent=2,
        )
    )
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

    verifier = sub.add_parser("train-verifier", help="generate verifier data and train")
    verifier.add_argument("--config", default="configs/smoke.yaml")
    verifier.add_argument("--train-problems", type=int, default=120)
    verifier.add_argument("--eval-problems", type=int, default=40)
    verifier.add_argument("--k", type=int, default=4)
    verifier.add_argument("--epochs", type=int, default=35)
    verifier.add_argument("--seed", type=int, default=0)
    verifier.add_argument("--artifact-dir", default="artifacts/verifier")
    verifier.add_argument("--results", default="results/verifier_metrics.json")
    verifier.add_argument("--no-trl", action="store_true")
    verifier.add_argument("--reuse-data", action="store_true")
    verifier.set_defaults(func=_cmd_train_verifier)

    sweep = sub.add_parser("sweep", help="run the configured cross-product sweep")
    sweep.add_argument("--config", default="configs/full.yaml")
    sweep.add_argument("--output-dir", default=None)
    sweep.add_argument("--no-resume", action="store_true")
    sweep.set_defaults(func=_cmd_sweep)

    bench = sub.add_parser("bench-serve", help="measure serving latency and throughput")
    bench.add_argument("--config", default="configs/serving.yaml")
    bench.add_argument("--strategy", default=None)
    bench.add_argument("--budget", type=int, default=None)
    bench.add_argument("--n-problems", type=int, default=None)
    bench.add_argument("--seed", type=int, default=None)
    bench.add_argument("--output", default="results/serving_bench.json")
    bench.set_defaults(func=_cmd_bench_serve)

    report = sub.add_parser("report", help="regenerate analysis, tables, figures and summary")
    report.add_argument("--results-dir", default="results")
    report.add_argument("--bootstrap", type=int, default=2000)
    report.set_defaults(func=_cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
