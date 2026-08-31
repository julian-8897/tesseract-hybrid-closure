"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp

from .checkpointing import load_training_checkpoint
from .configs import DNSConfig, SolverConfig, TrainingConfig
from .constants import VALIDATION_SEED_RANGE
from .engine import assert_smoke_passes, run_gradient_smoke
from .final_eval import (
    FINAL_EVALUATION_HORIZONS,
    MATCHED_APRIORI_UPDATES,
    SELECTION_HORIZON,
    run_evaluation_stage,
    run_model_selection,
    train_matched_apriori_baseline,
    validate_selection_report,
)
from .reporting import write_smoke_report
from .served_training import (
    SERVED_UNROLL,
    preflight_served_training_outputs,
    run_served_training,
)
from .spectral_diagnostics import DIAGNOSTIC_SPLIT, run_spectral_diagnostic
from .tesseract_demo import (
    DEMO_MAX_UPDATES,
    run_optimiser_demo,
    write_optimiser_demo_report,
)
from .training import evaluate_rollout_mse, train_aposteriori_curriculum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid LES closure tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser(
        "gradient-smoke",
        help=(
            "differentiate a single hybrid step and check the gradient reaches "
            "every closure parameter"
        ),
    )
    smoke.add_argument("--output", help="optional JSON report path")

    train = subparsers.add_parser(
        "train", help="run the a-posteriori unroll curriculum"
    )
    train.add_argument(
        "--updates-per-stage",
        nargs=3,
        type=int,
        required=True,
        metavar=("N1", "N5", "N30"),
    )
    train.add_argument("--output-dir", required=True)
    train.add_argument("--dt", type=float)
    train.add_argument("--vorticity-amplitude", type=float, default=1.0)

    evaluate = subparsers.add_parser(
        "evaluate", help="evaluate a trained checkpoint or an untrained baseline"
    )
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument("--seeds", nargs="+", type=int, required=True)
    evaluate.add_argument("--unroll", type=int, required=True)
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--dt", type=float)
    evaluate.add_argument("--vorticity-amplitude", type=float)
    evaluate.add_argument(
        "--baseline",
        choices=("none", "static-smagorinsky", "dynamic-smagorinsky"),
        default="none",
    )

    demo = subparsers.add_parser(
        "demo",
        help="run one accepted Adam update through the two-Tesseract two-step rollout",
    )
    demo.add_argument("--output", help="optional JSON report path")
    demo.add_argument(
        "--images",
        action="store_true",
        help="use containerised Tesseract clients instead of in-process ones",
    )
    demo.add_argument(
        "--max-updates",
        type=int,
        default=DEMO_MAX_UPDATES,
        help="bounded Adam update budget (default: 1)",
    )

    final = subparsers.add_parser(
        "final", help="run the sealed final-submission evaluation protocol"
    )
    final.add_argument(
        "--stage",
        choices=("select", "apriori", "evaluate"),
        nargs="+",
        default=["select", "apriori", "evaluate"],
        help="protocol stages to run (default: all three in order)",
    )
    final.add_argument("--selection-output", help="selection report JSON path")
    final.add_argument(
        "--candidate",
        action="append",
        metavar="NAME:PATH",
        help="a-posteriori candidate (repeatable)",
    )
    final.add_argument(
        "--apriori-output-dir", help="matched a-priori baseline output directory"
    )
    final.add_argument(
        "--apriori-checkpoint", help="existing a-priori checkpoint (evaluate only)"
    )
    final.add_argument("--evaluation-output", help="evaluation report JSON path")
    final.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="evaluation split (default: test)",
    )

    spectra = subparsers.add_parser(
        "spectra",
        help=(
            "seed-averaged energy and enstrophy spectra on the validation split "
            "(diagnostic only; changes no reported number)"
        ),
    )
    spectra.add_argument("--checkpoint", required=True, help="a-posteriori checkpoint")
    spectra.add_argument(
        "--apriori-checkpoint",
        help="matched a-priori checkpoint, compared when supplied",
    )
    spectra.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(VALIDATION_SEED_RANGE),
        help="validation seeds (default: the whole validation split)",
    )
    spectra.add_argument(
        "--report-steps",
        nargs="+",
        type=int,
        default=[30, 120, 500],
        help="rollout steps at which to record spectra (default: 30 120 500)",
    )
    spectra.add_argument(
        "--include-smagorinsky",
        action="store_true",
        help="also roll the static and dynamic Smagorinsky baselines",
    )
    spectra.add_argument("--output", help="optional JSON report path")

    served = subparsers.add_parser(
        "train-served",
        help=(
            "train the closure with every gradient crossing the served "
            "Tesseract boundary (demonstration, not the submitted model)"
        ),
    )
    served.add_argument(
        "--updates",
        type=int,
        required=True,
        help="Adam updates, one train-split trajectory each",
    )
    served.add_argument(
        "--unroll-steps",
        type=int,
        default=SERVED_UNROLL,
        help=f"rollout steps per update (default: {SERVED_UNROLL})",
    )
    served.add_argument(
        "--local",
        action="store_true",
        help="use in-process clients instead of the built images (for testing)",
    )
    served.add_argument(
        "--reference-checkpoint",
        help=(
            "in-process-trained checkpoint to score on the same held-out seeds, "
            "for a like-for-like comparison"
        ),
    )
    served.add_argument(
        "--checkpoint-output",
        help=(
            "pickle checkpoint receiving the final trained parameters; "
            "preflighted with --output before any DNS generation or training"
        ),
    )
    served.add_argument(
        "--output",
        help=(
            "JSON report path; preflighted with --checkpoint-output before "
            "any DNS generation or training, refuses to overwrite"
        ),
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "gradient-smoke":
        result = run_gradient_smoke()
        assert_smoke_passes(result)
        if args.output:
            write_smoke_report(result, args.output)
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if args.command == "train":
        config = TrainingConfig(updates_per_stage=tuple(args.updates_per_stage))
        solver_config = SolverConfig() if args.dt is None else SolverConfig(dt=args.dt)
        dns_config = DNSConfig(
            dt=solver_config.dt,
            vorticity_amplitude=args.vorticity_amplitude,
        )
        result = train_aposteriori_curriculum(
            config,
            args.output_dir,
            solver_config=solver_config,
            dns_config=dns_config,
        )
        print(
            json.dumps(
                {"stages": [stage.__dict__ for stage in result.stages]}, indent=2
            )
        )
        return 0
    if args.command == "evaluate":
        if args.checkpoint and args.baseline != "none":
            raise ValueError(
                "checkpoint and baseline evaluation are mutually exclusive"
            )
        params = None
        checkpoint = None
        if args.checkpoint:
            checkpoint = load_training_checkpoint(args.checkpoint)
            params = jnp.asarray(checkpoint["params_flat"], dtype=jnp.float32)
        checkpoint_solver = checkpoint.get("solver_config") if checkpoint else None
        checkpoint_dns = checkpoint.get("dns_config") if checkpoint else None
        solver_config = (
            SolverConfig(**checkpoint_solver)
            if checkpoint_solver
            else SolverConfig(dt=args.dt)
            if args.dt is not None
            else SolverConfig()
        )
        dns_config = (
            DNSConfig(**checkpoint_dns)
            if checkpoint_dns
            else DNSConfig(
                dt=solver_config.dt,
                vorticity_amplitude=args.vorticity_amplitude or 1.0,
            )
        )
        if checkpoint and args.dt is not None and args.dt != solver_config.dt:
            raise ValueError("--dt does not match the checkpoint configuration")
        if (
            checkpoint
            and args.vorticity_amplitude is not None
            and args.vorticity_amplitude != dns_config.vorticity_amplitude
        ):
            raise ValueError(
                "--vorticity-amplitude does not match the checkpoint configuration"
            )
        metrics = evaluate_rollout_mse(
            params,
            split=args.split,
            seeds=tuple(args.seeds),
            unroll=args.unroll,
            baseline=args.baseline,
            solver_config=solver_config,
            dns_config=dns_config,
        )
        print(json.dumps(metrics, indent=2))
        return 0
    if args.command == "demo":
        result = run_optimiser_demo(
            use_images=args.images,
            max_updates=args.max_updates,
        )
        if args.output:
            write_optimiser_demo_report(result, args.output)
        print(json.dumps(result.to_dict(), indent=2))
        return 0
    if args.command == "spectra":
        return _run_spectra_command(args)
    if args.command == "train-served":
        return _run_served_training_command(args)
    if args.command == "final":
        return _run_final_protocol(args)
    raise AssertionError(f"Unhandled command: {args.command}")


def _run_served_training_command(args: argparse.Namespace) -> int:
    """Train through the served components and report the boundary evidence."""
    report_path = Path(args.output) if args.output else None
    checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else None
    preflight_served_training_outputs(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )
    result = run_served_training(
        updates=args.updates,
        use_images=not args.local,
        unroll_steps=args.unroll_steps,
        reference_checkpoint=args.reference_checkpoint,
        report_path=report_path,
        checkpoint_path=checkpoint_path,
    )
    summary = result.to_dict()
    # The full loss curve belongs in the report, not the terminal.
    summary.pop("loss_curve", None)
    summary.pop("training_seeds", None)
    print(json.dumps(summary, indent=2))
    return 0


def _run_spectra_command(args: argparse.Namespace) -> int:
    """Run the validation-split spectral diagnostic and report its distances."""
    report = run_spectral_diagnostic(
        aposteriori_checkpoint=args.checkpoint,
        apriori_checkpoint=args.apriori_checkpoint,
        seeds=args.seeds,
        report_steps=args.report_steps,
        split=DIAGNOSTIC_SPLIT,
        include_smagorinsky=args.include_smagorinsky,
    )
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n")
    # The full spectra are large; print the scalar summary that reads usefully.
    print(
        json.dumps(
            {
                "split": report["split"],
                "seeds": len(report["seeds"]),
                "report_steps": report["report_steps"],
                "spectral_distance": report["spectral_distance"],
            },
            indent=2,
        )
    )
    return 0


def _candidates_from_args(pairs: Sequence[str]) -> dict[str, str]:
    candidates: dict[str, str] = {}
    for pair in pairs:
        name, separator, path = pair.partition(":")
        if not separator or not name.strip() or not path.strip():
            raise ValueError(f"candidate must be NAME:PATH, got {pair!r}")
        if name in candidates:
            raise ValueError(f"duplicate candidate name: {name!r}")
        candidates[name] = path
    return candidates


def _require_final_argument(args: argparse.Namespace, name: str, stage: str) -> str:
    value = getattr(args, name)
    if value is None:
        raise ValueError(f"final stage {stage!r} requires --{name.replace('_', '-')}")
    return value


def _run_final_protocol(args: argparse.Namespace) -> int:
    """Run the selected protocol stages in lock-step, each refused if done."""
    stages = list(args.stage)
    results: dict[str, object] = {}
    selected_path: str | None = None
    selection_output = args.selection_output

    if "select" in stages:
        selection_output = _require_final_argument(args, "selection_output", "select")
        candidates = _candidates_from_args(args.candidate or ())
        if len(candidates) < 2:
            raise ValueError(
                "final select requires at least two --candidate NAME:PATH pairs"
            )
        selection = run_model_selection(
            candidates,
            output_path=selection_output,
            horizon=SELECTION_HORIZON,
        )
        selected_path = selection["candidates"][selection["selected"]["name"]][
            "checkpoint"
        ]
        results["selection"] = {
            "won": selection["selected"]["name"],
            "criterion": selection["criterion"],
            "mean_vorticity_mse": selection["selected"]["mean_vorticity_mse"],
        }
    elif selection_output is not None and Path(selection_output).is_file():
        with Path(selection_output).open() as handle:
            selection = json.load(handle)
        validate_selection_report(selection)
        selected_path = selection["candidates"][selection["selected"]["name"]][
            "checkpoint"
        ]
    else:
        raise ValueError(
            "final needs a selection report: run --stage select or point "
            "--selection-output at an existing report"
        )

    if "apriori" in stages:
        apriori_output_dir = _require_final_argument(
            args, "apriori_output_dir", "apriori"
        )
        if selected_path is None:
            raise ValueError(
                "final apriori requires a validated selection report to derive "
                "the matched configuration from"
            )
        results["apriori"] = train_matched_apriori_baseline(
            MATCHED_APRIORI_UPDATES,
            apriori_output_dir,
            reference_checkpoint=selected_path,
        )

    if "evaluate" in stages:
        evaluation_output = _require_final_argument(
            args, "evaluation_output", "evaluate"
        )
        if selection_output is None or not Path(selection_output).is_file():
            raise FileNotFoundError(f"selection report not found: {selection_output}")
        if args.apriori_checkpoint is not None:
            apriori_checkpoint = args.apriori_checkpoint
        elif "apriori" in stages:
            apriori_checkpoint = results["apriori"]["checkpoint"]
        else:
            apriori_checkpoint = _require_final_argument(
                args, "apriori_checkpoint", "evaluate"
            )
        report = run_evaluation_stage(
            selection_report_path=selection_output,
            apriori_checkpoint_path=apriori_checkpoint,
            output_path=evaluation_output,
            split=args.split,
            horizons=FINAL_EVALUATION_HORIZONS,
        )
        results["evaluation"] = {
            key: value for key, value in report.items() if key != "per_seed_errors"
        }

    print(json.dumps(results, indent=2))
    return 0
