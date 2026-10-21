"""Single entry point for raw datasets -> DPD -> Cadence PWL and feedback."""
import argparse
from pathlib import Path
from unified_dpd.pipeline import run_pipeline


def main(argv=None):
    p = argparse.ArgumentParser(description="Unified I/Q pipeline: training BIC, validation comparison, and PWL export")
    p.add_argument("--config", default=str(Path(__file__).with_name("pipeline.json")))
    p.add_argument("--stage", choices=("all", "preprocess", "compare"), default="all")
    p.add_argument("--models-dir", help="Reuse a saved run or model_results folder (legacy models folders accepted); no coefficient fitting")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--feedback-dir", help="Directory of five validation-named CSV/input files exported after Cadence")
    p.add_argument("--reference-run", help="Completed unified run whose exported desired signal is the feedback reference")
    p.add_argument("--feedback-model", help="Export folder: MP/EMP/GMP/PrunedVolterra/LUT/no_dpd (legacy sin_dpd alias accepted)")
    a = p.parse_args(argv)
    if a.feedback_dir:
        if not a.reference_run or not a.feedback_model:
            p.error("Feedback requires --reference-run and --feedback-model")
        from unified_dpd.feedback import run_feedback
        return run_feedback(a.config,a.reference_run,a.feedback_model,a.feedback_dir,not a.no_plots)
    if a.reference_run or a.feedback_model:
        p.error("Use --feedback-dir with feedback options")
    return run_pipeline(a.config,a.stage,a.models_dir,not a.no_plots)


if __name__ == "__main__":
    print(main())
