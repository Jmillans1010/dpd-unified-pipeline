"""CLI independiente; conserva run.py para el preprocesador original."""
import argparse
from pathlib import Path
import sys
from dpd.pipeline import TrainingPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Identificación inversa de PA: MP, EMP, GMP, Volterra y LUT")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("dpd_config.json"))
    args = parser.parse_args()
    try:
        directory = TrainingPipeline(args.config).run()
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error DPD: {exc}", file=sys.stderr)
        return 1
    print(f"Resultados: {directory.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
