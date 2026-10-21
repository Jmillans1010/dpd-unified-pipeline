"""Entrada CLI. Las rutas de config.json se resuelven desde su propia carpeta."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
# Las extensiones binarias se cargan del intérprete seleccionado, nunca de
# .runtime_deps: esa carpeta puede contener wheels de otra versión de Python.
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT/".mplconfig"))

from iq_processor.models import ProcessingConfig, ProcessingError
from iq_processor.readers import CSVReader, InputSCSReader, MetadataResolver, fingerprint
from iq_processor.processing import SignalValidator, DataProcessor
from iq_processor.outputs import CSVWriter, JSONWriter, Grapher, write_summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Alineación compleja y EVM del trayecto I/Q de un PA.")
    parser.add_argument("--config", default="config.json", help="Ruta relativa al proyecto (o absoluta) de la configuración.")
    parser.add_argument("--no-plots", action="store_true", help="Exportar CSV/JSON sin generar imágenes.")
    args = parser.parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT/config_path
    config_path = config_path.resolve()
    run_dir = None
    try:
        config = ProcessingConfig.read(config_path)
        if args.no_plots:
            config.plots = False
        root = config_path.parent
        raw = CSVReader().read(config.input_files, root, config)
        sampling = SignalValidator().validate(raw, config)
        netlist = root/config.netlist if config.netlist else None
        metadata = InputSCSReader().read(netlist)
        if netlist is not None and netlist.is_file():
            raw.sources.append(fingerprint(netlist, root, "netlist"))
        sps = MetadataResolver().resolve(metadata, config, root, raw.sources, sampling)
        result = DataProcessor().process(raw, metadata, config, sampling, sps)
        output_root = root/config.output_directory
        stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
        run_dir = output_root/stamp
        run_dir.mkdir(parents=True, exist_ok=False)
        paths = CSVWriter().write(result, run_dir)
        if config.plots:
            paths += Grapher().plot(result, run_dir)
        paths += ["metadata.json", "summary.txt"]
        result.metadata["processing"]["output_files"] = paths
        result.metadata["processing"]["configuration_file"] = config_path.name
        result.metadata["processing"]["stage_status"]["export"] = {"status": "completed"}
        JSONWriter().write(result.metadata, run_dir/"metadata.json")
        write_summary(result, run_dir/"summary.txt")
        JSONWriter().write({"directory": stamp, "metadata": f"{stamp}/metadata.json",
                            "summary": f"{stamp}/summary.txt"}, output_root/"latest_run.json")
        print((run_dir/"summary.txt").read_text(encoding="utf-8"))
        print("Resultados:", Path(os.path.relpath(run_dir, PROJECT_ROOT)).as_posix())
        return 0
    except (ProcessingError, OSError, ValueError, ImportError) as exc:
        if run_dir is not None:
            (run_dir/"FAILED.txt").write_text(str(exc), encoding="utf-8")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
