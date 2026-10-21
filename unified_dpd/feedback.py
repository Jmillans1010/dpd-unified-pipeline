"""Feedback is compared to the exported DESIRED signal, never to F(x)."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import numpy as np
from .pipeline import load_config, digest, CHANNELS, write_csv
from .plots import pyplot
from .layout import create_run_directories, resolve_saved_export
from iq_processor.models import ProcessingConfig, IQSignal, InputData
from iq_processor.readers import CSVReader, InputSCSReader
from iq_processor.processing import SignalValidator, SignalPreprocessor, SampleAligner
from dpd.serialization import JSONWriter
from cadence_bridge.feedback import analyze_feedback


def run_feedback(config_path, reference_run, model, feedback_dir, make_plots=True):
    path, config = load_config(config_path)
    model = "no_dpd" if model == "sin_dpd" else model
    if model not in (*config["models"], "no_dpd"):
        raise ValueError("Unknown feedback model")
    reference_run, feedback_dir = Path(reference_run).resolve(), Path(feedback_dir).resolve()
    manifest = JSONWriter.read(reference_run/"manifest.json")
    export = resolve_saved_export(reference_run, model)
    export_meta = JSONWriter.read(export/"metadata.json")
    desired = np.genfromtxt(export/"samples.csv", delimiter=",", names=True)
    out = (path.parent/config["output_dir"])/datetime.now(timezone.utc).strftime("feedback_%Y%m%dT%H%M%S_%fZ")
    out.mkdir(parents=True, exist_ok=False)
    layout = create_run_directories(out)
    state = dict(status="running", reference_run=str(reference_run), model=model,
                 schema_version="2.1", artifact_language="en", directories=layout,
                 reference_manifest_sha256=digest(reference_run/"manifest.json"), config=config)
    JSONWriter.write(state,out/"manifest.json")
    try:
        sources = out/"diagnostics"/"sources"; sources.mkdir()
        for f in [*[f"{ch}_validation.csv" for ch in CHANNELS],"input"]:
            shutil.copyfile(feedback_dir/f,sources/f)
        pc = ProcessingConfig(input_files={ch:f"{ch}_validation.csv" for ch in CHANNELS},netlist="input",**config["preprocessing"])
        pc.validate()
        netlist = InputSCSReader().read(sources/"input")
        measured = CSVReader().read(pc.input_files,sources,pc)
        reference = IQSignal(desired["time_s"], desired["I_desired_V"]+1j*desired["Q_desired_V"], np.arange(len(desired)))
        pair = InputData(reference,measured.output,measured.sources)
        sampling = SignalValidator().validate(pair,pc)
        prepared, preprocessing = SignalPreprocessor().prepare(pair,pc)
        alignment = SampleAligner().align(prepared,sampling,pc)
        idx = np.flatnonzero(desired["is_symbol"][alignment.input_indices] == 1)
        if not len(idx):
            raise ValueError("No reference symbols after feedback alignment")
        result = analyze_feedback(alignment.x,alignment.y,manifest["gain"],idx)
        result.update(status="measured_feedback", gain=manifest["gain"], gain_policy="fixed original training gain",
            alignment=alignment.diagnostics,preprocessing=preprocessing, symbols=len(idx),
            source_files={f.name:digest(f) for f in sources.iterdir()},reference_samples_sha256=digest(export/"samples.csv"),
            export_voltage_scale=export_meta["scale_applied_before_limit"],netlist=netlist.simulation_config,
            note="Compare DPD and no_dpd feedback from matching simulation conditions; no automatic retraining")
        JSONWriter.write(result,out/"model_comparisons"/"metrics.json")
        write_csv(out/"diagnostics"/"aligned_feedback.csv",[alignment.time,alignment.x.real,alignment.x.imag,alignment.y.real,alignment.y.imag],
                  ["time_s","I_desired_V","Q_desired_V","I_measured_V","Q_measured_V"])
        if make_plots:
            plt=pyplot(); fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
            z=alignment.y/manifest["gain"]
            axes[0].scatter(alignment.x[idx].real,alignment.x[idx].imag,s=10,label="Desired")
            axes[0].scatter(z[idx].real,z[idx].imag,s=10,label="Output / G")
            axes[0].set(xlabel="I (V)",ylabel="Q (V)",title=f"Symbols only · EVM {result['evm_percent']:.3f}%")
            axes[0].set_aspect("equal"); axes[0].legend()
            axes[1].scatter(result["am_am_input_V"],result["am_am_output_V"],s=2)
            axes[1].set(xlabel="Desired input (V)",ylabel="Output (V)",title="AM/AM")
            axes[2].scatter(result["am_am_input_V"],result["am_pm_deg"],s=2)
            axes[2].set(xlabel="Desired input (V)",ylabel="Residual phase (degrees)",title="AM/PM relative to G")
            fig.savefig(out/"model_comparisons"/"feedback.png",dpi=150); plt.close(fig)
        state["status"]="completed"
    except Exception as exc:
        state.update(status="failed",error=str(exc))
        raise
    finally:
        JSONWriter.write(state,out/"manifest.json")
    return out
