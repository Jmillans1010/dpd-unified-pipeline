"""Prepare desired I/Q references alongside space for later simulation outputs."""
from pathlib import Path
import hashlib
import numpy as np
from dpd.serialization import JSONWriter


def prepare_results(run):
    """Extract exact exported references; refuse to overwrite existing result folders."""
    run = Path(run).resolve()
    sources = sorted((run / "outputs").glob("*/samples.csv"))
    if not sources:
        raise ValueError(f"No exported samples.csv files found in {run / 'outputs'}")
    targets = [run / "results" / source.parent.name for source in sources]
    for target in targets:
        if target.exists():
            raise FileExistsError(f"Results already exist; preserving saved references and simulation files: {target}")
    prepared = []
    for source, target in zip(sources, targets):
        samples = np.atleast_1d(np.genfromtxt(source, delimiter=",", names=True))
        time = samples["time_s"]
        reference = np.column_stack((time, samples["I_desired_V"], samples["Q_desired_V"]))
        if not len(time) or not np.isfinite(reference).all() or np.any(np.diff(time) <= 0):
            raise ValueError(f"Invalid desired reference: {source}")
        target.mkdir(parents=True)
        for channel, column in (("I", "I_desired_V"), ("Q", "Q_desired_V")):
            np.savetxt(target / f"{channel}_in.csv", np.column_stack((time, samples[column])),
                       delimiter=",", header="time_s,voltage_V", comments="", fmt="%.17g")
        export_metadata = JSONWriter.read(source.parent / "metadata.json")
        JSONWriter.write(dict(
            model=source.parent.name, status="awaiting_simulation_outputs",
            reference="Desired input before predistortion; not the predistorted Cadence input",
            source_samples=str(source.relative_to(run)),
            source_samples_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_columns=["time_s", "I_desired_V", "Q_desired_V"],
            num_samples=len(time), units={"time": "s", "amplitude": "V"},
            voltage_scale=export_metadata["scale_applied_before_limit"],
            note="Exact desired values and time grid from the matching export, including its scale and trimming; no PA gain or phase applied.",
            reference_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in target.glob("*_in.csv")}
        ), target / "reference_metadata.json")
        (target / "README.md").write_text(
            "# Simulation results\n\n"
            "I_in.csv and Q_in.csv are the original desired reference, before predistortion. "
            "Each file contains time_s and voltage_V. They match I_desired_V and Q_desired_V "
            "in the corresponding outputs folder; the configured export scale is already included.\n\n"
            "Drive Cadence with I.txt and Q.txt from outputs, not with these reference files. "
            "Save the simulated I_OUT and Q_OUT here as I_out.csv and Q_out.csv, preserving their timestamps. "
            "No simulated output files are created in advance.\n\n"
            "Compare the returned output against this desired reference after time alignment and "
            "compensation with the original complex training gain. Matching row numbers alone "
            "do not establish alignment. Use the no_dpd case as the baseline.\n\n"
            "This directory collects references and simulation results. The existing --feedback-dir "
            "command still expects the documented five-file Cadence dataset; these two reference "
            "files do not replace its measured input channels.\n", encoding="utf-8")
        prepared.append(source.parent.name)
    return prepared
