from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import platform
import shutil
import numpy as np
from . import __version__
from .search import TrainingBICSearch
from .layout import create_run_directories, resolve_saved_models
from . import plots
from iq_processor.models import ProcessingConfig
from iq_processor.readers import CSVReader, InputSCSReader, MetadataResolver, fingerprint
from iq_processor.processing import SignalValidator, DataProcessor as IQProcessor
from iq_processor.outputs import CSVWriter, Grapher, write_summary
from dpd.contracts import DataPartition
from dpd.data import DataProcessor as DPDProcessor
from dpd.pipeline import POLICIES, ESTIMATORS
from dpd.trainers import TRAINERS
from dpd.serialization import JSONWriter
from dpd.metrics import normalized_error, error_to_db
from cadence_bridge.core import DPDModel, signal_stats, export_to_cadence, checked_signal

CHANNELS = ("I_in", "Q_in", "I_out", "Q_out")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, columns, names):
    np.savetxt(path, np.column_stack(columns), delimiter=",", header=",".join(names), comments="", fmt="%.17g")


def load_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text(encoding="utf-8-sig"))
    if c.get("source_units") != {"time": "s", "amplitude": "V"}:
        raise ValueError("Declare source_units time=s, amplitude=V; automatic unit guessing is disabled")
    if not c.get("models") or set(c["models"]) - set(TRAINERS):
        raise ValueError("Invalid model families")
    for k in ("max_iterations", "max_candidates", "beam_width", "max_basis_terms"):
        if type(c["search"][k]) is not int or c["search"][k] < 1:
            raise ValueError(f"search.{k} must be a positive integer")
    if type(c["search"]["plateau_extra_depth"]) is not int or c["search"]["plateau_extra_depth"] < 0:
        raise ValueError("plateau_extra_depth must be a nonnegative integer")
    if not np.isfinite(c["search"]["max_working_mib"]) or c["search"]["max_working_mib"] <= 0:
        raise ValueError("Invalid memory planning budget")
    e = c["export"]
    if e["overflow_policy"] not in ("zero_pair", "reject") or e["models"] not in ("all", "winner"):
        raise ValueError("Invalid export policy")
    for k in ("max_abs_voltage", "voltage_scale"):
        if not np.isfinite(e[k]) or e[k] <= 0:
            raise ValueError(f"Invalid export.{k}")
    return path, c


def preprocess(source, role, directory, config, make_plots=True):
    source = Path(source).resolve()
    files = {ch: f"{ch}_{role}.csv" for ch in CHANNELS}
    required = [*files.values(), "input"]
    if any(not (source/f).is_file() for f in required):
        raise ValueError(f"{role}: required files: {required}; directory={source}")
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = directory/"sources"
    snapshot.mkdir()
    provenance = []
    for filename in required:
        path = source/filename
        initial_hash = digest(path)
        shutil.copyfile(path, snapshot/filename)
        if digest(snapshot/filename) != initial_hash or digest(path) != initial_hash:
            raise ValueError("Source changed during snapshot")
        provenance.append(dict(name=filename, original_path=str(path), sha256=initial_hash))
    # Read the immutable per-run snapshot, including its matching input netlist.
    headers = {}
    for ch, filename in files.items():
        with (snapshot/filename).open(encoding="utf-8-sig", newline="") as f:
            header = next(csv.reader(f))
        canonical = [s.strip().upper() for s in header]
        expected = [f"/{ch.upper()} X", f"/{ch.upper()} Y"]
        if canonical != expected and canonical != ["TIME_S", "VOLTAGE_V"]:
            raise ValueError(f"Unrecognized axes for {filename}: {header}; expected {expected} or time_s,voltage_V")
        headers[ch] = header
    pc = ProcessingConfig(input_files=files, netlist="input", **config["preprocessing"])
    pc.validate()
    meta = InputSCSReader().read(snapshot/"input")
    statements = meta.simulation_config.get("raw_statements", {})
    ofdm = [v["MSource_OFDMType"] for v in statements.values() if "MSource_OFDMType" in v]
    if pc.source_data_rate_is_bit_rate and (not ofdm or any(v != "No" for v in ofdm)):
        raise ValueError("Automatic bit-rate/symbol-rate conversion requires explicit MSource_OFDMType=No")
    raw = CSVReader().read(files, snapshot, pc)
    raw.sources.append(fingerprint(snapshot/"input", snapshot, "netlist"))
    sampling = SignalValidator().validate(raw, pc)
    sps = MetadataResolver().resolve(meta, pc, snapshot, raw.sources, sampling)
    if sps is None or meta.signal_config.get("symbol_rate_hz") is None:
        raise ValueError("Symbol rate must be resolved from input or explicit metadata")
    strobe = meta.simulation_config.get("strobe_period_s")
    if strobe is not None and not np.isclose(strobe, sampling["sample_period_s"], rtol=pc.time_rtol, atol=pc.time_atol_s):
        raise ValueError("CSV sampling and input strobeperiod disagree; verify dataset provenance")
    result = IQProcessor().process(raw, meta, pc, sampling, sps)
    if result.selection is None:
        raise ValueError("Cannot calculate symbol EVM: " + str(result.metadata["processing"]["stage_status"]["symbols_evm"]))
    result.metadata["processing"]["original_sources"] = provenance
    result.metadata["processing"]["csv_headers"] = headers
    result.metadata["processing"]["units"] = config["source_units"]
    result.metadata["processing"]["seed_provenance"] = config.get("seed_provenance", {})
    result.metadata["signal_config"]["samples_per_symbol_status"] = "derived_from_input_rates_and_verified_against_csv_clock"
    result.metadata["statistics"]["raw_voltage"] = {"input": signal_stats(raw.input.values), "output": signal_stats(raw.output.values)}
    CSVWriter().write(result, directory)
    JSONWriter.write(result.metadata, directory/"metadata.json")
    write_summary(result, directory/"summary.txt")
    if make_plots:
        plots.diagnostics(raw, sampling["sample_rate_hz"], directory)
        Grapher().plot(result, directory)
    return result


def support(model):
    delays = [0]
    for t in model.basis_terms:
        if t["kind"] == "envelope":
            delays.append(t["signal_delay"])
            if t["envelope_power"]:
                delays.append(t["envelope_delay"])
        else:
            delays.extend(t["direct_delays"]); delays.extend(t["conjugate_delays"])
    return max(delays), max(0, -min(delays))


def common_context(policies, limit):
    past = future = 0
    for name, policy in policies.items():
        trainer = TRAINERS[name]()
        for config in policy.reference_configs():
            if hasattr(trainer, "terms"):
                terms = trainer.terms(config)
                if len(terms) > limit:
                    raise ValueError(f"{name}: maximum structure exceeds max_basis_terms")
                ds = [d for term in terms for d in term.delays()]
                past = max(past, max(ds)); future = max(future, -min(ds))
    return past, future


def partition(alignment, gain, past, future):
    guard = max(past, future)
    idx = np.arange(guard+past, len(alignment.x)-guard-future)
    if len(idx) < 3:
        raise ValueError("Insufficient data after memory context and guards")
    return DataPartition(alignment.x, alignment.y/gain, alignment.time, idx)


def voltage_policy(signal, scale, limit, policy):
    v = checked_signal(checked_signal(signal)*scale)
    mask = (np.abs(v.real)>limit) | (np.abs(v.imag)>limit)
    if policy == "reject" and mask.any():
        raise ValueError("Voltage limit exceeded; export policy=reject")
    if policy not in ("zero_pair", "reject"):
        raise ValueError("Unknown voltage policy")
    result = v.copy()
    result[mask] = 0j
    return v, result, np.flatnonzero(mask)


def export_signals(run, config, aligned, model_names, symbol_indices):
    cfg = config["export"]
    fs = 1/np.median(np.diff(aligned.time))
    reports = {}
    for name in ["no_dpd", *model_names]:
        dest = run/"outputs"/name
        dest.mkdir(parents=True)
        if name == "no_dpd":
            value = aligned.x.copy(); indices = np.arange(len(value)); past = future = 0
        else:
            model = DPDModel(run/"model_results"/f"{name}.json", boundary=cfg["boundary"])
            value = model.apply(aligned.x)
            indices = model.original_indices
            past, future = model.past, model.future
        original, modified, affected = voltage_policy(value, cfg["voltage_scale"], cfg["max_abs_voltage"], cfg["overflow_policy"])
        is_symbol = np.isin(indices, symbol_indices)
        write_csv(dest/"samples.csv", [indices, np.arange(len(value))/fs, aligned.x[indices].real*cfg["voltage_scale"],
            aligned.x[indices].imag*cfg["voltage_scale"], original.real, original.imag, modified.real, modified.imag, is_symbol.astype(int)],
            ["aligned_index", "time_s", "I_desired_V", "Q_desired_V", "I_before_V", "Q_before_V", "I_export_V", "Q_export_V", "is_symbol"])
        record = export_to_cadence(modified, fs, dest/"I.txt", dest/"Q.txt", 1.0, cfg["max_abs_voltage"],
                                  cfg["interpolation"], cfg.get("transition_s"))
        record.update(model=name, direction="F(x); no division by gain", source_time_origin_s=float(aligned.time[indices[0]]),
            export_time_origin_s=0, first_aligned_index=int(indices[0]), last_aligned_index=int(indices[-1]),
            scale_applied_before_limit=cfg["voltage_scale"], overflow_policy=cfg["overflow_policy"],
            zeroed_output_indices=affected.tolist(), zeroed_aligned_indices=indices[affected].tolist(),
            zeroed_count=len(affected), before=signal_stats(original), after=signal_stats(modified),
            past_samples=past, future_samples=future, boundary=cfg["boundary"], offline_lookahead_s=future/fs,
            padding_affected_output_indices=(sorted(set(range(min(past,len(value)))) | set(range(max(0,len(value)-future),len(value)))) if cfg["boundary"]=="zero_pad" else []),
            model_sha256=digest(run/"model_results"/f"{name}.json") if name!="no_dpd" else None,
            reference_sha256=digest(run/"diagnostics"/"validation"/"aligned_waveform.csv"),
            status="exported_not_simulated")
        record["I_sha256"] = digest(dest/"I.txt"); record["Q_sha256"] = digest(dest/"Q.txt")
        JSONWriter.write(record, dest/"metadata.json")
        reports[name] = record
    return reports


def run_pipeline(config_path, stage="all", models_dir=None, make_plots=True):
    path, config = load_config(config_path)
    policies = {n: POLICIES[n](**limits) for n, limits in config["models"].items()}
    out = (path.parent/config["output_dir"]).resolve()
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run = out/stamp
    run.mkdir(parents=True, exist_ok=False)
    layout = create_run_directories(run)
    manifest = dict(schema_version="2.1", version=__version__, created_at_utc=stamp, status="running",
        artifact_language="en", directories=layout,
        config=config, python=platform.python_version(), numpy=np.__version__, stage=stage,
        selection_scope="validation used to choose winner; no independent test; no simulated PA improvement claim")
    JSONWriter.write(manifest, run/"manifest.json")
    print(f"Results: {run}", flush=True)
    try:
        results = {}
        for role in ("training", "validation"):
            print(f"Preprocessing {role}...", flush=True)
            results[role] = preprocess(path.parent/config[f"{role}_dir"], role, run/"diagnostics"/role, config, make_plots)
        a, b = (results[r].alignment for r in ("training", "validation"))
        if np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y):
            raise ValueError("Training and validation duplicate the same signals")
        ma, mb = (results[r].metadata for r in ("training", "validation"))
        for key in ("modulation", "symbol_rate_hz", "carrier_frequency_hz"):
            if ma["signal_config"].get(key) != mb["signal_config"].get(key):
                raise ValueError(f"Incompatible datasets: {key}")
        if not np.isclose(np.median(np.diff(a.time)), np.median(np.diff(b.time)), rtol=1e-5, atol=1e-18):
            raise ValueError("Incompatible training/validation sample periods")
        manifest["datasets"] = {r: dict(sources=results[r].metadata["processing"]["original_sources"],
            aligned_sha256=digest(run/"diagnostics"/r/"aligned_waveform.csv"), aligned_samples=len(results[r].alignment.x)) for r in results}
        manifest["warnings"] = ["Different declared seeds do not prove statistical independence of periodic PN sequences."]
        if stage == "preprocess":
            manifest["status"] = "completed"
        else:
            models = {}
            if models_dir:
                source = Path(models_dir).resolve()
                source_models, source_manifest = resolve_saved_models(source)
                old = JSONWriter.read(source_manifest)
                gain = old["gain"]
                if not np.isscalar(gain) or not np.isfinite(gain) or abs(gain)==0:
                    raise ValueError("Saved gain must be finite and nonzero")
                for name in policies:
                    shutil.copyfile(source_models/f"{name}.json", run/"model_results"/f"{name}.json")
                    models[name] = JSONWriter.load_model(run/"model_results"/f"{name}.json")
                    if models[name].model_name != name:
                        raise ValueError("Model filename/family mismatch")
                past = max(support(m)[0] for m in models.values()); future = max(support(m)[1] for m in models.values())
                manifest["reused_models"] = {"source": str(source), "manifest_sha256": digest(source_manifest),
                    "note": "Original training gain reused; models not retrained; see source manifest for training provenance"}
            else:
                past, future = common_context(policies, config["search"]["max_basis_terms"])
                gain_idx = partition(a, 1, past, future).indices
                gain = DPDProcessor.estimate_gain(a.x[gain_idx], a.y[gain_idx])
            train, validation = partition(a, gain, past, future), partition(b, gain, past, future)
            manifest.update(gain=gain, common_past=past, common_future=future, guard_samples_each_edge=max(past,future),
                training_scored=len(train.indices), validation_scored=len(validation.indices),
                validation_start_index=int(validation.indices[0]), validation_end_index=int(validation.indices[-1]))
            JSONWriter.write(manifest, run/"manifest.json")
            hardware = dict(config["hardware"])
            estimator = ESTIMATORS[hardware.pop("mode")](**hardware)
            if not models_dir:
                for name, policy in policies.items():
                    print(f"BIC training: {name}...", flush=True)
                    result = TrainingBICSearch(TRAINERS[name](**config["numerics"]), policy, estimator, config["search"]).run(train)
                    JSONWriter.write(result, run/"model_results"/f"{name}.json")
                    models[name] = result["model"]
                    print(f"  {name}: {result['best_config']} - {result['optimization_trace']['stop_reason']}", flush=True)
            rows, predictions, symbol_predictions = [], {}, {}
            symbol_mask = np.isin(validation.indices, results["validation"].selection.indices)
            symbols = validation.indices[symbol_mask]
            if len(symbols)<1:
                raise ValueError("No common symbols after guards")
            comparison = run/"model_comparisons"
            for name, model in models.items():
                predicted = model.predict(validation)
                full_error = normalized_error(validation.target, predicted)
                symbol_error = normalized_error(b.x[symbols], predicted[symbol_mask])
                rows.append(dict(model=name, config=model.config, nmse_waveform_db=error_to_db(full_error),
                    nmse_symbols_db=error_to_db(symbol_error), relative_rms_percent=100*np.sqrt(full_error),
                    symbol_relative_rms_percent=100*np.sqrt(symbol_error), num_coefficients=len(model.coefficients),
                    samples=len(validation.indices), symbols=len(symbols), hardware=asdict(estimator.estimate(model, model.config)),
                    model_sha256=digest(run/"model_results"/f"{name}.json")))
                predictions[name]=predicted; symbol_predictions[name]=predicted[symbol_mask]
                write_csv(comparison/f"{name}_waveform.csv", [validation.indices,b.time[validation.indices],validation.target.real,validation.target.imag,predicted.real,predicted.imag],
                          ["aligned_index","time_s","I_reference_V","Q_reference_V","I_reconstructed_V","Q_reconstructed_V"])
                write_csv(comparison/f"{name}_symbols.csv", [symbols,b.time[symbols],b.x[symbols].real,b.x[symbols].imag,predicted[symbol_mask].real,predicted[symbol_mask].imag],
                          ["aligned_index","time_s","I_reference_V","Q_reference_V","I_reconstructed_V","Q_reconstructed_V"])
            winner = min(rows, key=lambda r:(r["nmse_waveform_db"],r["num_coefficients"],r["model"]))["model"]
            JSONWriter.write(dict(winner=winner, selection_metric="nmse_waveform_db", rows=rows,
                symbol_indices=symbols, phase=results["validation"].selection.phase,
                note="Reconstruction F(y/G); no voltage limiting of reconstruction metrics"), comparison/"metrics.json")
            if make_plots:
                plots.constellations(b.x[symbols],symbol_predictions,rows,comparison)
                plots.errors(validation.indices,validation.target,predictions,comparison)
            exports = export_signals(run,config,b,list(models) if config["export"]["models"]=="all" else [winner],results["validation"].selection.indices) if stage=="all" else {}
            manifest.update(status="completed", winner=winner, selection_metric="validation_nmse_waveform_db", exported=list(exports),
                zeroed_samples={n:r["zeroed_count"] for n,r in exports.items()})
            lines=["# Reconstruction comparison", "", "Selection uses the full usable validation set; no independent test set.",
                   f"Winner by waveform NMSE: **{winner}**.", "", "| Model | Waveform NMSE (dB) | Symbol NMSE (dB) | Coefficients |", "|---|---:|---:|---:|"]
            lines += [f"| {r['model']} | {r['nmse_waveform_db']:.4f} | {r['nmse_symbols_db']:.4f} | {r['num_coefficients']} |" for r in rows]
            lines += ["", "Constellations show symbol instants only. Models use the full waveform.",
                      "PWL: F(x), with voltage limiting after scaling; these are not PA simulation results with DPD.",
                      "PA improvement must be measured by simulating in Cadence and analyzing the returned signal."]
            (comparison/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        JSONWriter.write(manifest,run/"manifest.json")
        JSONWriter.write(dict(directory=str(run),status=manifest["status"]),out/"latest_run.json")
    except Exception as exc:
        manifest.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        JSONWriter.write(manifest,run/"manifest.json")
        raise
    return run
