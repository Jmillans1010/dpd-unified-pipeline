"""Lectores de CSV Cadence y del subconjunto de Spectre usado por este proyecto."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shlex
import numpy as np
from .models import IQSignal, InputData, InputMetadata, ProcessingError


def fingerprint(path: Path, root: Path, role: str) -> dict:
    import os
    return {"role": role, "path": Path(os.path.relpath(path, root)).as_posix(),
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class CSVReader:
    """Dos columnas con cabecera: tiempo en segundos y valor I o Q."""

    def read(self, files: dict, root: Path, config) -> InputData:
        arrays, sources = {}, []
        for role, relative in files.items():
            path = root / relative
            if not path.is_file():
                raise ProcessingError(f"Falta el CSV {relative}.")
            with path.open(encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
            if len(header) != 2:
                raise ProcessingError(f"{relative}: se esperan exactamente dos columnas con cabecera.")
            try:
                arr = np.loadtxt(path, delimiter=",", skiprows=1, encoding="utf-8-sig", ndmin=2)
            except (ValueError, OSError) as exc:
                raise ProcessingError(f"CSV inválido {relative}: {exc}") from exc
            if arr.shape[1] != 2 or len(arr) < 3 or not np.isfinite(arr).all():
                raise ProcessingError(f"{relative}: mínimo tres filas, dos columnas y valores finitos.")
            arrays[role] = arr
            sources.append(fingerprint(path, root, role))
        signals = []
        for side in ("in", "out"):
            i, q = arrays[f"I_{side}"], arrays[f"Q_{side}"]
            # Tolerancia relativa al intervalo, no al tiempo absoluto.
            dt = np.median(np.diff(i[:, 0]))
            tolerance = config.time_atol_s + config.time_rtol * abs(dt)
            if i.shape != q.shape or np.max(np.abs(i[:, 0] - q[:, 0])) > tolerance:
                raise ProcessingError(f"I/Q de {side} tienen tiempos o longitudes diferentes.")
            signals.append(IQSignal(i[:, 0], i[:, 1] + 1j*q[:, 1], np.arange(len(i))))
        return InputData(*signals, sources)


def spectre_number(raw):
    """Convierte literales; conserva expresiones sin evaluarlas."""
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)([TGMKkmunpfa]?)", str(raw))
    if not match:
        return None
    scale = {"": 1., "T": 1e12, "G": 1e9, "M": 1e6, "K": 1e3, "k": 1e3,
             "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15, "a": 1e-18}
    value = float(match[1]) * scale[match[2]]
    return value if np.isfinite(value) else None


def logical_lines(text):
    """Elimina comentarios // fuera de comillas y une continuaciones."""
    pending = ""
    for raw in text.splitlines():
        quote = False
        stop = len(raw)
        for i, char in enumerate(raw):
            if char == '"' and (i == 0 or raw[i-1] != "\\"):
                quote = not quote
            if not quote and raw[i:i+2] == "//":
                stop = i
                break
        line = raw[:stop].strip()
        if not line:
            continue
        continued = line.endswith("\\")
        pending += " " + (line[:-1] if continued else line)
        if not continued:
            yield pending.strip()
            pending = ""
    if pending:
        raise ProcessingError("El netlist termina con una continuación incompleta.")


class InputSCSReader:
    def read(self, path: Path | None) -> InputMetadata:
        result = InputMetadata()
        if path is None or not path.is_file():
            result.warnings.append("Netlist no disponible; se usarán otras fuentes de configuración.")
            return result
        lines = list(logical_lines(path.read_text(encoding="utf-8-sig")))
        if not any(re.match(r"simulator\s+lang\s*=\s*spectre\b", line) for line in lines):
            raise ProcessingError("El archivo de netlist no declara simulator lang=spectre.")
        result.simulation_config["simulator"] = "spectre"
        raw_statements, includes, depth = {}, [], 0
        transient, options, source = None, None, None
        for line in lines:
            if line.startswith("subckt "):
                depth += 1
                continue
            if line.startswith("ends ") or line == "ends":
                depth -= 1
                continue
            if depth:
                continue
            if line.startswith(("include ", "ahdl_include ")):
                includes.append(line)
                continue
            tokens = shlex.split(line, posix=True)
            params = dict(re.findall(r'(\w+)\s*=\s*("[^"\n]*"|[^\s]+)', line))
            params = {k: v.strip('"') for k, v in params.items()}
            if len(tokens) < 2:
                continue
            if tokens[1] == "tran":
                if transient is not None:
                    raise ProcessingError("Hay varios análisis tran; seleccione un netlist con un único análisis.")
                transient = (tokens[0], params)
            elif tokens[1] == "options":
                options = (tokens[0], params)
            elif re.search(r"\)\s+wsource\b", line):
                if source is not None:
                    raise ProcessingError("Hay varias fuentes wsource; se requiere selección explícita.")
                source = (tokens[0], params)
            if params:
                raw_statements[tokens[0]] = params
        result.simulation_config["raw_statements"] = raw_statements
        result.simulation_config["includes_not_expanded"] = includes

        def extract(statement, mapping, target, section):
            if statement is None:
                return
            instance, parameters = statement
            for raw_name, (field, numeric) in mapping.items():
                if raw_name not in parameters:
                    continue
                raw = parameters[raw_name]
                value = spectre_number(raw) if numeric else raw
                target[field] = value
                result.provenance[f"{section}.{field}"] = {
                    "source": "netlist", "parameter": f"{instance}.{raw_name}", "raw": raw}
                if numeric and value is None:
                    result.warnings.append(f"Parámetro no resuelto: {instance}.{raw_name}={raw}")

        extract(transient, {"stop": ("stop_s", True), "outputstart": ("output_start_s", True),
                           "strobeperiod": ("strobe_period_s", True), "maxstep": ("max_step_s", True),
                           "strobeoutput": ("strobe_output", False), "errpreset": ("errpreset", False)},
                result.simulation_config, "simulation_config")
        extract(options, {"temp": ("temperature_c", True), "reltol": ("reltol", True),
                          "vabstol": ("vabstol", True), "iabstol": ("iabstol", True)},
                result.simulation_config, "simulation_config")
        extract(source, {"MSource_ModType": ("modulation", False),
                         "MSource_Frequency": ("carrier_frequency_hz", True),
                         "MSource_DataRate": ("source_data_rate_value", True),
                         "oversample": ("source_oversample", True),
                         "MSource_SymbolNum": ("source_symbol_count", True),
                         "MSource_DataType": ("data_type", False),
                         "MSource_FilterType": ("source_filter", False),
                         "dbm": ("source_level_dbm", True), "r": ("source_impedance_ohm", True)},
                result.signal_config, "signal_config")
        return result


class MetadataResolver:
    def resolve(self, metadata, config, root, sources, sampling):
        previous = None
        if config.previous_metadata:
            path = root / config.previous_metadata
            previous = json.loads(path.read_text(encoding="utf-8-sig"))
            old = {s["role"]: s["sha256"] for s in previous.get("processing", {}).get("sources", [])}
            if any(old.get(s["role"]) != s["sha256"] for s in sources if s["role"] != "netlist"):
                raise ProcessingError("El JSON anterior no corresponde a los cuatro CSV actuales (SHA-256).")
        for section in ("simulation_config", "signal_config"):
            target = getattr(metadata, section)
            for source_name, obj in (("previous_metadata", previous or {}), ("manual_metadata", config.manual_metadata)):
                values = obj.get(section, {})
                if not isinstance(values, dict):
                    raise ProcessingError(f"{source_name}.{section} debe ser un objeto.")
                for key, value in values.items():
                    if key in ("raw_statements", "includes_not_expanded"):
                        continue
                    if target.get(key) is None:
                        target[key] = value
                        metadata.provenance[f"{section}.{key}"] = {"source": source_name}
                    elif value is not None and target[key] != value:
                        raise ProcessingError(f"Conflicto de metadatos: {section}.{key}.")
        signal = metadata.signal_config
        rate = signal.get("symbol_rate_hz")
        if config.source_data_rate_is_bit_rate and rate is None:
            modulation = re.fullmatch(r"QAM(\d+)", str(signal.get("modulation", "")))
            bit_rate = signal.get("source_data_rate_value")
            if modulation and bit_rate:
                order = int(modulation[1])
                if order >= 2 and order & (order - 1) == 0:
                    rate = float(bit_rate) / np.log2(order)
                    signal["symbol_rate_hz"] = rate
                    metadata.provenance["signal_config.symbol_rate_hz"] = {
                        "source": "derived", "assumption": "source_data_rate_is_bit_rate explicitly enabled"}
        inferred = None
        if rate is not None:
            if not isinstance(rate, (float, int)) or not np.isfinite(rate) or rate <= 0:
                raise ProcessingError("symbol_rate_hz debe ser positivo y finito.")
            ratio = sampling["sample_rate_hz"] / rate
            if abs(ratio - round(ratio)) > 1e-5 or ratio < 1:
                raise ProcessingError("Fs/Rs no es entero; esta versión necesita una rejilla con L entero.")
            inferred = round(ratio)
        provided = signal.get("samples_per_symbol")
        choices = [v for v in (config.samples_per_symbol, provided, inferred) if v is not None]
        for value in choices:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 1 or int(value) != value:
                raise ProcessingError("samples_per_symbol debe ser un entero positivo.")
        if len(set(choices)) > 1:
            raise ProcessingError("Conflicto entre muestras/símbolo configuradas y Fs/Rs o metadatos.")
        sps = int(choices[0]) if choices else None
        signal["samples_per_symbol"] = sps
        signal["samples_per_symbol_status"] = "verified_by_configuration" if config.samples_per_symbol_verified else "unverified"
        if sps is not None and not config.samples_per_symbol_verified:
            metadata.warnings.append(f"Se usan {sps} muestras/símbolo; su correspondencia física no está verificada.")
        metadata.provenance["signal_config.samples_per_symbol"] = {
            "source": "processing_config" if config.samples_per_symbol is not None else ("metadata" if provided is not None else "derived_from_rates"),
            "source_oversample_is_not_exported_sps": True}
        strobe = metadata.simulation_config.get("strobe_period_s")
        if strobe is not None and not np.isclose(strobe, sampling["sample_period_s"], rtol=config.time_rtol, atol=config.time_atol_s):
            metadata.warnings.append("El intervalo observado del CSV difiere de strobeperiod; se usan los tiempos observados.")
        return sps
