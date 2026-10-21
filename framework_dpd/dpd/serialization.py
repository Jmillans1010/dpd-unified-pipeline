from __future__ import annotations
from dataclasses import fields, is_dataclass
from pathlib import Path
import json
import math
import numpy as np
from .contracts import TrainedModel, FitDiagnostics, ModelResult


def encode(value):
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "dtype": str(value.dtype), "values": encode(value.tolist())}
    if isinstance(value, (complex, np.complexfloating)):
        return {"__complex__": [float(value.real), float(value.imag)]}
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else {"__nonfinite__": str(float(value))}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def decode_object(value: dict):
    if "__complex__" in value:
        return complex(*value["__complex__"])
    if "__ndarray__" in value:
        return np.asarray(value["values"], dtype=value["dtype"])
    if "__nonfinite__" in value:
        return float(value["__nonfinite__"])
    return value


class JSONWriter:
    @staticmethod
    def write(value, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix+".tmp")
        temporary.write_text(json.dumps(encode(value), indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
        temporary.replace(path)

    def write_all(self, results: list[ModelResult], directory: Path) -> None:
        for result in results:
            self.write(result, directory / f"{result.model_name}.json")

    @staticmethod
    def read(path: str | Path):
        return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_hook=decode_object)

    @classmethod
    def load_model(cls, path: str | Path) -> TrainedModel:
        result = cls.read(path)
        model = result.get("model", result)
        if model is None:
            raise ValueError("El resultado no contiene un modelo válido.")
        return TrainedModel(**{**model, "diagnostics": FitDiagnostics(**model["diagnostics"])})
