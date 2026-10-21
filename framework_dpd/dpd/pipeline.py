from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import json
import platform
import numpy as np
from . import __version__
from .contracts import OptimizationCriteria, ModelResult, AlignedInputData
from .data import CSVReader, DataProcessor
from .hardware import DirectHardwareEstimator, LUTAssistedHardwareEstimator, MultiplierSharingEstimator
from .optimizer import HyperparameterOptimizer
from .search import MPSearchPolicy, EMPSearchPolicy, GMPSearchPolicy, PrunedVolterraSearchPolicy, LUTSearchPolicy
from .serialization import JSONWriter
from .trainers import TRAINERS


POLICIES = {"MP": MPSearchPolicy, "EMP": EMPSearchPolicy, "GMP": GMPSearchPolicy,
            "PrunedVolterra": PrunedVolterraSearchPolicy, "LUT": LUTSearchPolicy}
ESTIMATORS = {"direct": DirectHardwareEstimator, "lut_assisted": LUTAssistedHardwareEstimator,
              "multiplier_sharing": MultiplierSharingEstimator}


class TrainingPipeline:
    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        self.base = self.config_path.parent
        self.policies = {name: POLICIES[name](**limits) for name, limits in self.config["models"].items()}
        if not self.policies:
            raise ValueError("Seleccione al menos una arquitectura.")
        hardware = dict(self.config.get("hardware", {}))
        self.estimator = ESTIMATORS[hardware.pop("mode", "direct")](**hardware)
        self.criteria = OptimizationCriteria(**self.config.get("optimization", {}))
        self.data: AlignedInputData | None = None

    def prepare_data(self) -> AlignedInputData:
        # Actual basis supports determine common context, including GMP lookahead.
        past = future = 0
        for name, policy in self.policies.items():
            trainer = TRAINERS[name]()
            if hasattr(trainer, "terms"):
                for config in policy.reference_configs():
                    terms = trainer.terms(config)
                    if len(terms) > self.config.get("max_basis_terms", 2048):
                        raise ValueError(f"{name}: referencia excede max_basis_terms.")
                    delays = [d for term in terms for d in term.delays()]
                    past, future = max(past, max(delays)), max(future, -min(delays))
        processor = DataProcessor(past, future, self.config.get("guard_samples"))
        reader = CSVReader()
        inputs = self.config["inputs"]
        training = reader.read(self.base / inputs["training"])
        validation = reader.read(self.base / inputs["validation"])
        mode = self.config.get("split_mode", "validation_holdout")
        if mode == "validation_holdout":
            fraction = self.config.get("validation_fraction", .5)
            if not 0 < fraction < 1:
                raise ValueError("validation_fraction debe estar entre 0 y 1.")
            boundary = int(len(validation.x)*fraction)
            original_count = len(validation.x)
            test = processor.subset(validation, boundary+processor.guard, original_count)
            validation = processor.subset(validation, 0, boundary)
            data = processor.prepare(training, validation, test)
            data.provenance.update(split_mode="training_folder_plus_chronological_validation_holdout",
                                   validation_boundary=boundary, validation_original_samples=original_count,
                                   excluded_input=inputs.get("test"), validation_fraction=fraction)
        elif mode == "provided_partitions":
            test = reader.read(self.base / inputs["test"])
            data = processor.prepare(training, validation, test, self.config.get("duplicate_policy", "error"))
        else:
            raise ValueError("split_mode desconocido.")
        self.data = data
        return data

    def run_model(self, model_name: str) -> ModelResult:
        if self.data is None:
            self.prepare_data()
        trainer = TRAINERS[model_name](**self.config.get("numerics", {}))
        return HyperparameterOptimizer(trainer, self.policies[model_name], self.estimator, self.criteria).optimize(self.data)

    def run(self) -> Path:
        data = self.prepare_data()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        directory = self.base / self.config.get("output_directory", "data/dpd_results") / f"run_{stamp}"
        directory.mkdir(parents=True, exist_ok=False)
        writer = JSONWriter()
        manifest = {"schema_version": "1.1", "dpd_version": __version__, "created_at_utc": stamp,
                    "python": platform.python_version(), "numpy": np.__version__, "config": self.config,
                    "gain": data.gain, "provenance": data.provenance,
                    "scope": "inverse/postdistorter z=y/G -> x; no PA application or final EVM/ACLR",
                    "status": "running"}
        writer.write(manifest, directory / "manifest.json")
        results = []
        try:
            for name in self.policies:
                print(f"Entrenando {name}...", flush=True)
                result = self.run_model(name)
                writer.write(result, directory / f"{name}.json")
                results.append(result)
                detail = (f"BIC={result.optimization_trace.selected_bic_score_db}, "
                          f"umbral/coef={result.optimization_trace.bic_penalty_db_per_coefficient} dB"
                          if result.optimization_trace.stopping_rule == "bic" else f"progreso={result.progress_fraction}")
                print(f"  {result.optimization_trace.stop_reason}: validation={result.nmse_validation_db}, test={result.nmse_test_db}, {detail}", flush=True)
            summary = [{"model": r.model_name, "config": r.best_config, "nmse_train_db": r.nmse_train_db,
                        "nmse_validation_db": r.nmse_validation_db, "nmse_test_db": r.nmse_test_db,
                        "progress_fraction": r.progress_fraction, "structural_complexity": r.structural_complexity,
                        "stopping_rule": r.optimization_trace.stopping_rule,
                        "bic_score_db": r.optimization_trace.selected_bic_score_db,
                        "bic_coefficients": r.model.diagnostics.rank if r.model else None,
                        "bic_penalty_db_per_coefficient": r.optimization_trace.bic_penalty_db_per_coefficient,
                        "search_boundary_reached": r.optimization_trace.boundary_reached,
                        "stop_reason": r.optimization_trace.stop_reason,
                        "evaluated_candidates": len(r.optimization_trace.candidates)} for r in results]
            writer.write(summary, directory / "summary.json")
            lines = ["# Identificación inversa DPD", "", "Test: segunda mitad de Validation, excluida de selección de hiperparámetros." if self.config.get("split_mode", "validation_holdout") == "validation_holdout" else "Test: partición proporcionada.", "",
                     "| Modelo | NMSE train dB | NMSE val dB | NMSE test dB | BIC dB / progreso histórico | Complejidad | Parada |",
                     "|---|---:|---:|---:|---:|---:|---|"]
            def fmt(value):
                return f"{value:.6g}" if value is not None else "N/D"
            for r in results:
                criterion = fmt(r.optimization_trace.selected_bic_score_db) if r.optimization_trace.stopping_rule == "bic" else f"{r.progress_fraction:.2%}"
                lines.append(f"| {r.model_name} | {fmt(r.nmse_train_db)} | {fmt(r.nmse_validation_db)} | {fmt(r.nmse_test_db)} | {criterion} | {fmt(r.structural_complexity)} | {r.optimization_trace.stop_reason} |")
            lines += ["", "Son errores de identificación del inverso, no EVM/ACLR del PA linealizado.",
                      "BIC usa error de entrenamiento y penaliza coeficientes complejos según Becerra et al. (2020). Menor es mejor.",
                      "Una parada local por BIC no garantiza un óptimo global. Los límites e inestabilidad se informan por separado.", "",
                      *[f"- {w}" for w in data.provenance["warnings"]]]
            (directory / "summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
            manifest["status"] = "completed" if all(r.model is not None for r in results) else "completed_with_unavailable_models"
            writer.write(manifest, directory / "manifest.json")
            writer.write({"directory": str(directory.resolve()), "status": manifest["status"]}, directory.parent / "latest_run.json")
        except Exception as exc:
            manifest.update(status="failed", error=str(exc))
            writer.write(manifest, directory / "manifest.json")
            raise
        return directory
