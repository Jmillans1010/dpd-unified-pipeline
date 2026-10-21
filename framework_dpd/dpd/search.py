from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from .basis import polynomial_terms, gmp_terms, volterra_terms, PruningRule
from .contracts import ModelConfig


class SearchPolicy(ABC):
    @abstractmethod
    def minimal_config(self) -> ModelConfig: ...
    @abstractmethod
    def reference_configs(self) -> list[ModelConfig]: ...
    @abstractmethod
    def generate_neighbors(self, config: ModelConfig) -> list[ModelConfig]: ...
    @abstractmethod
    def structural_complexity(self, config: ModelConfig) -> float: ...
    @abstractmethod
    def is_valid(self, config: ModelConfig) -> bool: ...


class MPSearchPolicy(SearchPolicy):
    def __init__(self, max_P: int = 7, max_M: int = 3):
        if max_P < 1 or max_P % 2 != 1 or max_M < 0:
            raise ValueError("Límites P/M inválidos.")
        self.max_P, self.max_M = max_P, max_M

    def minimal_config(self) -> ModelConfig:
        return {"P": 1, "M": 0}

    def reference_configs(self) -> list[ModelConfig]:
        return [{"P": self.max_P, "M": self.max_M},
                {"P": max(1, self.max_P-2), "M": self.max_M}]

    def generate_neighbors(self, config: ModelConfig) -> list[ModelConfig]:
        return [c for c in ({**config, "P": config["P"]+2}, {**config, "M": config["M"]+1}) if self.is_valid(c)]

    def structural_complexity(self, config: ModelConfig) -> float:
        return float(len(polynomial_terms(config)))

    def is_valid(self, config: ModelConfig) -> bool:
        return 1 <= config["P"] <= self.max_P and config["P"] % 2 == 1 and 0 <= config["M"] <= self.max_M


class EMPSearchPolicy(MPSearchPolicy):
    def structural_complexity(self, config: ModelConfig) -> float:
        return float(len(polynomial_terms(config, envelope=True)))


class GMPSearchPolicy(SearchPolicy):
    def __init__(self, max_P: int = 7, max_M: int = 3, max_Q: int = 2):
        MPSearchPolicy(max_P, max_M)
        if max_Q < 1:
            raise ValueError("max_Q debe ser positivo.")
        self.max_P, self.max_M, self.max_Q = max_P, max_M, max_Q

    def minimal_config(self) -> ModelConfig:
        return {"aligned": {"P": 1, "M": 0}, "lagging": None, "leading": None}

    def reference_configs(self) -> list[ModelConfig]:
        full = {"aligned": {"P": self.max_P, "M": self.max_M},
                "lagging": {"P": self.max_P, "M": self.max_M, "Q": self.max_Q},
                "leading": {"P": self.max_P, "M": self.max_M, "Q": self.max_Q}}
        if self.max_P < 3 or self.max_M < 1:
            full["lagging"] = full["leading"] = None
        return [full, {**deepcopy(full), "leading": None}, {**deepcopy(full), "lagging": None}]

    def generate_neighbors(self, config: ModelConfig) -> list[ModelConfig]:
        result = []
        for branch in ("aligned", "lagging", "leading"):
            if config[branch] is None:
                c = deepcopy(config)
                c[branch] = {"P": 3, "M": 1, "Q": 1}
                result.append(c)
            else:
                moves = [("P", 2), ("M", 1)]
                if branch != "aligned":
                    moves.append(("Q", 1))
                for key, step in moves:
                    c = deepcopy(config)
                    c[branch][key] += step
                    result.append(c)
        return [c for c in result if self.is_valid(c)]

    def structural_complexity(self, config: ModelConfig) -> float:
        return float(len(gmp_terms(config)))

    def is_valid(self, config: ModelConfig) -> bool:
        if not MPSearchPolicy(self.max_P, self.max_M).is_valid(config["aligned"]):
            return False
        return all(c is None or (3 <= c["P"] <= self.max_P and c["P"] % 2 == 1 and
                   1 <= c["M"] <= self.max_M and 1 <= c["Q"] <= self.max_Q)
                   for c in (config["lagging"], config["leading"]))


class PrunedVolterraSearchPolicy(MPSearchPolicy):
    def __init__(self, max_P: int = 5, max_M: int = 2, max_D: int = 1,
                 pruning_rule: PruningRule | None = None):
        super().__init__(max_P, max_M)
        if max_D < 0:
            raise ValueError("max_D no puede ser negativo.")
        self.max_D, self.pruning_rule = max_D, pruning_rule

    def minimal_config(self) -> ModelConfig:
        return {"P": 1, "M": 0, "D": 0}

    def reference_configs(self) -> list[ModelConfig]:
        return [{"P": self.max_P, "M": self.max_M, "D": min(self.max_D, self.max_M)},
                {"P": max(1, self.max_P-2), "M": self.max_M, "D": min(self.max_D, self.max_M)}]

    def generate_neighbors(self, config: ModelConfig) -> list[ModelConfig]:
        return super().generate_neighbors(config) + ([{**config, "D": config["D"]+1}]
                if config["D"] < min(self.max_D, config["M"]) else [])

    def structural_complexity(self, config: ModelConfig) -> float:
        return float(len(volterra_terms(config, self.pruning_rule)))

    def is_valid(self, config: ModelConfig) -> bool:
        return super().is_valid(config) and 0 <= config["D"] <= min(self.max_D, config["M"])


class LUTSearchPolicy(SearchPolicy):
    methods = ("uniform_amplitude", "uniform_power", "quantile")

    def __init__(self, min_bins: int = 8, max_bins: int = 1024):
        if min_bins < 2 or max_bins < min_bins or (max_bins/min_bins) % 1 or (int(max_bins/min_bins) & (int(max_bins/min_bins)-1)):
            raise ValueError("Bins deben crecer por potencias de dos desde min_bins.")
        self.min_bins, self.max_bins = min_bins, max_bins

    def minimal_config(self) -> ModelConfig:
        return {"bins": self.min_bins, "binning": "uniform_amplitude", "interpolation": "nearest"}

    def reference_configs(self) -> list[ModelConfig]:
        # Two sizes protect against high-resolution overfitting and empty cells.
        return [{"bins": b, "binning": method, "interpolation": interp}
                for b in sorted({max(self.min_bins, self.max_bins//2), self.max_bins})
                for method in self.methods for interp in ("nearest", "linear")]

    def generate_neighbors(self, config: ModelConfig) -> list[ModelConfig]:
        result = [{**config, "binning": method} for method in self.methods if method != config["binning"]]
        if config["bins"] < self.max_bins:
            result.append({**config, "bins": config["bins"]*2})
        if config["interpolation"] == "nearest":
            result.append({**config, "interpolation": "linear"})
        return [c for c in result if self.is_valid(c)]

    def structural_complexity(self, config: ModelConfig) -> float:
        return float(config["bins"] + (4 if config["interpolation"] == "linear" else 0))

    def is_valid(self, config: ModelConfig) -> bool:
        ratio = config["bins"] / self.min_bins
        return (self.min_bins <= config["bins"] <= self.max_bins and ratio.is_integer() and
                int(ratio) & (int(ratio)-1) == 0 and config["binning"] in self.methods and
                config["interpolation"] in ("nearest", "linear"))
