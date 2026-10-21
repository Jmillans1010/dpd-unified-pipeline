"""Bases explícitas: el mismo catálogo alimenta entrenamiento y hardware."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from itertools import combinations_with_replacement
from .contracts import ModelConfig


@dataclass(frozen=True)
class BasisTerm:
    kind: str
    signal_delay: int = 0
    envelope_delay: int = 0
    envelope_power: int = 0
    direct_delays: tuple[int, ...] = ()
    conjugate_delays: tuple[int, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def delays(self) -> tuple[int, ...]:
        if self.kind == "volterra":
            return self.direct_delays + self.conjugate_delays
        return (self.signal_delay, self.envelope_delay) if self.envelope_power else (self.signal_delay,)


class PruningRule(ABC):
    @abstractmethod
    def keep(self, term: BasisTerm, config: ModelConfig) -> bool: ...


class MaxDelaySpread(PruningRule):
    def keep(self, term: BasisTerm, config: ModelConfig) -> bool:
        delays = term.delays()
        return max(delays) - min(delays) <= config["D"]


def polynomial_terms(config: ModelConfig, envelope: bool = False) -> list[BasisTerm]:
    terms = []
    for p in range(1, config["P"]+1, 2):
        for m in range(config["M"]+1):
            # EMP linear terms z[n] repeated for each m would be identical.
            if envelope and p == 1 and m > 0:
                continue
            terms.append(BasisTerm("envelope", 0 if envelope else m, m, p-1))
    return terms


def gmp_terms(config: ModelConfig) -> list[BasisTerm]:
    terms = polynomial_terms(config["aligned"])
    for branch, sign in (("lagging", 1), ("leading", -1)):
        c = config[branch]
        if c is None:
            continue
        for p in range(3, c["P"]+1, 2):
            for m in range(c["M"]+1):
                for q in range(1, c["Q"]+1):
                    terms.append(BasisTerm("envelope", m, m+sign*q, p-1))
    return terms


def generate_volterra_terms(config: ModelConfig):
    """Complejo banda base: (p+1)/2 factores z, (p-1)/2 conjugados.

    Simetría dentro de cada grupo; no duplicar permutaciones conmutativas.
    """
    delays = range(config["M"]+1)
    for p in range(1, config["P"]+1, 2):
        for direct in combinations_with_replacement(delays, (p+1)//2):
            for conjugate in combinations_with_replacement(delays, (p-1)//2):
                yield BasisTerm("volterra", direct_delays=direct, conjugate_delays=conjugate)


def volterra_terms(config: ModelConfig, rule: PruningRule | None = None) -> list[BasisTerm]:
    rule = rule or MaxDelaySpread()
    return [t for t in generate_volterra_terms(config) if rule.keep(t, config)]
