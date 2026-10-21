"""Estimaciones de un datapath documentado, no resultados de síntesis FPGA."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import replace
from math import ceil, log2
from .contracts import TrainedModel, ModelConfig, HardwareComplexity
from .basis import BasisTerm


class HardwareComplexityEstimator(ABC):
    @abstractmethod
    def estimate(self, model: TrainedModel, config: ModelConfig) -> HardwareComplexity: ...


class DirectHardwareEstimator(HardwareComplexityEstimator):
    def __init__(self, word_bits: int = 18):
        if word_bits < 2:
            raise ValueError("word_bits debe ser >= 2.")
        self.word_bits = word_bits

    def estimate(self, model: TrainedModel, config: ModelConfig) -> HardwareComplexity:
        n = model.diagnostics.num_coefficients
        assumptions = ["Estimación analítica de datapath paralelo; no síntesis ni cierre temporal.",
                       "Contadores reales excluyen los operadores ya contados como complejos.",
                       "Multiplicación compleja convencional equivale a 4 reales + 2 sumas reales.",
                       f"Palabra real {self.word_bits} bits; coeficiente complejo 2 palabras.",
                       "No incluye interfaces, control, normalización y/G ni cuantización de coeficientes."]
        if model.lut_values is not None:
            linear = config["interpolation"] == "linear"
            # Arbitrary nodes use binary search; intervals use reciprocal widths.
            entries = n*2 + (n-1 if linear else 0)
            return HardwareComplexity(n, 1, 2 + (3 if linear else 0), 2 if linear else 0,
                                      1 + (1 if linear else 0), 0, 3 if linear else 2, entries,
                                      self.word_bits*(3*n + (n-1 if linear else 0)), 1, 1.0,
                                      assumptions + ["LUT: n ganancias complejas, n nodos reales; interpolación almacena n-1 anchos inversos.",
                                                     "Magnitud requiere raíz cuadrada; búsqueda binaria desenrollada; latencia no modelada."],
                                      {"sqrt": 1, "comparators": ceil(log2(n)), "envelope_power_multiplies": 0})
        terms = [BasisTerm(**t) for t in model.basis_terms]
        envelope_powers: dict[int, int] = {}
        delays = [d for term in terms for d in term.delays()]
        cm, rm, ra = n, 0, 0  # one coefficient product per term
        for term in terms:
            if term.kind == "envelope" and term.envelope_power:
                envelope_powers[term.envelope_delay] = max(envelope_powers.get(term.envelope_delay, 0), term.envelope_power//2)
                rm += 2  # complex signal times real power
            elif term.kind == "volterra":
                cm += len(term.direct_delays) + len(term.conjugate_delays) - 1
        # Share powers within each delay; build |z|² = I² + Q², then recurrence.
        power_multiplies = sum(k-1 for k in envelope_powers.values())
        rm += 2*len(envelope_powers) + power_multiplies
        ra += len(envelope_powers)
        delay = max(delays)-min(0, min(delays)) if delays else 0
        return HardwareComplexity(n, cm, rm, max(0, n-1), ra, delay, 0, 0,
                                  2*self.word_bits*(n+delay), 1, 1.0, assumptions +
                                  ["Comparte potencias de envolvente por retardo; productos Volterra no comparten subexpresiones.",
                                   f"Adelanto GMP causalizable con latencia de {max(0, -min(delays)) if delays else 0} muestras."],
                                  {"envelope_power_multiplies": power_multiplies})


class LUTAssistedHardwareEstimator(DirectHardwareEstimator):
    def __init__(self, word_bits: int = 18, power_lut_entries: int = 256):
        super().__init__(word_bits)
        if power_lut_entries < 2:
            raise ValueError("power_lut_entries debe ser >= 2.")
        self.power_lut_entries = power_lut_entries

    def estimate(self, model: TrainedModel, config: ModelConfig) -> HardwareComplexity:
        base = super().estimate(model, config)
        replaced = base.auxiliary_operations.get("envelope_power_multiplies", 0)
        # Each higher power output at each delay is a separate logical table port.
        entries = replaced*self.power_lut_entries
        return replace(base, real_multipliers=base.real_multipliers-replaced,
                       num_luts=base.num_luts+replaced, lut_entries=base.lut_entries+entries,
                       memory_bits=base.memory_bits+entries*self.word_bits,
                       assumptions=base.assumptions + ["Variante conceptual: tablas de potencias de |z|²; no aplica fórmulas particulares Guan–Zhu.",
                                                      "Requiere rango/escala y cuantización; exactitud de tablas no simulada."])


class MultiplierSharingEstimator(DirectHardwareEstimator):
    def __init__(self, word_bits: int = 18, sharing_factor: int = 4):
        super().__init__(word_bits)
        if sharing_factor < 1:
            raise ValueError("sharing_factor debe ser positivo.")
        self.sharing_factor = sharing_factor

    def estimate(self, model: TrainedModel, config: ModelConfig) -> HardwareComplexity:
        base = super().estimate(model, config)
        # Conservative serial upper-bound schedule: one multiplication at a time.
        # Counts are dedicated lanes pooled by type, not a claim of an optimal scheduler.
        cm = ceil(base.complex_multipliers/self.sharing_factor)
        rm = ceil(base.real_multipliers/self.sharing_factor)
        cycles = max(1, base.complex_multipliers+base.real_multipliers)
        registers = 2*base.complex_multipliers+base.real_multipliers
        return replace(base, complex_multipliers=cm, real_multipliers=rm,
                       cycles_per_sample=cycles, clock_multiplier=float(cycles),
                       memory_bits=base.memory_bits+registers*self.word_bits,
                       assumptions=base.assumptions + ["Variante conceptual con bancos de multiplicadores y registros intermedios.",
                                                      "Ciclos: cota serial conservadora con operadores de un ciclo, no planificación óptima ni Fmax."])
