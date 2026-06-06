"""Heuristic strategy interfaces and simple defaults."""

from __future__ import annotations

from typing import Optional, Protocol

from .data import LpProblem


class StepSizeRule(Protocol):
    def initialize(self, problem: LpProblem) -> tuple[float, float]:
        ...

    def update(self, stats: "IterationStats") -> tuple[float, float]:
        ...


class RestartRule(Protocol):
    def should_restart(self, stats: "IterationStats") -> bool:
        ...


class ScalingRule(Protocol):
    def apply(self, problem: LpProblem) -> tuple[LpProblem, "ScalingState"]:
        ...


class StoppingRule(Protocol):
    def check(self, stats: "IterationStats") -> Optional["Termination"]:
        ...


class NoRestart:
    def should_restart(self, stats: "IterationStats") -> bool:
        return False


class FixedStepSize:
    def __init__(self, tau: float = 1.0, sigma: float = 1.0) -> None:
        self.tau = tau
        self.sigma = sigma

    def initialize(self, problem: LpProblem) -> tuple[float, float]:
        return self.tau, self.sigma

    def update(self, stats: "IterationStats") -> tuple[float, float]:
        return self.tau, self.sigma


class NoScaling:
    def apply(self, problem: LpProblem) -> tuple[LpProblem, "ScalingState"]:
        return problem, ScalingState()


class ScalingState:
    pass


class Termination:
    pass


class IterationStats:
    pass