"""Declared compute turn-down as a CONDITIONAL overlay on the dispatch (WP-SIZE-11, B8-03).

The two production entry points for an IC-9 declaration, one per consumer, both through the one
library implementation (`gridq.constraint_library.compute`) and the one model build
(`MilpDispatchSolver._solve_milp`):

- `solve_with_compute_turndown`: the sizer's question (A5). The floor solve is the solver's own,
  untouched; a SECOND solve of the same model carries the declared envelopes, and what the
  turn-down is worth is priced as a delta on it, net of the floor £ the overlay's battery gave up
  (the MICASC-3 method). D-NEW-SIZE11-2.
- `solve_window_with_compute`: live dispatch's question (B2). One receding-horizon window solved
  through `MilpDispatchSolver.solve_window_outcome` with the envelopes resolved at the moment the
  calls would be issued and the history of calls already made, so a call is never planned inside
  an envelope's notice or past a budget the history has used.

Both return a `ConditionalComputeOutcome`: firmness ``conditional``, never in the priced floor,
every call checked by the contract's `check_call`. With no declaration nothing is compiled and
nothing is solved here (D-NEW-SIZE11-6): the floor is the solver's ``solve`` exactly as before.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from gridq import constraint_library as _cl
from gridq.contracts.compute_turndown import ComputeTurndownDeclaration, PastCall
from gridq.dispatch.contract import DispatchProblem
from gridq.engine.errors import InvalidRequestError
from gridq.optimisation._assumptions import maybe, numeric
from gridq.optimisation.milp_solver import (
    MilpDispatchSolver,
    ObjectiveWeights,
    WindowOutcome,
    _floor_value,
    _UnpriceableDispatch,
)
from gridq.optimisation.schedule import HalfHourlyDispatchSchedule

__all__ = [
    "NO_ENVELOPE",
    "ComputeOverlayResult",
    "ComputeWindowResult",
    "solve_window_with_compute",
    "solve_with_compute_turndown",
]

#: The statement made, and the only thing done, when no envelope is declared.
NO_ENVELOPE = (
    "No compute turn-down envelope declared: no conditional compute resource is compiled and the "
    "dispatch is the floor solve, unchanged (WP-SIZE-11, D-NEW-SIZE11-6)."
)


@dataclass(frozen=True)
class ComputeOverlayResult:
    """The sizer's answer: the floor schedule, the overlay's, and the conditional outcome.

    ``outcome`` is ``None`` when no envelope is declared or a solve did not reach an optimum (the
    reason is on ``warnings``); it is never a zero standing in for a refusal.
    """

    outcome: _cl.compute.ConditionalComputeOutcome | None
    floor_schedule: HalfHourlyDispatchSchedule | None
    overlay_schedule: HalfHourlyDispatchSchedule | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ComputeWindowResult:
    """Live dispatch's answer for one window: the window outcome and the calls it plans.

    ``window`` is the overlay window's `WindowOutcome` (its schedule is the battery's plan with the
    calls in it). ``outcome`` is ``None`` when no envelope is declared or the window refused.
    """

    window: WindowOutcome
    outcome: _cl.compute.ConditionalComputeOutcome | None
    warnings: tuple[str, ...]


def _it_load(
    it_load_kw: pd.Series | Sequence[float] | np.ndarray, timestamps: pd.DatetimeIndex
) -> tuple[np.ndarray, list[str]]:
    """The IT load aligned to the solve's half-hours.

    A series is aligned by timestamp; a half-hour it does not cover is read as no IT load running,
    so no turn-down is offered there (the direction that lowers the value), and the count is
    stated. An array must already have one value per half-hour.
    """
    if isinstance(it_load_kw, pd.Series):
        aligned = it_load_kw.reindex(timestamps)
        missing = int(aligned.isna().sum())
        notes = []
        if missing:
            notes.append(
                f"IT load not supplied for {missing} of {len(timestamps)} half-hours: no compute "
                "turn-down is offered there (never assumed running)."
            )
        return aligned.fillna(0.0).to_numpy(dtype=float), notes
    values = np.asarray(it_load_kw, dtype=float).reshape(-1)
    if values.size != len(timestamps):
        raise InvalidRequestError(
            f"it_load_kw has {values.size} values for {len(timestamps)} half-hours"
        )
    return values, []


def _rte(problem: DispatchProblem) -> float:
    rte = numeric(maybe(problem.assumptions, "round_trip_efficiency", "ac_round_trip"))
    if rte is None or not 0 < rte <= 1:
        raise InvalidRequestError("round_trip_efficiency.ac_round_trip is missing or invalid")
    return float(rte)


def _idle(resource: _cl.compute.ComputeResource) -> _cl.compute.ComputeSolution:
    """The solution of a resource that offers nothing: no call anywhere."""
    n = len(resource.instants)
    zeros = np.zeros(n, dtype=float)
    return _cl.compute.ComputeSolution(
        turn_down_kw=tuple(zeros for _ in resource.envelopes),
        called=tuple(np.zeros(n, dtype=bool) for _ in resource.classes),
        avoided_price_gbp_per_kwh=zeros,
        rebound_price_gbp_per_kwh=zeros,
        segment_starts=np.array([0], dtype=np.intp),
    )


def solve_with_compute_turndown(
    solver: MilpDispatchSolver,
    problem: DispatchProblem,
    declaration: ComputeTurndownDeclaration | None,
    *,
    it_load_kw: pd.Series | Sequence[float] | np.ndarray,
    duration_min: int,
    meter_factor: float | Sequence[float] | np.ndarray = 1.0,
    history: Sequence[PastCall] = (),
) -> ComputeOverlayResult:
    """Price a declared compute turn-down on top of the floor dispatch (module docstring).

    The floor and the overlay are solved over the site's own half-hourly series, exactly as
    ``solver.solve`` reads it. Neither solve touches the solver's ``last_*`` state, so a caller's
    floor result is never replaced by an overlay's.
    """
    if declaration is None:
        return ComputeOverlayResult(None, None, None, (NO_ENVELOPE,))
    file_ref = problem.site_model.load_data.half_hourly_file_ref
    if not file_ref:
        raise InvalidRequestError("the compute overlay needs the site's half-hourly series")
    load, net_load, timestamps, interval_min, warnings = solver._load_series_signed(file_ref)
    rte = _rte(problem)
    it_load, it_notes = _it_load(it_load_kw, timestamps)
    warnings = [*warnings, *it_notes]
    resource = _cl.compute.compute_resource(
        declaration,
        timestamps,
        it_load,
        duration_min=duration_min,
        interval_min=interval_min,
        meter_factor=meter_factor,
        history=history,
    )
    try:
        floor, _b, floor_warn = solver._solve_milp(
            problem, load, timestamps, interval_min, rte, net_load_kw=net_load
        )
    except _UnpriceableDispatch as exc:
        return ComputeOverlayResult(None, None, None, (*warnings, str(exc)))
    if floor is None:
        return ComputeOverlayResult(None, None, None, (*warnings, *floor_warn))
    dt = interval_min / 60.0
    if not resource.offers_anything:
        outcome = _cl.compute.account_conditional_compute(
            resource,
            _idle(resource),
            dt=dt,
            annual_factor=float(floor.annual_factor),
            floor_value_gbp=_floor_value(floor),
            overlay_floor_value_gbp=_floor_value(floor),
        )
        note = "The declaration offers no turn-down over this horizon: nothing is compiled."
        return ComputeOverlayResult(outcome, floor, floor, (*warnings, note))
    try:
        overlay, _b, overlay_warn = solver._solve_milp(
            problem,
            load,
            timestamps,
            interval_min,
            rte,
            net_load_kw=net_load,
            compute_resource=resource,
        )
    except _UnpriceableDispatch as exc:
        return ComputeOverlayResult(None, floor, None, (*warnings, str(exc)))
    solution = solver._last_compute_solution
    if overlay is None or solution is None:
        return ComputeOverlayResult(None, floor, None, (*warnings, *overlay_warn))
    outcome = _cl.compute.account_conditional_compute(
        resource,
        solution,
        dt=dt,
        annual_factor=float(floor.annual_factor),
        floor_value_gbp=_floor_value(floor),
        overlay_floor_value_gbp=_floor_value(overlay),
    )
    return ComputeOverlayResult(outcome, floor, overlay, (*warnings, *overlay_warn))


def solve_window_with_compute(
    solver: MilpDispatchSolver,
    problem: DispatchProblem,
    declaration: ComputeTurndownDeclaration | None,
    *,
    load: np.ndarray,
    timestamps: pd.DatetimeIndex,
    interval_min: int,
    initial_soe_kwh: float,
    it_load_kw: pd.Series | Sequence[float] | np.ndarray,
    duration_min: int,
    issued_at: datetime,
    history: Sequence[PastCall] = (),
    weights: ObjectiveWeights | None = None,
    meter_factor: float | Sequence[float] | np.ndarray = 1.0,
) -> ComputeWindowResult:
    """Solve one receding-horizon window with the declared envelopes (live dispatch, B2).

    ``issued_at`` is now: a call is issued at it, so an envelope whose notice has not run by a
    half-hour offers nothing there. ``history`` is every call already made (the budgets, the rest
    and a call still in force read it through the contract). The window's floor is solved too, so
    the outcome is netted the same way as the sizer's.
    """
    rte = _rte(problem)
    if declaration is None:
        window = solver.solve_window_outcome(
            problem, load, timestamps, interval_min, rte, initial_soe_kwh, weights
        )
        return ComputeWindowResult(window, None, (NO_ENVELOPE,))
    it_load, notes = _it_load(it_load_kw, timestamps)
    resource = _cl.compute.compute_resource(
        declaration,
        timestamps,
        it_load,
        duration_min=duration_min,
        interval_min=interval_min,
        meter_factor=meter_factor,
        issued_at=issued_at,
        history=history,
    )
    floor = solver.solve_window_outcome(
        problem, load, timestamps, interval_min, rte, initial_soe_kwh, weights
    )
    if floor.schedule is None:
        return ComputeWindowResult(floor, None, (*notes, *floor.reasons))
    floor_value = _floor_value(floor.schedule)
    dt = interval_min / 60.0
    if not resource.offers_anything:
        outcome = _cl.compute.account_conditional_compute(
            resource,
            _idle(resource),
            dt=dt,
            annual_factor=float(floor.schedule.annual_factor),
            floor_value_gbp=floor_value,
            overlay_floor_value_gbp=floor_value,
        )
        return ComputeWindowResult(floor, outcome, tuple(notes))
    window = solver.solve_window_outcome(
        problem,
        load,
        timestamps,
        interval_min,
        rte,
        initial_soe_kwh,
        weights,
        compute_resource=resource,
    )
    solution = solver._last_compute_solution
    if window.schedule is None or solution is None:
        return ComputeWindowResult(window, None, (*notes, *window.reasons))
    outcome = _cl.compute.account_conditional_compute(
        resource,
        solution,
        dt=dt,
        annual_factor=float(window.schedule.annual_factor),
        floor_value_gbp=floor_value,
        overlay_floor_value_gbp=_floor_value(window.schedule),
    )
    return ComputeWindowResult(window, outcome, tuple(notes))
