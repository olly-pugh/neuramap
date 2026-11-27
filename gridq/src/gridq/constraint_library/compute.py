"""Conditional compute resources: declared IC-9 turn-down envelopes in the dispatch (WP-SIZE-11).

Releases ``conditional_compute_resource`` (spec B8-03: envelopes enter the sizer A5 and live
dispatch B2 "as conditional resources only"). A GPU operator's IC-9 declaration
(`gridq.contracts.compute_turndown`) becomes turn-down variables, call rows and an objective term
on the same linopy build the annual MILP and every MPC window solve, so sizing and live dispatch
take it from one place. Decisions D-NEW-SIZE11-1 to -6 (``docs/decisions.md``).

**Conditional only (D-NEW-SIZE11-2).** Every £ this resource earns is classified
`Firmness.CONDITIONAL` (`FIRMNESS`): an offer by a third party's workloads, never in the priced
floor. Three things make that structural rather than a label:

1. the resource is compiled only on a conditional overlay solve; the priced-floor solve has no
   argument that carries it, so the floor dispatch is the same model whether or not an envelope is
   declared;
2. `ConditionalComputeOutcome` fixes ``firmness`` to ``conditional`` and ``in_priced_floor`` to
   ``False`` (its constructor refuses anything else), and it is not an IC-12 member, so no reader
   of the priced floor can sum it;
3. the floor £ the overlay's battery gives up is measured on both schedules and netted off, so
   the conditional figure never restates a floor pound (the MICASC-3 method).

**One reader per question (D-NEW-SIZE11-1).** The contract's own readers supply every number:
`offered_at` and `stack_at` (merit order, one envelope per exclusion group, the concurrent cap and
the retained IT floor) give each envelope's layer of turn-down at each half-hour, at a call of
``duration_min`` and the IT load then running (floored to a whole kW, the contract's integers);
`call_budget_used` gives what a history of past calls has used; `check_call` checks every planned
call; `call_cost_pence` and `availability_cost_pence` price it. Nothing is summed or ranked here
that the contract does not sum or rank.

Constraints (catalogue family ``conditional_resource``; D-NEW-SIZE11-3):

- ``compute_turn_down_limit``: ``u_e,t <= layer_e,t x call_c,t`` for envelope ``e`` of SLA class
  ``c``, with ``call_c,t`` binary;
- ``compute_call_shape``: a call holds one kW per envelope for its whole run, runs at most
  ``duration_min`` (the duration the layers were read at), and its start and end are counted;
- ``compute_call_budget``: per class, starts in every rolling event window, called minutes in
  every rolling duration window (extended forward by one call, so every call starting in a window
  is counted whole), the minimum rest after a call, and what the history has already used;
- ``compute_meter_displacement``: ``discharge_t + m_t x sum_e u_e,t <= load_t``: one kW of site
  load is displaced once, by the battery or by compute, never by both.

Then every planned call is round-tripped through the contract's `check_call` and only the calls it
accepts are credited, at the contract's whole kW (D-NEW-SIZE11-4). The economics take the safe side
where the declaration is silent (D-NEW-SIZE11-5); every such choice is a note on the outcome.

**No envelope, no change (D-NEW-SIZE11-6).** With no declaration there is no resource, and a
declaration that offers nothing over the horizon compiles nothing (`compile_conditional_compute`
returns ``None`` and the model is untouched).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from typing import Any, Final, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from gridq.contracts._base import NotApplicable, Unknown
from gridq.contracts.compute_turndown import (
    CLOCK,
    MAX_HOLD_MIN,
    RESOLUTION_MIN,
    AvailabilityPrice,
    CallCheck,
    ComputeTurndownDeclaration,
    DeclarationRef,
    DurationBudget,
    EventBudget,
    EventFee,
    MinimumRest,
    NoReturn,
    Offered,
    PastCall,
    PriceOffer,
    ProposedCall,
    ReboundSize,
    ReturnsLater,
    UtilisationPrice,
    availability_cost_pence,
    call_budget_used,
    call_cost_pence,
    check_call,
    declaration_ref,
    offered_at,
    price_basis_label,
    stack_at,
)
from gridq.schemas import Firmness

__all__ = [
    "FIRMNESS",
    "RESOURCE_ID",
    "ClassTerms",
    "CompiledCompute",
    "ComputeResource",
    "ComputeSolution",
    "ConditionalComputeOutcome",
    "EnvelopeTerms",
    "PlannedCall",
    "account_conditional_compute",
    "compile_conditional_compute",
    "compute_resource",
]

#: The roadmap release this module is (WP-SIZE-11).
RESOURCE_ID: Final = "conditional_compute_resource"
#: The one classification any compute turn-down £ ever carries (B8-03).
FIRMNESS: Final = Firmness.CONDITIONAL

_TZ: Final = ZoneInfo(CLOCK)
_HALF_HOUR: Final = timedelta(minutes=RESOLUTION_MIN)
_INTERVALS_PER_DAY: Final = 1440 // RESOLUTION_MIN
#: A solved kW within this of the next whole kW is that kW (HiGHS returns 99.9999999 for 100).
_KW_TOL: Final = 1e-6


# ============================================================================ the resource


@dataclass(frozen=True)
class EnvelopeTerms:
    """One envelope's resolved numbers over the horizon (read from the contract, never invented).

    ``layer_kw[t]`` is the envelope's layer of `stack_at` at half-hour ``t`` (0 where it is not
    offered, outside the validity, inside the notice of a live window, or blocked by the history).
    ``utilisation_gbp_per_kwh`` / ``event_fee_gbp`` are ``None`` when the operator has not stated
    them (unknown is never read as zero; the objective then carries no cost for it and the outcome
    says the value is gross of it). ``recovery`` is the energy that comes back per kWh turned down
    (0 when the class's energy does not return); ``return_intervals`` the half-hours it may come
    back within (``None``: by the end of the horizon).
    """

    envelope_id: str
    class_id: str
    class_index: int
    layer_kw: np.ndarray
    price: PriceOffer | Unknown
    utilisation_gbp_per_kwh: float | None
    event_fee_gbp: float | None
    ramp_s: int
    min_notice_s: int
    recovery: float
    return_intervals: int | None


@dataclass(frozen=True)
class ClassTerms:
    """One SLA class's call limits, in half-hours, net of what the history has already used.

    ``None`` is a limit the class does not have (``not_applicable``) or does not know
    (``unknown``); an unknown one is named in ``unchecked`` and is never enforced or assumed (the
    contract's rule, `CallLimits`).
    """

    class_id: str
    events: tuple[int, int] | None  # (events still allowed, window in half-hours)
    minutes: tuple[int, int] | None  # (whole half-hours still allowed, window in half-hours)
    rest_intervals: int | None
    blocked: np.ndarray  # bool: no call of this class may be in force here
    unchecked: tuple[str, ...]


@dataclass(frozen=True)
class ComputeResource:
    """A declaration resolved onto one solve's half-hours, for `compile_conditional_compute`.

    Built only by `compute_resource`. ``instants`` are the half-hours as aware UTC instants (a
    naive timestamp is the GB clock under PEP 495 ``fold=0``, the contract's `declared_at_local`
    rule); ``it_load_kw`` the IT load running, floored to a whole kW; ``meter_factor`` the meter kW
    one IT kW of turn-down saves. ``issued_at`` is set on a live window (calls are then issued
    then, and a call inside an envelope's notice is not offered); ``None`` is a plan, whose calls
    are issued each at its own notice ahead of its start.
    """

    declaration: ComputeTurndownDeclaration
    reference: DeclarationRef
    duration_min: int
    run_intervals: int
    instants: tuple[datetime, ...]
    it_load_kw: np.ndarray
    meter_factor: np.ndarray
    envelopes: tuple[EnvelopeTerms, ...]
    classes: tuple[ClassTerms, ...]
    issued_at: datetime | None
    history: tuple[PastCall, ...]
    notes: tuple[str, ...]

    @property
    def offers_anything(self) -> bool:
        """Whether any envelope has a positive layer at any half-hour of the horizon."""
        return any(bool(np.any(e.layer_kw > 0.0)) for e in self.envelopes)


def _instant(ts: pd.Timestamp) -> datetime:
    t = ts.to_pydatetime()
    if t.tzinfo is None or t.utcoffset() is None:
        return t.replace(tzinfo=_TZ, fold=0).astimezone(UTC)
    return t.astimezone(UTC)


def _part(
    part: AvailabilityPrice | UtilisationPrice | EventFee | NotApplicable | Unknown,
) -> int | None:
    """The stated amount, 0 when the component is not charged, ``None`` when it is unknown."""
    if isinstance(part, Unknown):
        return None
    if isinstance(part, NotApplicable):
        return 0
    if isinstance(part, AvailabilityPrice):
        return part.pence_per_mw_per_hour
    if isinstance(part, UtilisationPrice):
        return part.pence_per_mwh
    return part.pence


def compute_resource(
    declaration: ComputeTurndownDeclaration,
    timestamps: pd.DatetimeIndex,
    it_load_kw: Sequence[float] | np.ndarray,
    *,
    duration_min: int,
    interval_min: int = RESOLUTION_MIN,
    meter_factor: float | Sequence[float] | np.ndarray = 1.0,
    issued_at: datetime | None = None,
    history: Sequence[PastCall] = (),
) -> ComputeResource:
    """Resolve ``declaration`` onto a solve's half-hours (module docstring; D-NEW-SIZE11-1, -3).

    ``it_load_kw`` is the IT load running at each half-hour, aligned with ``timestamps`` (the
    caller's: the load model's expected series, or metering). ``duration_min`` is the call the
    layers are read at (D127: the consumer states it); it must be whole half-hours, because a
    call is credited over whole settlement periods. ``meter_factor`` defaults to 1.0, the IT kW
    alone, which never claims a UPS, distribution or cooling saving the caller has not evidenced.
    """
    if interval_min != RESOLUTION_MIN:
        raise ValueError(
            f"a compute resource is resolved on the IC-9 {RESOLUTION_MIN}-minute grid; this solve "
            f"runs at {interval_min} minutes"
        )
    if isinstance(duration_min, bool) or not isinstance(duration_min, int):
        raise ValueError("duration_min must be an integer number of minutes")
    if not (RESOLUTION_MIN <= duration_min <= MAX_HOLD_MIN) or duration_min % RESOLUTION_MIN:
        raise ValueError(
            f"duration_min must be whole half-hours from {RESOLUTION_MIN} to {MAX_HOLD_MIN}: a "
            "call is credited over whole settlement periods"
        )
    n = len(timestamps)
    load = np.asarray(it_load_kw, dtype=float).reshape(-1)
    if load.size != n:
        raise ValueError(f"it_load_kw has {load.size} values for {n} half-hours")
    if n == 0 or not np.all(np.isfinite(load)) or bool(np.any(load < 0.0)):
        raise ValueError("it_load_kw must be finite, non-negative and non-empty")
    factor = np.broadcast_to(np.asarray(meter_factor, dtype=float), (n,)).copy()
    if not np.all(np.isfinite(factor)) or bool(np.any(factor <= 0.0)):
        raise ValueError("meter_factor must be finite and positive")
    if issued_at is not None and (issued_at.tzinfo is None or issued_at.utcoffset() is None):
        raise ValueError("issued_at must be timezone-aware")
    whole_kw = np.floor(np.round(load, 6)).astype(np.int64)
    instants = tuple(_instant(ts) for ts in timestamps)
    for when in instants:
        if when.second or when.microsecond or when.minute % RESOLUTION_MIN:
            raise ValueError(f"{when.isoformat()} is not on the {RESOLUTION_MIN}-minute grid")

    d = declaration
    merit = list(d.stacking.merit_order)
    by_id = {e.envelope_id: e for e in d.envelopes}
    layers = np.zeros((len(merit), n), dtype=float)
    cache: dict[tuple[str, int], tuple[int, ...]] = {}
    for t, instant in enumerate(instants):
        if not (d.valid_from <= instant < d.valid_until):
            continue
        state = offered_at(d, instant).state
        if not isinstance(state, Offered):
            continue
        key = (state.model_dump_json(), int(whole_kw[t]))
        hit = cache.get(key)
        if hit is None:
            stack = stack_at(d, instant, duration_min, int(whole_kw[t]))
            got = {x.envelope_id: x.kw for x in stack.layers}
            hit = tuple(got.get(i, 0) for i in merit)
            cache[key] = hit
        layers[:, t] = hit

    notes: list[str] = [
        f"CONDITIONAL (B8-03): every £ of IC-9 declaration {d.declaration_id} "
        f"({d.content_hash()}) is an offer by a third party's workloads, held outside the priced "
        "floor, P50, P90 and the DSCR.",
        f"Layers read by the contract's stack_at at a call of {duration_min} min and the IT load "
        "running at each half-hour, floored to a whole kW.",
    ]
    if d.record_class == "synthetic_fixture":
        notes.append(
            "SYNTHETIC FIXTURE: the declaration is schema-test data, not an operator's offer."
        )
    if np.allclose(factor, 1.0):
        notes.append(
            "Meter factor 1.0: the meter saving is the IT kW alone (no UPS, distribution or "
            "cooling saving is claimed without the caller's evidence)."
        )

    # The history's use (the contract's call_budget_used) and its calls still in force.
    start = instants[0]
    hist = tuple(history)
    ongoing_end = max(
        (
            c.start + timedelta(minutes=c.duration_min)
            for c in hist
            if c.start <= start < c.start + timedelta(minutes=c.duration_min)
        ),
        default=None,
    )
    in_force = np.array([ongoing_end is not None and t < ongoing_end for t in instants])
    if bool(in_force.any()):
        notes.append(
            f"A past call is in force until {ongoing_end.isoformat() if ongoing_end else ''}: "
            "no new call starts before it ends (its kW is already committed)."
        )

    classes: list[ClassTerms] = []
    for c in d.sla_classes:
        limits = c.call_limits
        where = f"sla_classes[{c.class_id}].call_limits"
        unchecked: list[str] = []
        used = call_budget_used(d, c.class_id, hist, as_of=start)
        events: tuple[int, int] | None = None
        if isinstance(limits.max_events, EventBudget):
            prior = used.events_in_window if isinstance(used.events_in_window, int) else 0
            events = (
                max(0, limits.max_events.max_events - prior),
                limits.max_events.window_days * _INTERVALS_PER_DAY,
            )
        elif isinstance(limits.max_events, Unknown):
            unchecked.append(f"{where}.max_events")
        minutes: tuple[int, int] | None = None
        if isinstance(limits.max_cumulative, DurationBudget):
            prior_min = used.minutes_in_window if isinstance(used.minutes_in_window, int) else 0
            left = max(0, limits.max_cumulative.max_minutes - prior_min)
            minutes = (
                left // RESOLUTION_MIN,
                limits.max_cumulative.window_days * _INTERVALS_PER_DAY,
            )
        elif isinstance(limits.max_cumulative, Unknown):
            unchecked.append(f"{where}.max_cumulative")
        rest: int | None = None
        blocked = in_force.copy()
        if isinstance(limits.min_rest, MinimumRest):
            rest = math.ceil(limits.min_rest.minutes / RESOLUTION_MIN)
            if isinstance(used.last_end, datetime):
                until = used.last_end + timedelta(minutes=limits.min_rest.minutes)
                blocked |= np.array([t < until for t in instants])
        elif isinstance(limits.min_rest, Unknown):
            unchecked.append(f"{where}.min_rest")
        if unchecked:
            notes.append(
                f"NOT ENFORCED, NOT ASSUMED (the contract's rule for an unknown limit): "
                f"{', '.join(unchecked)}. The value is an upper bound in that respect."
            )
        classes.append(
            ClassTerms(
                class_id=c.class_id,
                events=events,
                minutes=minutes,
                rest_intervals=rest,
                blocked=blocked,
                unchecked=tuple(unchecked),
            )
        )
    class_index = {c.class_id: j for j, c in enumerate(classes)}
    sla = {c.class_id: c for c in d.sla_classes}

    envelopes: list[EnvelopeTerms] = []
    for i, envelope_id in enumerate(merit):
        e = by_id[envelope_id]
        layer = layers[i].copy()
        layer[classes[class_index[e.sla_class_id]].blocked] = 0.0
        if issued_at is not None:
            notice = timedelta(seconds=e.response.min_notice_s)
            layer[np.array([t - issued_at < notice for t in instants])] = 0.0
        price = e.price
        util = fee = None
        if isinstance(price, PriceOffer):
            u_p, f_p = _part(price.utilisation), _part(price.per_event)
            util = None if u_p is None else u_p / 100.0 / 1000.0  # pence/MWh -> GBP/kWh
            fee = None if f_p is None else f_p / 100.0
        ret = sla[e.sla_class_id].energy_return
        recovery = 1.0
        window: int | None = None
        if isinstance(ret, NoReturn):
            recovery = 0.0
        else:
            if isinstance(e.rebound, ReboundSize) and isinstance(e.rebound.recovery_permille, int):
                recovery = e.rebound.recovery_permille / 1000.0
            else:
                notes.append(
                    f"Envelope {envelope_id}: the energy that comes back is not stated, so it is "
                    "taken to come back in full (deferred work is the same work later; the "
                    "direction that lowers the value)."
                )
            if isinstance(ret, ReturnsLater) and isinstance(ret.return_window_min, int):
                window = math.ceil(ret.return_window_min / RESOLUTION_MIN)
            else:
                notes.append(
                    f"Envelope {envelope_id}: no return window is stated, so the rebound is "
                    "priced at the dearest half-hour left in the horizon."
                )
        if util is None:
            notes.append(
                f"Envelope {envelope_id}: the utilisation price is unknown, so its value is GROSS "
                "of the operator's call cost (never read as zero)."
            )
        envelopes.append(
            EnvelopeTerms(
                envelope_id=envelope_id,
                class_id=e.sla_class_id,
                class_index=class_index[e.sla_class_id],
                layer_kw=layer,
                price=price,
                utilisation_gbp_per_kwh=util,
                event_fee_gbp=fee,
                ramp_s=e.response.ramp_s,
                min_notice_s=e.response.min_notice_s,
                recovery=recovery,
                return_intervals=window,
            )
        )
    return ComputeResource(
        declaration=d,
        reference=declaration_ref(d),
        duration_min=duration_min,
        run_intervals=duration_min // RESOLUTION_MIN,
        instants=instants,
        it_load_kw=whole_kw,
        meter_factor=factor,
        envelopes=tuple(envelopes),
        classes=tuple(classes),
        issued_at=issued_at,
        history=hist,
        notes=tuple(notes),
    )


# ============================================================================ the rows


def _worst_ahead(values: np.ndarray, window: int | None) -> np.ndarray:
    """``max(values[t : t + window + 1])`` per ``t`` (to the end of the series when ``None``)."""
    rev = pd.Series(values[::-1])
    if window is None:
        out = rev.cummax()
    else:
        out = rev.rolling(window + 1, min_periods=1).max()
    return np.asarray(out.to_numpy()[::-1], dtype=float)


@dataclass(frozen=True)
class ComputeSolution:
    """The solved turn-down per envelope (kW, zeros where not compiled) and calls per class."""

    turn_down_kw: tuple[np.ndarray, ...]
    called: tuple[np.ndarray, ...]
    avoided_price_gbp_per_kwh: np.ndarray
    rebound_price_gbp_per_kwh: np.ndarray
    segment_starts: np.ndarray


@dataclass
class CompiledCompute:
    """What `compile_conditional_compute` added: variable handles and the objective term."""

    resource: ComputeResource
    turn_down: list[Any]
    call: list[Any]
    objective: Any
    avoided_price_gbp_per_kwh: np.ndarray
    rebound_price_gbp_per_kwh: np.ndarray
    segment_starts: np.ndarray

    def read(self) -> ComputeSolution:
        """The solution, after an optimal solve (rounded like the battery's flows)."""
        n = len(self.resource.instants)

        def values(var: Any) -> np.ndarray:
            if var is None:
                return np.zeros(n, dtype=float)
            return np.clip(np.round(np.array(var.solution.values, dtype=float), 6), 0.0, None)

        return ComputeSolution(
            turn_down_kw=tuple(values(v) for v in self.turn_down),
            called=tuple(values(v) > 0.5 for v in self.call),
            avoided_price_gbp_per_kwh=self.avoided_price_gbp_per_kwh,
            rebound_price_gbp_per_kwh=self.rebound_price_gbp_per_kwh,
            segment_starts=self.segment_starts,
        )


def _at(var: Any, idx: np.ndarray, coords: Any) -> Any:
    """``var`` read at positions ``idx``, relabelled onto ``coords`` (so it aligns with ``var``)."""
    import xarray as xr

    picked = var.isel(t=xr.DataArray(np.asarray(idx, dtype=np.intp), dims=["t"]))
    return picked.assign_coords(t=coords)


def _window_rows(
    model: Any, cum: Any, lo: int, hi: int, rhs: float, name: str, n: int, coords: Any
) -> None:
    """``sum(x over (t - lo, t + hi]) <= rhs`` for every ``t``, on a cumulative ``cum`` of ``x``.

    Positional over the whole series: a window that spans a gap in the metering counts more real
    time than its width, so it is the stricter reading (never the looser one).
    """
    import xarray as xr

    t = np.arange(n)
    upper = _at(cum, np.minimum(t + hi, n - 1), coords)
    lower_idx = t - lo
    has_lower = lower_idx >= 0
    if bool(has_lower.any()):
        model.add_constraints(
            upper - _at(cum, np.maximum(lower_idx, 0), coords) <= rhs,
            name=f"{name}",
            mask=xr.DataArray(has_lower, coords=[coords]),
        )
    if bool((~has_lower).any()):
        model.add_constraints(
            1.0 * upper <= rhs,
            name=f"{name}_head",
            mask=xr.DataArray(~has_lower, coords=[coords]),
        )


def _cumulative(model: Any, x: Any, name: str, n: int, prev: np.ndarray, coords: Any) -> Any:
    """A variable ``C`` with ``C_t = C_(t-1) + x_t`` and ``C_0 = x_0`` (positional)."""
    import xarray as xr

    cum = model.add_variables(lower=0.0, coords=[coords], name=name)
    first = np.zeros(n, dtype=bool)
    first[0] = True
    model.add_constraints(
        cum - _at(cum, prev, coords) - x == 0,
        name=f"{name}_step",
        mask=xr.DataArray(~first, coords=[coords]),
    )
    model.add_constraints(
        cum - x == 0, name=f"{name}_first", mask=xr.DataArray(first, coords=[coords])
    )
    return cum


def compile_conditional_compute(
    model: Any,  # linopy.Model (untyped upstream)
    resource: ComputeResource,
    *,
    discharge: Any,
    load_kw: np.ndarray,
    avoided_price_gbp_per_kwh: np.ndarray,
    dt: float,
    coords: Any,
    segment_starts: np.ndarray,
    rebound_price_gbp_per_kwh: np.ndarray | None = None,
) -> CompiledCompute | None:
    """Add the conditional compute rows and return the objective term (module docstring).

    ``avoided_price_gbp_per_kwh`` is the delivered price of a kWh the site does not import (the
    price a discharged kWh earns in the same build); ``rebound_price_gbp_per_kwh`` the price of a
    kWh imported again when the deferred work returns (the build's charge-side price where it
    differs, the dearer reading; ``None``: the avoided price); ``segment_starts`` the first index
    of each contiguous metered segment (no call runs across a gap). Returns ``None``, having added
    nothing, when no envelope offers anything over the horizon (D-NEW-SIZE11-6).
    """
    import xarray as xr

    if not resource.offers_anything:
        return None
    n = len(resource.instants)
    if load_kw.size != n or avoided_price_gbp_per_kwh.size != n:
        raise ValueError("the compute resource and the solve disagree on the horizon")
    seg_start = np.zeros(n, dtype=bool)
    seg_start[np.asarray(segment_starts, dtype=np.intp)] = True
    seg_start[0] = True
    seg_end = np.zeros(n, dtype=bool)
    seg_end[-1] = True
    seg_end[:-1] = seg_start[1:]
    prev_idx = np.arange(n) - 1
    prev_idx[0] = 0
    next_idx = np.minimum(np.arange(n) + 1, n - 1)
    prev = prev_idx
    cont = xr.DataArray(~seg_start, coords=[coords])
    m = resource.meter_factor
    price = np.asarray(avoided_price_gbp_per_kwh, dtype=float)
    back = (
        price
        if rebound_price_gbp_per_kwh is None
        else np.asarray(rebound_price_gbp_per_kwh, dtype=float)
    )
    if back.size != n:
        raise ValueError("the compute resource and the solve disagree on the horizon")
    value_kw = price * m  # GBP per kWh of IT turn-down at the meter
    back_kw = back * m
    R = resource.run_intervals

    turn_down: list[Any] = [None] * len(resource.envelopes)
    call: list[Any] = [None] * len(resource.classes)
    displaced: Any = None
    objective: Any = None
    for j, cls in enumerate(resource.classes):
        members = [
            (i, e)
            for i, e in enumerate(resource.envelopes)
            if e.class_index == j and bool(np.any(e.layer_kw > 0.0))
        ]
        if not members:
            continue
        c = model.add_variables(binary=True, coords=[coords], name=f"compute_call_{j}")
        call[j] = c
        # compute_call_shape: starts (a segment start is always a start: no run crosses a gap).
        s = model.add_variables(lower=0.0, upper=1.0, coords=[coords], name=f"compute_start_{j}")
        model.add_constraints(
            s - c + _at(c, prev, coords) >= 0, name=f"compute_start_{j}", mask=cont
        )
        model.add_constraints(
            s - c >= 0,
            name=f"compute_start_first_{j}",
            mask=xr.DataArray(seg_start, coords=[coords]),
        )
        calls_cum = _cumulative(model, c, f"compute_calls_cum_{j}", n, prev, coords)
        # A run of R + 1 called half-hours would hold longer than the layers were read at.
        _window_rows(model, calls_cum, R + 1, 0, float(R), f"compute_run_{j}", n, coords)
        starts_cum: Any = None
        if cls.events is not None or cls.rest_intervals is not None:
            starts_cum = _cumulative(model, s, f"compute_starts_cum_{j}", n, prev, coords)
        # compute_call_budget.
        if cls.events is not None:
            allowed, width = cls.events
            _window_rows(
                model, starts_cum, width, 0, float(allowed), f"compute_events_{j}", n, coords
            )
        if cls.minutes is not None:
            allowed_hh, width = cls.minutes
            _window_rows(
                model, calls_cum, width, R - 1, float(allowed_hh), f"compute_minutes_{j}", n, coords
            )
        if cls.rest_intervals is not None:
            rest = cls.rest_intervals
            e_var = model.add_variables(
                lower=0.0, upper=1.0, coords=[coords], name=f"compute_end_{j}"
            )
            model.add_constraints(
                e_var - c + _at(c, next_idx, coords) >= 0,
                name=f"compute_end_{j}",
                mask=xr.DataArray(~seg_end, coords=[coords]),
            )
            model.add_constraints(
                e_var - c >= 0,
                name=f"compute_end_last_{j}",
                mask=xr.DataArray(seg_end, coords=[coords]),
            )
            ahead = _at(starts_cum, np.minimum(np.arange(n) + rest, n - 1), coords)
            model.add_constraints(
                ahead - starts_cum + rest * e_var <= float(rest),
                name=f"compute_rest_{j}",
            )
        if bool(cls.blocked.any()):
            model.add_constraints(
                1.0 * c <= 0.0,
                name=f"compute_blocked_{j}",
                mask=xr.DataArray(cls.blocked, coords=[coords]),
            )
        start_cost = np.zeros(n, dtype=float)
        for i, e in members:
            layer = e.layer_kw
            big_m = float(layer.max())
            u = model.add_variables(
                lower=0.0,
                upper=xr.DataArray(layer, coords=[coords]),
                name=f"compute_turn_down_{i}",
            )
            turn_down[i] = u
            model.add_constraints(
                u - xr.DataArray(layer, coords=[coords]) * c <= 0,
                name=f"compute_turn_down_limit_{i}",
            )
            # One kW per envelope for the whole run (|u_t - u_t-1| <= M (2 - c_t - c_t-1)).
            model.add_constraints(
                u - _at(u, prev, coords) + big_m * c + big_m * _at(c, prev, coords) <= 2.0 * big_m,
                name=f"compute_call_level_up_{i}",
                mask=cont,
            )
            model.add_constraints(
                _at(u, prev, coords) - u + big_m * c + big_m * _at(c, prev, coords) <= 2.0 * big_m,
                name=f"compute_call_level_down_{i}",
                mask=cont,
            )
            # The return follows the run's end, at most R - 1 half-hours after t: priced at the
            # dearest half-hour of [t, t + R - 1 + window], a superset (never the cheaper reading).
            reach = None if e.return_intervals is None else e.return_intervals + R - 1
            rebound = e.recovery * _worst_ahead(back_kw, reach)
            util = e.utilisation_gbp_per_kwh or 0.0
            coef = dt * (value_kw - util - rebound)
            term = (u * xr.DataArray(coef, coords=[coords])).sum()
            objective = term if objective is None else objective + term
            scaled = u * xr.DataArray(m, coords=[coords])
            displaced = scaled if displaced is None else displaced + scaled
            # A start costs the event fee and, at most, the ramp of the whole layer.
            start_cost += (e.event_fee_gbp or 0.0) + value_kw * layer * (e.ramp_s / 3600.0)
        objective = objective - (s * xr.DataArray(start_cost, coords=[coords])).sum()
    # compute_meter_displacement: one kW of site load is displaced once.
    model.add_constraints(
        discharge + displaced <= xr.DataArray(np.asarray(load_kw, dtype=float), coords=[coords]),
        name="compute_meter_displacement",
    )
    return CompiledCompute(
        resource=resource,
        turn_down=turn_down,
        call=call,
        objective=objective,
        avoided_price_gbp_per_kwh=price,
        rebound_price_gbp_per_kwh=back,
        segment_starts=np.flatnonzero(seg_start),
    )


# ============================================================================ the account


@dataclass(frozen=True)
class PlannedCall:
    """One call the solve planned, as the contract's `ProposedCall`, and the contract's verdict."""

    class_id: str
    call: ProposedCall
    envelope_kw: tuple[tuple[str, int], ...]
    first_interval: int
    intervals: int
    check: CallCheck

    @property
    def credited(self) -> bool:
        return self.check.ok


@dataclass(frozen=True)
class ConditionalComputeOutcome:
    """What a declared compute turn-down is worth on top of the floor, CONDITIONAL only.

    £ per year (horizon £ x the schedule's annual factor). ``net_value_gbp`` is avoided import,
    less the ramp, the operator's utilisation and event fees, the rebound, the availability fee and
    the floor £ the overlay's battery gave up; it may be negative (the offer then costs more than
    it saves, which is a result). ``cost_unknown`` names each envelope whose price has an unknown
    component: the value is then gross of that cost.
    """

    declaration: DeclarationRef
    duration_min: int
    planned_calls: tuple[PlannedCall, ...]
    avoided_import_gbp: float
    ramp_loss_gbp: float
    call_cost_gbp: float
    rebound_cost_gbp: float
    availability_cost_gbp: float
    sacrificed_floor_gbp: float
    net_value_gbp: float
    cost_unknown: tuple[str, ...]
    price_basis: tuple[tuple[str, str], ...]
    notes: tuple[str, ...]
    resource_id: Literal["conditional_compute_resource"] = RESOURCE_ID
    firmness: Literal[Firmness.CONDITIONAL] = FIRMNESS
    in_priced_floor: Literal[False] = False
    unchecked: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.firmness is not Firmness.CONDITIONAL or self.in_priced_floor is not False:
            raise ValueError(
                "a compute turn-down is a CONDITIONAL resource (B8-03): it is never firm or "
                "contracted and never in the priced floor"
            )
        if self.resource_id != RESOURCE_ID:
            raise ValueError(f"resource_id must be {RESOURCE_ID}")

    @property
    def credited_calls(self) -> tuple[PlannedCall, ...]:
        return tuple(p for p in self.planned_calls if p.credited)


def _runs(called: np.ndarray, seg_start: np.ndarray) -> list[tuple[int, int]]:
    """Maximal ``[first, last]`` runs of called half-hours that do not cross a segment start."""
    runs: list[tuple[int, int]] = []
    first: int | None = None
    for t, on in enumerate(called):
        if first is not None and (not on or seg_start[t]):
            runs.append((first, t - 1))
            first = None
        if on and first is None:
            first = t
    if first is not None:
        runs.append((first, len(called) - 1))
    return runs


def _pounds(pence: Fraction) -> float:
    return float(pence / 100)


def account_conditional_compute(
    resource: ComputeResource,
    solution: ComputeSolution,
    *,
    dt: float,
    annual_factor: float,
    floor_value_gbp: float,
    overlay_floor_value_gbp: float,
) -> ConditionalComputeOutcome:
    """Turn a solve's calls into contract-checked calls and conditional £ (D-NEW-SIZE11-4).

    Every run of called half-hours becomes a `ProposedCall` at the solve's whole kW per envelope,
    is checked by `check_call` against the history and the calls accepted before it (in start
    order), and only an accepted call is credited. ``floor_value_gbp`` and
    ``overlay_floor_value_gbp`` are the priced-floor £/yr of the floor solve's schedule and of the
    overlay's; the overlay can only give floor £ up, and what it gives up is netted off.
    """
    d = resource.declaration
    n = len(resource.instants)
    value_kw = solution.avoided_price_gbp_per_kwh * resource.meter_factor
    back_kw = solution.rebound_price_gbp_per_kwh * resource.meter_factor
    seg_start = np.zeros(n, dtype=bool)
    seg_start[np.asarray(solution.segment_starts, dtype=np.intp)] = True
    seg_start[0] = True
    by_id = {e.envelope_id: e for e in d.envelopes}

    drafts: list[
        tuple[datetime, int, str, ProposedCall, tuple[tuple[str, int], ...], int, int]
    ] = []
    notes = list(resource.notes)
    for j, cls in enumerate(resource.classes):
        members = [(i, e) for i, e in enumerate(resource.envelopes) if e.class_index == j]
        for first, last in _runs(solution.called[j], seg_start):
            kw = tuple(
                (e.envelope_id, int(math.floor(float(solution.turn_down_kw[i][first]) + _KW_TOL)))
                for i, e in members
            )
            named = tuple(sorted((eid, k) for eid, k in kw if k >= 1))
            if not named:
                continue
            start = resource.instants[first]
            if resource.issued_at is not None:
                issued = resource.issued_at
            else:
                notice = max(by_id[eid].response.min_notice_s for eid, _ in named)
                issued = start - timedelta(seconds=notice)
            proposed = ProposedCall(
                kind="proposed_call",
                envelope_ids=tuple(eid for eid, _ in named),
                issued_at=issued,
                start=start,
                duration_min=(last - first + 1) * RESOLUTION_MIN,
                turn_down_kw=sum(k for _, k in named),
                it_load_kw=int(resource.it_load_kw[first]),
            )
            drafts.append((start, j, cls.class_id, proposed, named, first, last - first + 1))
    drafts.sort(key=lambda x: (x[0], x[1]))

    accepted: list[PastCall] = list(resource.history)
    planned: list[PlannedCall] = []
    avoided_f = ramp_f = rebound_f = 0.0
    call_cost = Fraction(0)
    cost_unknown: set[str] = set()
    terms = {e.envelope_id: e for e in resource.envelopes}
    for start, _j, class_id, proposed, named, first, length in drafts:
        verdict = check_call(d, proposed, accepted)
        planned.append(
            PlannedCall(
                class_id=class_id,
                call=proposed,
                envelope_kw=named,
                first_interval=first,
                intervals=length,
                check=verdict,
            )
        )
        if not verdict.ok:
            notes.append(
                f"NOT CREDITED: the planned {class_id} call at {start.isoformat()} fails the "
                f"contract's check_call ({', '.join(r.value for r in verdict.reasons)})."
            )
            continue
        accepted.append(
            PastCall(
                kind="past_call",
                envelope_ids=proposed.envelope_ids,
                start=proposed.start,
                duration_min=proposed.duration_min,
                turn_down_kw=proposed.turn_down_kw,
            )
        )
        last = first + length - 1
        total_kw = proposed.turn_down_kw
        avoided_f += float(np.sum(value_kw[first : last + 1])) * total_kw * dt
        for eid, k in named:
            term = terms[eid]
            ramp_f += float(value_kw[first]) * k * term.ramp_s / 3600.0
            window = n - 1 if term.return_intervals is None else last + term.return_intervals
            worst = float(np.max(back_kw[last : min(window, n - 1) + 1]))
            rebound_f += term.recovery * k * (proposed.duration_min / 60.0) * worst
            cost = call_cost_pence(term.price, reduction_kw=k, duration_min=proposed.duration_min)
            if isinstance(cost, Unknown):
                cost_unknown.add(eid)
            else:
                call_cost += cost

    availability = Fraction(0)
    for term in resource.envelopes:
        held = term.layer_kw
        if not bool(np.any(held > 0.0)):
            continue
        for kw_held in held[held > 0.0]:
            cost = availability_cost_pence(
                term.price, held_kw=int(kw_held), held_min=RESOLUTION_MIN
            )
            if isinstance(cost, Unknown):
                cost_unknown.add(term.envelope_id)
                break
            availability += cost

    sacrificed = max(0.0, float(floor_value_gbp) - float(overlay_floor_value_gbp))
    f = float(annual_factor)
    avoided_y = avoided_f * f
    ramp_y = ramp_f * f
    rebound_y = rebound_f * f
    call_y = _pounds(call_cost) * f
    avail_y = _pounds(availability) * f
    net = avoided_y - ramp_y - call_y - rebound_y - avail_y - sacrificed
    notes.append(
        f"NETTED AGAINST THE PRICED FLOOR (no double-count): the overlay's battery gives up "
        f"£{sacrificed:,.2f}/yr of floor £ (measured on both schedules), subtracted here. The "
        "floor solve's dispatch, bill and reported revenue are unchanged by this resource."
    )
    if cost_unknown:
        notes.append(f"GROSS of an unknown operator cost for: {', '.join(sorted(cost_unknown))}.")
    unchecked = tuple(u for cls in resource.classes for u in cls.unchecked)
    return ConditionalComputeOutcome(
        declaration=resource.reference,
        duration_min=resource.duration_min,
        planned_calls=tuple(planned),
        avoided_import_gbp=round(avoided_y, 2),
        ramp_loss_gbp=round(ramp_y, 2),
        call_cost_gbp=round(call_y, 2),
        rebound_cost_gbp=round(rebound_y, 2),
        availability_cost_gbp=round(avail_y, 2),
        sacrificed_floor_gbp=round(sacrificed, 2),
        net_value_gbp=round(net, 2),
        cost_unknown=tuple(sorted(cost_unknown)),
        price_basis=tuple((e.envelope_id, price_basis_label(e.price)) for e in d.envelopes),
        notes=tuple(notes),
        unchecked=unchecked,
    )
