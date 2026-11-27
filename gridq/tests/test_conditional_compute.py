"""WP-SIZE-11: declared compute turn-down envelopes enter the constraint library as CONDITIONAL
resources only (spec B8-03), so the sizer (A5) and live dispatch (B2) take them from one place.

Acceptance, as the roadmap states it:

1. **Envelope value is always classified conditional and never enters the priced floor.**
   `test_the_outcome_is_conditional_by_construction`,
   `test_no_floor_reader_can_reach_the_compute_resource`,
   `test_the_floor_solve_is_untouched_by_a_declaration`.
2. **With no envelope declared the solution is unchanged.**
   `test_no_envelope_compiles_nothing_and_writes_the_floor_lp_byte_for_byte`, and the LP-file
   capture of the existing dispatch suites before and after the change (the WP-SIZE-02 method,
   recorded in ``docs/changelog.md``), and the golden tier with no golden file changed.

Every declaration here is a SYNTHETIC fixture (``record_class: synthetic_fixture``, ids starting
``SYNTHETIC-``, placeholder prices): no operator, site or workload is described.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import linopy
import numpy as np
import pandas as pd
import pytest
from tests._milp_support import multi_rate_site, write_hh_csv
from tests.contracts import _ic9_fixtures as ic9

import gridq.constraint_library as cl
from gridq.assumptions import AssumptionLibrary, load_assumptions
from gridq.constraint_library import catalogue
from gridq.constraint_library.compute import (
    FIRMNESS,
    ConditionalComputeOutcome,
    compute_resource,
)
from gridq.contracts.compute_turndown import (
    ComputeTurndownDeclaration,
    PastCall,
    check_call,
    declaration_ref,
)
from gridq.dispatch import DispatchProblem
from gridq.engine.errors import InvalidRequestError
from gridq.optimisation.compute_overlay import (
    NO_ENVELOPE,
    solve_window_with_compute,
    solve_with_compute_turndown,
)
from gridq.optimisation.milp_solver import MilpDispatchSolver, _floor_value
from gridq.schemas import PRICED_FLOOR_FIRMNESS, Firmness, SiteModel

SRC = Path(__file__).resolve().parents[1] / "src" / "gridq"

MOVE = "SYNTHETIC-CLASS-MOVED-INFERENCE"
DEFER = "SYNTHETIC-CLASS-DEFERRED-BATCH"
ENV_MOVE = "env-move"
ENV_DEFER = "env-defer"
POWER_KW, ENERGY_KWH = 200.0, 400.0
IT_KW = 200.0
DURATION_MIN = 120


# ============================================================================ fixtures


def _declaration(
    *,
    valid_from: str = "2025-01-06T00:00:00Z",
    valid_until: str = "2025-01-13T00:00:00Z",
    weekday_peak: dict[str, Any] | None = None,
    defer_price: Any = None,
) -> dict[str, Any]:
    """A SYNTHETIC two-class site offering two envelopes in the weekday 16:00-19:00 band.

    The moved-inference class's energy does not return, it may be called twice a week with 12 h
    rest, and its cumulative budget is unknown (never enforced, never assumed). The deferred batch
    class returns half its energy within 4 h, has a 240-minute weekly duration budget and an
    unknown price. Stack cap 150 kW, retained IT floor 50 kW.
    """
    seasons = [{"season_id": "all-year", "months": list(range(1, 13))}]
    groups = [
        {"day_group_id": "weekday", "iso_weekdays": [1, 2, 3, 4, 5]},
        {"day_group_id": "weekend", "iso_weekdays": [6, 7]},
    ]
    bands = [
        ic9._band("early", 0, 960),
        ic9._band("peak", 960, 1140),
        ic9._band("late", 1140, 1440),
    ]
    off = ic9._unavailable("workload_protected")
    peak = weekday_peak if weekday_peak is not None else ic9._offered(ENV_MOVE, ENV_DEFER)
    grid = {"all-year": {"weekday": [off, peak, off], "weekend": [off, off, off]}}
    return ic9._declaration(
        declaration_id="ic9-synthetic-size11",
        site_id="SYNTHETIC-DC-GPU-SIZE11",
        operator_id="SYNTHETIC-OPERATOR-GPU",
        issued_at="2025-01-01T00:00:00Z",
        valid_from=valid_from,
        valid_until=valid_until,
        supersedes=ic9._na("the only declaration for this window"),
        sla_classes=[
            {
                "class_id": DEFER,
                "operator_label": "SYNTHETIC-batch training, checkpoint and resume",
                "workload_action": "deferred",
                "energy_return": {"kind": "returns_later", "return_window_min": 240},
                "call_limits": ic9._limits(
                    ic9._na("no event budget"),
                    {"kind": "duration_budget", "max_minutes": 240, "window_days": 7},
                    ic9._na("no minimum rest"),
                ),
                "sla_evidence": ic9._unknown("no service-level document exists"),
            },
            {
                "class_id": MOVE,
                "operator_label": "SYNTHETIC-inference moved to another site",
                "workload_action": "moved",
                "energy_return": {"kind": "no_return"},
                "call_limits": ic9._limits(
                    {"kind": "event_budget", "max_events": 2, "window_days": 7},
                    ic9._unknown("the cumulative budget is not yet agreed"),
                    {"kind": "minimum_rest", "minutes": 720},
                ),
                "sla_evidence": ic9._unknown("no service-level document exists"),
            },
        ],
        envelopes=[
            {
                "envelope_id": ENV_DEFER,
                "sla_class_id": DEFER,
                "response": ic9._response(900, 120),
                "price": defer_price
                if defer_price is not None
                else ic9._unknown("the operator has not stated a price"),
                "curve": ic9._absolute((60, 80), (240, 50)),
                "rebound": {
                    "kind": "rebound_size",
                    "recovery_permille": 500,
                    "max_rebound_kw": ic9._unknown("not measured"),
                },
                "exclusion_group": ic9._na("stacks with the move envelope"),
            },
            {
                "envelope_id": ENV_MOVE,
                "sla_class_id": MOVE,
                "response": ic9._response(1800, 300),
                "price": ic9._price(
                    utilisation={"kind": "utilisation_price", "pence_per_mwh": 5000},
                    per_event={"kind": "event_fee", "pence": 500},
                ),
                "curve": ic9._absolute((120, 100), (240, 60)),
                "rebound": ic9._na("moved work does not return at this site"),
                "exclusion_group": ic9._na("stacks with the defer envelope"),
            },
        ],
        availability={
            "seasons": seasons,
            "day_groups": groups,
            "bands": bands,
            "cells": ic9._cells(seasons, groups, bands, grid),
            "overrides": [],
        },
        stacking={
            "kind": "stacking",
            "merit_order": [ENV_MOVE, ENV_DEFER],
            "max_concurrent_turn_down_kw": 150,
        },
        it_reference={"kind": "it_reference", "retained_it_floor_kw": 50},
        basis={"kind": "operator_declared", "supporting_evidence": []},
    )


def _decl(**kw: Any) -> ComputeTurndownDeclaration:
    return ComputeTurndownDeclaration.model_validate_json(json.dumps(_declaration(**kw)))


@pytest.fixture(scope="module")
def site_and_lib(tmp_path_factory: pytest.TempPathFactory) -> tuple[SiteModel, AssumptionLibrary]:
    path = tmp_path_factory.mktemp("size11") / "hh.csv"
    rows, _ = write_hh_csv(path, days=7)
    return multi_rate_site(path, rows=rows), load_assumptions("gb_v0_1")


def _problem(site: SiteModel, lib: AssumptionLibrary) -> DispatchProblem:
    return DispatchProblem(
        site_model=site,
        assumptions=lib,
        power_kw=POWER_KW,
        energy_kwh=ENERGY_KWH,
        duration_hours=ENERGY_KWH / POWER_KW,
    )


@pytest.fixture(scope="module")
def overlay_solver() -> MilpDispatchSolver:
    return MilpDispatchSolver()


@pytest.fixture(scope="module")
def overlay(
    site_and_lib: tuple[SiteModel, AssumptionLibrary], overlay_solver: MilpDispatchSolver
) -> Any:
    site, lib = site_and_lib
    return solve_with_compute_turndown(
        overlay_solver,
        _problem(site, lib),
        _decl(),
        it_load_kw=np.full(7 * 48, IT_KW),
        duration_min=DURATION_MIN,
    )


def _resource(decl: ComputeTurndownDeclaration, **kw: Any) -> Any:
    ts = pd.date_range("2025-01-06 00:00", periods=7 * 48, freq="30min")
    return compute_resource(decl, ts, np.full(ts.size, IT_KW), duration_min=DURATION_MIN, **kw)


# ============================================================================ the catalogue


def test_the_conditional_family_is_catalogued_and_compiled_by_both_paths() -> None:
    ids = {
        "compute_call_budget",
        "compute_call_shape",
        "compute_meter_displacement",
        "compute_turn_down_limit",
    }
    family = {
        s.constraint_id
        for s in catalogue.CATALOGUE
        if s.family is catalogue.Family.CONDITIONAL_RESOURCE
    }
    assert family == ids
    assert ids <= set(cl.constraint_ids()) and ids <= set(cl.library_ref().constraint_ids)
    for i in ids:
        spec = catalogue.spec(i)
        assert spec.module == "compute"
        assert set(spec.applied_by) == {catalogue.ANNUAL_MILP, catalogue.MPC_WINDOW}
        assert "never the floor" in spec.summary


def test_the_library_release_is_a_minor_with_its_hash_pinned() -> None:
    from gridq.constraint_library import _releases, identity

    assert identity.LIBRARY_VERSION == "1.1.0"
    assert _releases.RELEASES["1.1.0"] == identity.library_hash()


# ============================================================================ 1. conditional


def test_the_outcome_is_conditional_by_construction(overlay: Any) -> None:
    """ACCEPTANCE 1. The one classification any compute £ carries is CONDITIONAL, outside the
    priced floor's firmness set, and the record refuses to be built as anything else."""
    outcome = overlay.outcome
    assert isinstance(outcome, ConditionalComputeOutcome)
    assert outcome.firmness is Firmness.CONDITIONAL is FIRMNESS
    assert outcome.firmness not in PRICED_FLOOR_FIRMNESS
    assert outcome.in_priced_floor is False
    assert outcome.resource_id == "conditional_compute_resource"
    assert any(n.startswith("CONDITIONAL (B8-03)") for n in outcome.notes)
    for bad in (
        {"firmness": Firmness.FIRM},
        {"firmness": Firmness.CONTRACTED},
        {"in_priced_floor": True},
    ):
        with pytest.raises(ValueError, match="CONDITIONAL resource"):
            ConditionalComputeOutcome(
                **{
                    f: getattr(outcome, f)
                    for f in outcome.__dataclass_fields__
                    if f not in ("firmness", "in_priced_floor")
                },
                **bad,
            )


def _names_in(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(f"kw:{node.arg}")
    return names


def test_no_floor_reader_can_reach_the_compute_resource() -> None:
    """ACCEPTANCE 1 (static). The only code that hands a compute resource to a solve, or reads
    one back, is the conditional overlay module; the floor path (valuation, schedule, finance, the
    engine, the sizer, the MPC loop, the quote and the outputs) never names it. Delete the guard
    in `_solve_milp` or call the overlay from a floor module and this fails."""
    overlay_module = SRC / "optimisation" / "compute_overlay.py"
    solver_module = SRC / "optimisation" / "milp_solver.py"
    reaches = {
        "compute_overlay",
        "gridq.optimisation.compute_overlay",
        "compute_resource",
        "kw:compute_resource",
        "_last_compute_solution",
        "ConditionalComputeOutcome",
        "account_conditional_compute",
        "compile_conditional_compute",
    }
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path in (overlay_module, solver_module) or path.parent.name == "constraint_library":
            continue
        hit = reaches & _names_in(path)
        if hit:
            offenders.append((str(path.relative_to(SRC)), sorted(hit)))
    assert offenders == []
    # Inside the solver, the resource is handed on only by the window entry point, never by
    # `solve` (the floor) or by the export / agreed-capacity overlays.
    tree = ast.parse(solver_module.read_text(encoding="utf-8"))
    passers = sorted(
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        for node in ast.walk(fn)
        if isinstance(node, ast.keyword) and node.arg == "compute_resource"
    )
    assert passers == ["solve_window_outcome"]


def test_the_floor_solve_is_untouched_by_a_declaration(
    site_and_lib: tuple[SiteModel, AssumptionLibrary],
    overlay: Any,
    overlay_solver: MilpDispatchSolver,
) -> None:
    """ACCEPTANCE 1 (dynamic). The overlay's floor is the solver's own `solve`, bit for bit; the
    conditional £ is netted against it (the floor £ the overlay's battery gave up is measured on
    both schedules and subtracted), so it never restates a floor pound."""
    site, lib = site_and_lib
    solver = MilpDispatchSolver()
    result = solver.solve(_problem(site, lib))
    floor = solver.last_schedule()
    assert floor is not None and overlay.floor_schedule is not None
    for name in ("charge_kw", "discharge_kw", "soe_kwh"):
        np.testing.assert_array_equal(getattr(overlay.floor_schedule, name), getattr(floor, name))
    assert result.value_streams  # the floor's own streams, from its own solve
    # The overlay never sets the solver's floor state: a caller's floor result is never replaced.
    assert overlay_solver.last_schedule() is None
    assert overlay_solver.last_customer_bill() is None
    outcome = overlay.outcome
    sacrificed = max(0.0, _floor_value(floor) - _floor_value(overlay.overlay_schedule))
    assert outcome.sacrificed_floor_gbp == pytest.approx(round(sacrificed, 2), abs=0.01)
    gross = (
        outcome.avoided_import_gbp
        - outcome.ramp_loss_gbp
        - outcome.call_cost_gbp
        - outcome.rebound_cost_gbp
        - outcome.availability_cost_gbp
    )
    assert outcome.net_value_gbp == pytest.approx(gross - outcome.sacrificed_floor_gbp, abs=0.05)


# ============================================================================ 2. no envelope


def _capture_lp(
    monkeypatch: pytest.MonkeyPatch,
    out: Path,
    names: list[set[str]] | None = None,
    *,
    run_solver: bool = True,
) -> list[Path]:
    """Write every model handed to the solver as an LP file (and its row and variable names:
    the LP file itself carries linopy's positional labels only). ``run_solver=False`` stops at the
    build: the solver then reports a non-optimal termination and reads no solution."""
    written: list[Path] = []
    real = linopy.Model.solve

    def solve(self: linopy.Model, *args: Any, **kwargs: Any) -> Any:
        path = out / f"model_{len(written)}.lp"
        self.to_file(path, io_api="lp", progress=False)
        written.append(path)
        if names is not None:
            names.append(set(self.constraints) | set(self.variables))
        if not run_solver:
            return ("warning", "not_solved_build_only")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(linopy.Model, "solve", solve)
    return written


def test_no_envelope_compiles_nothing_and_writes_the_floor_lp_byte_for_byte(
    site_and_lib: tuple[SiteModel, AssumptionLibrary],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """ACCEPTANCE 2. No declaration: nothing is solved or compiled by the overlay. A declaration
    that offers nothing over the horizon (valid only after it) compiles nothing: the model the
    solver writes is the floor's, byte for byte, in the annual build and in an MPC window."""
    site, lib = site_and_lib
    problem = _problem(site, lib)
    solver = MilpDispatchSolver()
    names: list[set[str]] = []
    written = _capture_lp(monkeypatch, tmp_path, names)

    nothing = solve_with_compute_turndown(
        solver, problem, None, it_load_kw=np.full(7 * 48, IT_KW), duration_min=DURATION_MIN
    )
    assert nothing.outcome is None and nothing.warnings == (NO_ENVELOPE,)
    assert written == []

    load, net, ts, interval, _ = solver._load_series_signed(site.load_data.half_hourly_file_ref)
    later = _decl(valid_from="2025-02-03T00:00:00Z", valid_until="2025-02-10T00:00:00Z")
    idle = compute_resource(later, ts, np.full(ts.size, IT_KW), duration_min=DURATION_MIN)
    assert not idle.offers_anything
    rte = 0.85
    solver._solve_milp(problem, load, ts, interval, rte, net_load_kw=net)
    solver._solve_milp(problem, load, ts, interval, rte, net_load_kw=net, compute_resource=idle)
    window = slice(0, 96)
    solver.solve_window_outcome(problem, load[window], ts[window], interval, rte, 40.0)
    solver.solve_window_outcome(
        problem,
        load[window],
        ts[window],
        interval,
        rte,
        40.0,
        compute_resource=compute_resource(
            later, ts[window], np.full(96, IT_KW), duration_min=DURATION_MIN
        ),
    )
    assert len(written) == 4
    assert written[0].read_bytes() == written[1].read_bytes()
    assert written[2].read_bytes() == written[3].read_bytes()
    assert names[0] == names[1] and names[2] == names[3]
    assert not any(n.startswith("compute") for n in names[1] | names[3])


def test_an_offering_declaration_does_compile_the_library_rows(
    site_and_lib: tuple[SiteModel, AssumptionLibrary],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The control for acceptance 2: the byte identity above is not a resource that is never read.
    An offering declaration adds the catalogue's rows to the same build."""
    site, lib = site_and_lib
    solver = MilpDispatchSolver()
    load, net, ts, interval, _ = solver._load_series_signed(site.load_data.half_hourly_file_ref)
    names: list[set[str]] = []
    _capture_lp(monkeypatch, tmp_path, names, run_solver=False)
    schedule, _binding, _warn = solver._solve_milp(
        _problem(site, lib),
        load,
        ts,
        interval,
        0.85,
        net_load_kw=net,
        compute_resource=compute_resource(
            _decl(), ts, np.full(ts.size, IT_KW), duration_min=DURATION_MIN
        ),
    )
    (built,) = names
    for row in (
        "compute_turn_down_limit_0",
        "compute_call_level_up_0",
        "compute_run_0",
        "compute_minutes_0",
        "compute_events_1",
        "compute_rest_1",
        "compute_meter_displacement",
    ):
        assert any(name == row or name.startswith(f"{row}_") for name in built), row
    assert schedule is None and solver._last_compute_solution is None  # built, never solved


# ============================================================================ the calls


def test_every_credited_call_passes_the_contracts_check(overlay: Any) -> None:
    """D-NEW-SIZE11-4. Every planned call is the contract's `ProposedCall` at whole kW, checked
    by `check_call` against the calls accepted before it; the budgets, the rest and the stack
    cap hold over the credited set; the unknown cumulative budget is stated, not enforced."""
    outcome = overlay.outcome
    decl = _decl()
    credited = outcome.credited_calls
    assert credited, "the synthetic peak offer should be worth calling"
    accepted: list[PastCall] = []
    for p in credited:
        assert check_call(decl, p.call, accepted).ok
        accepted.append(
            PastCall(
                kind="past_call",
                envelope_ids=p.call.envelope_ids,
                start=p.call.start,
                duration_min=p.call.duration_min,
                turn_down_kw=p.call.turn_down_kw,
            )
        )
        assert p.call.duration_min <= DURATION_MIN
        assert p.call.start - p.call.issued_at >= timedelta(seconds=900)
    moves = sorted((p.call for p in credited if p.class_id == MOVE), key=lambda c: c.start)
    assert len(moves) <= 2
    for a, b in zip(moves, moves[1:], strict=False):
        assert b.start - (a.start + timedelta(minutes=a.duration_min)) >= timedelta(minutes=720)
    defer_minutes = sum(p.call.duration_min for p in credited if p.class_id == DEFER)
    assert defer_minutes <= 240
    assert outcome.unchecked == (f"sla_classes[{MOVE}].call_limits.max_cumulative",)
    assert any("NOT ENFORCED, NOT ASSUMED" in n for n in outcome.notes)
    assert outcome.declaration == declaration_ref(decl)


def test_an_unknown_price_leaves_the_value_gross_and_says_so(overlay: Any) -> None:
    outcome = overlay.outcome
    assert outcome.cost_unknown == (ENV_DEFER,)
    assert any(n.startswith("GROSS of an unknown operator cost") for n in outcome.notes)
    assert dict(outcome.price_basis)[ENV_MOVE] == "SYNTHETIC PLACEHOLDER, not an offer"


def test_one_kw_of_load_is_displaced_once(overlay: Any) -> None:
    """``compute_meter_displacement``: the battery and the turn-down never displace the same kW."""
    sched = overlay.overlay_schedule
    turned = np.zeros(sched.discharge_kw.size)
    for p in overlay.outcome.credited_calls:
        turned[p.first_interval : p.first_interval + p.intervals] += p.call.turn_down_kw
    assert np.all(sched.discharge_kw + turned <= sched.site_load_kw + 1e-6)


# ============================================================================ live dispatch


def test_a_live_window_honours_notice_and_the_history(
    site_and_lib: tuple[SiteModel, AssumptionLibrary], monkeypatch: pytest.MonkeyPatch
) -> None:
    """B2 inherits the same implementation: a window solved through `solve_window_outcome` with
    the calls issued now. A history that has used the moved class's two events leaves it nothing
    to call; an envelope inside its notice is not offered; the rest is still planned and checked."""
    site, lib = site_and_lib
    solver = MilpDispatchSolver()
    load, _net, ts, interval, _ = solver._load_series_signed(site.load_data.half_hourly_file_ref)
    built: list[set[str]] = []
    real = linopy.Model.solve

    def spy(self: linopy.Model, *args: Any, **kwargs: Any) -> Any:
        built.append(set(self.constraints))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(linopy.Model, "solve", spy)
    # Monday 15:45 UTC: the 16:00 half-hour is inside env-move's 30-minute notice.
    window = slice(32, 32 + 96)
    now = datetime(2025, 1, 6, 15, 45, tzinfo=UTC)
    history = [
        PastCall(
            kind="past_call",
            envelope_ids=(ENV_MOVE,),
            start=datetime(2025, 1, 3, 16, 0, tzinfo=UTC),
            duration_min=60,
            turn_down_kw=10,
        ),
        PastCall(
            kind="past_call",
            envelope_ids=(ENV_MOVE,),
            start=datetime(2025, 1, 4, 16, 0, tzinfo=UTC),
            duration_min=60,
            turn_down_kw=10,
        ),
    ]
    decl = _decl(valid_from="2025-01-01T00:00:00Z")
    result = solve_window_with_compute(
        solver,
        _problem(site, lib),
        decl,
        load=load[window],
        timestamps=ts[window],
        interval_min=interval,
        initial_soe_kwh=40.0,
        it_load_kw=np.full(96, IT_KW),
        duration_min=DURATION_MIN,
        issued_at=now,
        history=history,
    )
    assert result.window.schedule is not None and result.outcome is not None
    assert len(built) == 2  # the window's floor, then the overlay window
    assert "compute_turn_down_limit_1" in built[1] and "compute_meter_displacement" in built[1]
    assert not any(name.startswith("compute") for name in built[0])
    resource = compute_resource(
        decl,
        ts[window],
        np.full(96, IT_KW),
        duration_min=DURATION_MIN,
        issued_at=now,
        history=history,
    )
    move = next(e for e in resource.envelopes if e.envelope_id == ENV_MOVE)
    defer = next(e for e in resource.envelopes if e.envelope_id == ENV_DEFER)
    move_class = next(c for c in resource.classes if c.class_id == MOVE)
    assert move_class.events == (0, 7 * 48)  # the history used both events
    assert move.layer_kw[0] == 0.0 and move.layer_kw[1] > 0.0  # 30-minute notice from 15:45
    assert defer.layer_kw[0] > 0.0  # 15-minute notice: 16:00 is callable
    accepted = list(history)
    for p in result.outcome.planned_calls:
        assert p.class_id == DEFER or not p.credited
        assert p.call.issued_at == now
        if p.credited:
            assert check_call(decl, p.call, accepted).ok
            accepted.append(
                PastCall(
                    kind="past_call",
                    envelope_ids=p.call.envelope_ids,
                    start=p.call.start,
                    duration_min=p.call.duration_min,
                    turn_down_kw=p.call.turn_down_kw,
                )
            )


# ============================================================================ the guards


def test_the_resource_refuses_what_it_cannot_credit() -> None:
    decl = _decl()
    ts = pd.date_range("2025-01-06 00:00", periods=48, freq="30min")
    with pytest.raises(ValueError, match="whole half-hours"):
        compute_resource(decl, ts, np.full(48, IT_KW), duration_min=45)
    with pytest.raises(ValueError, match="30-minute grid"):
        compute_resource(decl, ts, np.full(48, IT_KW), duration_min=60, interval_min=15)
    with pytest.raises(ValueError, match="it_load_kw"):
        compute_resource(decl, ts, np.full(47, IT_KW), duration_min=60)
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_resource(
            decl, ts, np.full(48, IT_KW), duration_min=60, issued_at=datetime(2025, 1, 6)
        )


def test_a_weighted_proxy_solve_refuses_the_resource(
    site_and_lib: tuple[SiteModel, AssumptionLibrary],
) -> None:
    site, lib = site_and_lib
    solver = MilpDispatchSolver()
    load, net, ts, interval, _ = solver._load_series_signed(site.load_data.half_hourly_file_ref)
    with pytest.raises(InvalidRequestError, match="weighted"):
        solver._solve_milp(
            _problem(site, lib),
            load,
            ts,
            interval,
            0.85,
            interval_weights=np.ones(load.size),
            compute_resource=_resource(_decl()),
        )


def test_layers_come_from_the_contracts_stack() -> None:
    """The layers are `stack_at`'s: at a 120-minute call and 200 kW IT, env-move holds 100 kW
    first in merit order and env-defer 50 kW (the cap of 150 kW and the 50 kW retained IT floor
    leave 50), only in the weekday 16:00-19:00 band."""
    res = _resource(_decl())
    move = next(e for e in res.envelopes if e.envelope_id == ENV_MOVE)
    defer = next(e for e in res.envelopes if e.envelope_id == ENV_DEFER)
    peak = np.zeros(7 * 48, dtype=bool)
    for day in range(5):
        peak[day * 48 + 32 : day * 48 + 38] = True
    assert set(np.unique(move.layer_kw[peak])) == {100.0}
    assert set(np.unique(defer.layer_kw[peak])) == {50.0}
    assert not move.layer_kw[~peak].any() and not defer.layer_kw[~peak].any()
    assert move.recovery == 0.0 and defer.recovery == 0.5
    assert move.utilisation_gbp_per_kwh == pytest.approx(0.05)
    assert defer.utilisation_gbp_per_kwh is None
