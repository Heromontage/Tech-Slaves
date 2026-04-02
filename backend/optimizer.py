"""
optimizer.py — SentinelFlow OR-Tools Linear Programming Rerouting Engine
=========================================================================
Solves a minimum-cost flow problem when a node in the supply chain graph
is disrupted.  Given a set of alternative routes and a total cargo volume
to redistribute, the LP finds the optimal split across available lanes to:

    Minimize Z = Σ(cost_per_teu * flow) + Σ(risk_factor * delay_hours * flow)

Subject to:
    • Flow conservation  : total outbound flow == total_cargo
    • Capacity bounds    : 0 ≤ flow[route] ≤ route.capacity_teu
    • Throughput floor   : Σ flow ≥ min_throughput_pct * total_cargo
    • Non-negativity     : flow[route] ≥ 0

Usage
-----
    from optimizer import optimize_rerouting, RouteOption, GraphData

    graph = GraphData(
        disrupted_node_id="PORT-CN-SHA",
        routes=[
            RouteOption("SHA→BSN→LGB",  mode="Rail+Sea", cost=2230.0,
                        delay_hours=528.0, risk_factor=0.15, capacity_teu=8000),
            RouteOption("SHA→SIN→LGB",  mode="Sea",      cost=2450.0,
                        delay_hours=576.0, risk_factor=0.22, capacity_teu=6000),
            RouteOption("SHA→LGB_air",  mode="Air",       cost=9800.0,
                        delay_hours=48.0,  risk_factor=0.05, capacity_teu=1200),
        ]
    )
    result = optimize_rerouting(graph, disrupted_node_id="PORT-CN-SHA", total_cargo=12000)
    print(result)

Requirements
------------
    pip install ortools
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("sentinelflow.optimizer")


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class RouteOption:
    """
    A single alternative shipping lane that can absorb diverted cargo.

    Attributes
    ----------
    route_id        : Unique identifier, e.g. "SHA→BSN→LGB-Rail".
    mode            : Transport mode string for display purposes.
    cost_per_teu    : Base USD cost per Twenty-foot Equivalent Unit.
    delay_hours     : Expected end-to-end transit time in hours.
    risk_factor     : Dimensionless multiplier in [0, 1] representing
                      the probability-weighted disruption risk on this lane.
                      0 = zero risk, 1 = near-certain disruption.
    capacity_teu    : Hard upper-bound on flow for this lane (TEU).
    base_delay_hours: Normal (undisrupted) transit time used to compute
                      incremental delay cost.  Defaults to delay_hours * 0.8
                      if not provided.
    carrier         : Optional carrier name for display.
    """
    route_id:         str
    mode:             str
    cost_per_teu:     float
    delay_hours:      float
    risk_factor:      float
    capacity_teu:     int
    base_delay_hours: Optional[float] = None
    carrier:          Optional[str]   = None

    def __post_init__(self) -> None:
        if self.base_delay_hours is None:
            self.base_delay_hours = self.delay_hours * 0.8

        if not (0.0 <= self.risk_factor <= 1.0):
            raise ValueError(
                f"risk_factor must be in [0, 1], got {self.risk_factor} "
                f"for route '{self.route_id}'."
            )
        if self.capacity_teu <= 0:
            raise ValueError(
                f"capacity_teu must be > 0, got {self.capacity_teu} "
                f"for route '{self.route_id}'."
            )

    @property
    def incremental_delay(self) -> float:
        """Extra hours above the undisrupted baseline."""
        return max(0.0, self.delay_hours - self.base_delay_hours)

    @property
    def composite_cost(self) -> float:
        """
        Single scalar used as the LP objective coefficient per TEU.

        Combines direct shipping cost with a monetised risk-delay penalty:

            composite = cost_per_teu + risk_factor * incremental_delay

        The risk_factor * incremental_delay term gives heavier weight to
        routes that are both risky AND slow, nudging the solver toward
        safer, faster alternatives even at slightly higher base cost.
        """
        return self.cost_per_teu + self.risk_factor * self.incremental_delay


@dataclass
class GraphData:
    """
    Lightweight representation of the supply chain graph state passed
    to the optimiser.

    Attributes
    ----------
    disrupted_node_id : The node that is congested or offline.
    routes            : Available alternative RouteOptions for rerouting.
    min_throughput_pct: Minimum fraction of total_cargo that must be
                        delivered (default 0.90 → 90 %).
    """
    disrupted_node_id: str
    routes:            list[RouteOption]
    min_throughput_pct: float = 0.90

    def total_capacity(self) -> int:
        return sum(r.capacity_teu for r in self.routes)


@dataclass
class RouteAllocation:
    """Per-route result from the optimiser."""
    route_id:     str
    mode:         str
    carrier:      Optional[str]
    flow_teu:     float          # TEU assigned by the LP
    cost_usd:     float          # cost_per_teu * flow_teu
    delay_hours:  float          # route delay_hours (fixed; not flow-dependent)
    risk_factor:  float
    pct_of_total: float          # flow_teu / total_cargo * 100

    def __str__(self) -> str:
        carrier_str = f" [{self.carrier}]" if self.carrier else ""
        return (
            f"  {self.route_id}{carrier_str} ({self.mode})\n"
            f"    Flow    : {self.flow_teu:>10,.1f} TEU  ({self.pct_of_total:.1f}%)\n"
            f"    Cost    : ${self.cost_usd:>12,.2f}\n"
            f"    Delay   : {self.delay_hours:.0f}h\n"
            f"    Risk    : {self.risk_factor:.2f}"
        )


@dataclass
class OptimizationResult:
    """
    Full output of a single optimisation run.

    Attributes
    ----------
    status              : "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | "ERROR"
    disrupted_node_id   : Which node triggered this run.
    total_cargo_teu     : Input cargo volume.
    total_flow_teu      : Actually allocated flow (may be < total_cargo if
                          capacity is insufficient but min_throughput met).
    total_cost_usd      : Σ cost_per_teu * flow across all routes.
    weighted_delay_hours: Cargo-weighted average delay across all routes.
    throughput_pct      : total_flow_teu / total_cargo_teu * 100.
    allocations         : Per-route breakdown.
    solver_wall_ms      : Wall-clock time spent in the LP solver (ms).
    message             : Human-readable status note.
    """
    status:               str
    disrupted_node_id:    str
    total_cargo_teu:      float
    total_flow_teu:       float          = 0.0
    total_cost_usd:       float          = 0.0
    weighted_delay_hours: float          = 0.0
    throughput_pct:       float          = 0.0
    allocations:          list[RouteAllocation] = field(default_factory=list)
    solver_wall_ms:       float          = 0.0
    message:              str            = ""

    def __str__(self) -> str:
        lines = [
            "═" * 60,
            f"  SentinelFlow Rerouting Optimisation — {self.status}",
            "═" * 60,
            f"  Disrupted node  : {self.disrupted_node_id}",
            f"  Total cargo     : {self.total_cargo_teu:,.0f} TEU",
            f"  Allocated flow  : {self.total_flow_teu:,.1f} TEU "
            f"({self.throughput_pct:.1f}% throughput)",
            f"  Total cost      : ${self.total_cost_usd:,.2f}",
            f"  Avg delay       : {self.weighted_delay_hours:.1f}h",
            f"  Solver time     : {self.solver_wall_ms:.1f}ms",
            "",
            "  Route Allocations:",
            "─" * 60,
        ]
        if self.allocations:
            for alloc in sorted(self.allocations, key=lambda a: -a.flow_teu):
                lines.append(str(alloc))
        else:
            lines.append("  (none)")

        if self.message:
            lines += ["─" * 60, f"  Note: {self.message}"]
        lines.append("═" * 60)
        return "\n".join(lines)


# ── Core optimisation function ────────────────────────────────────────────────

def optimize_rerouting(
    graph_data:       GraphData,
    disrupted_node_id: str,
    total_cargo:      int,
    *,
    risk_weight:      float = 1.0,
    allow_partial:    bool  = True,
) -> OptimizationResult:
    """
    Solve a Linear Programme to minimise total cost + risk-delay across all
    available rerouting lanes when ``disrupted_node_id`` goes offline.

    Objective
    ---------
        Minimize Z = Σᵢ [ (cost_per_teu[i] + w * risk_factor[i] * incremental_delay[i])
                          * flow[i] ]

    Constraints
    -----------
        Σᵢ flow[i]   ≤ total_cargo                    (demand upper bound)
        Σᵢ flow[i]   ≥ min_throughput_pct * total_cargo (throughput floor)
        0 ≤ flow[i]  ≤ capacity_teu[i]               (per-route capacity)

    Parameters
    ----------
    graph_data          : GraphData instance with alternative RouteOptions.
    disrupted_node_id   : The congested / offline node to route around.
    total_cargo         : Total TEU volume to redistribute.
    risk_weight         : Scalar multiplier applied to the risk-delay cost
                          term.  Increase to penalise risky routes more
                          heavily (default 1.0).
    allow_partial       : If True, the solver may deliver less than
                          total_cargo when network capacity is insufficient
                          but the throughput floor can still be met.
                          If False, an exact-demand equality constraint is
                          added and the problem becomes infeasible when
                          capacity < total_cargo.

    Returns
    -------
    OptimizationResult
        Contains per-route flow allocations, total cost, weighted average
        delay, and solver diagnostics.

    Raises
    ------
    ImportError
        If ``ortools`` is not installed.
    ValueError
        If ``graph_data.routes`` is empty or ``total_cargo`` ≤ 0.
    """
    # ── Validation ─────────────────────────────────────────────────────────────
    if not graph_data.routes:
        raise ValueError("graph_data.routes must not be empty.")
    if total_cargo <= 0:
        raise ValueError(f"total_cargo must be > 0, got {total_cargo}.")

    # ── Import OR-Tools ────────────────────────────────────────────────────────
    try:
        from ortools.linear_solver import pywraplp
    except ImportError as exc:
        raise ImportError(
            "Google OR-Tools is required for the optimisation engine.\n"
            "Install it with: pip install ortools\n"
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "Optimising rerouting for node '%s' — %d TEU across %d routes "
        "(risk_weight=%.2f)",
        disrupted_node_id, total_cargo, len(graph_data.routes), risk_weight,
    )

    routes    = graph_data.routes
    n_routes  = len(routes)
    min_flow  = graph_data.min_throughput_pct * total_cargo
    total_cap = graph_data.total_capacity()

    # ── Build solver ───────────────────────────────────────────────────────────
    solver = pywraplp.Solver.CreateSolver("GLOP")   # Glop = Google's LP solver
    if solver is None:
        return OptimizationResult(
            status="ERROR",
            disrupted_node_id=disrupted_node_id,
            total_cargo_teu=float(total_cargo),
            message="Could not initialise GLOP solver.",
        )

    infinity = solver.infinity()

    # ── Decision variables: flow[i] = TEU routed via route i ──────────────────
    flow: list[pywraplp.Variable] = [
        solver.NumVar(0.0, float(r.capacity_teu), f"flow_{i}")
        for i, r in enumerate(routes)
    ]

    # ── Objective function ─────────────────────────────────────────────────────
    # Z = Σ (cost_per_teu[i] + risk_weight * risk_factor[i] * Δdelay[i]) * flow[i]
    objective = solver.Objective()
    for i, r in enumerate(routes):
        coeff = r.cost_per_teu + risk_weight * r.risk_factor * r.incremental_delay
        objective.SetCoefficient(flow[i], coeff)
        logger.debug(
            "  route[%d] %-30s  cost_coeff=%.4f", i, r.route_id, coeff
        )
    objective.SetMinimization()

    # ── Constraints ────────────────────────────────────────────────────────────

    # (1) Total flow upper bound: cannot ship more than demand
    c_upper = solver.Constraint(0.0, float(total_cargo), "total_flow_upper")
    for v in flow:
        c_upper.SetCoefficient(v, 1.0)

    # (2) Throughput floor: must ship at least min_throughput_pct * total_cargo
    #     If network capacity is insufficient, relax floor to total capacity
    #     so the problem stays feasible.
    effective_min = min(min_flow, float(total_cap))
    c_lower = solver.Constraint(effective_min, infinity, "total_flow_lower")
    for v in flow:
        c_lower.SetCoefficient(v, 1.0)

    if effective_min < min_flow:
        logger.warning(
            "Network capacity (%d TEU) < throughput floor (%.0f TEU). "
            "Relaxing floor to %d TEU.",
            total_cap, min_flow, total_cap,
        )

    # (3) Exact demand (optional — adds equality instead of upper bound)
    if not allow_partial:
        c_exact = solver.Constraint(
            float(total_cargo), float(total_cargo), "exact_demand"
        )
        for v in flow:
            c_exact.SetCoefficient(v, 1.0)

    # ── Solve ──────────────────────────────────────────────────────────────────
    status_code = solver.Solve()
    wall_ms     = solver.wall_time()   # milliseconds

    _STATUS_MAP = {
        pywraplp.Solver.OPTIMAL:   "OPTIMAL",
        pywraplp.Solver.FEASIBLE:  "FEASIBLE",
        pywraplp.Solver.INFEASIBLE:"INFEASIBLE",
        pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
        pywraplp.Solver.ABNORMAL:  "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED:"NOT_SOLVED",
    }
    status_str = _STATUS_MAP.get(status_code, "UNKNOWN")
    logger.info("Solver returned: %s in %.1f ms", status_str, wall_ms)

    if status_code not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        msg = (
            "Network capacity is insufficient to meet minimum throughput "
            "requirements." if status_code == pywraplp.Solver.INFEASIBLE
            else f"Solver ended with status {status_str}."
        )
        return OptimizationResult(
            status=status_str,
            disrupted_node_id=disrupted_node_id,
            total_cargo_teu=float(total_cargo),
            solver_wall_ms=wall_ms,
            message=msg,
        )

    # ── Extract solution ───────────────────────────────────────────────────────
    allocations:    list[RouteAllocation] = []
    total_flow_teu: float = 0.0
    total_cost_usd: float = 0.0
    weighted_delay: float = 0.0

    for i, r in enumerate(routes):
        f = flow[i].solution_value()
        if f < 1e-6:
            # Negligible — skip to keep output clean
            continue

        cost_usd = r.cost_per_teu * f
        total_flow_teu += f
        total_cost_usd += cost_usd
        weighted_delay += r.delay_hours * f

        allocations.append(RouteAllocation(
            route_id    = r.route_id,
            mode        = r.mode,
            carrier     = r.carrier,
            flow_teu    = round(f, 2),
            cost_usd    = round(cost_usd, 2),
            delay_hours = r.delay_hours,
            risk_factor = r.risk_factor,
            pct_of_total= 0.0,  # filled below
        ))

    # Compute percentages now that total_flow_teu is known
    for alloc in allocations:
        alloc.pct_of_total = (
            round(alloc.flow_teu / total_flow_teu * 100, 2)
            if total_flow_teu > 0 else 0.0
        )

    throughput_pct      = (total_flow_teu / total_cargo * 100) if total_cargo else 0.0
    avg_weighted_delay  = (weighted_delay / total_flow_teu) if total_flow_teu else 0.0

    result = OptimizationResult(
        status               = status_str,
        disrupted_node_id    = disrupted_node_id,
        total_cargo_teu      = float(total_cargo),
        total_flow_teu       = round(total_flow_teu, 2),
        total_cost_usd       = round(total_cost_usd, 2),
        weighted_delay_hours = round(avg_weighted_delay, 2),
        throughput_pct       = round(throughput_pct, 2),
        allocations          = allocations,
        solver_wall_ms       = round(wall_ms, 2),
        message              = (
            "Solution is feasible but not proven globally optimal."
            if status_str == "FEASIBLE" else ""
        ),
    )

    logger.info(
        "Optimisation complete — flow=%.0f TEU  cost=$%.2f  "
        "avg_delay=%.1fh  throughput=%.1f%%",
        total_flow_teu, total_cost_usd, avg_weighted_delay, throughput_pct,
    )
    return result


# ── Convenience builder for the SentinelFlow sample network ──────────────────

def build_shanghai_reroute_graph() -> GraphData:
    """
    Construct a GraphData instance representing the three rerouting strategies
    defined in the SentinelFlow analytics.html Strategy Recommendations:

        SF-001  Shanghai → Busan (Rail) → Long Beach (Sea)
        SF-002  Split load via Mediterranean (Sea)
        SF-003  Direct Shanghai → Long Beach (increased West Coast capacity)

    Also includes a high-cost Air express lane for priority cargo.
    """
    return GraphData(
        disrupted_node_id="PORT-CN-SHA",
        min_throughput_pct=0.90,
        routes=[
            RouteOption(
                route_id         = "SHA→BSN-Rail→LGB-Sea",
                mode             = "Rail + Sea",
                cost_per_teu     = 2_230.0,
                delay_hours      = 528.0,    # 22 days (base 504h + 24h rail leg)
                risk_factor      = 0.12,
                capacity_teu     = 8_000,
                base_delay_hours = 504.0,
                carrier          = "Korea Rail Logistics + HMM",
            ),
            RouteOption(
                route_id         = "SHA→MED-Split-HAM",
                mode             = "Sea (Mediterranean split)",
                cost_per_teu     = 2_550.0,
                delay_hours      = 816.0,    # 34 days — Med disruption +24h
                risk_factor      = 0.28,
                capacity_teu     = 5_500,
                base_delay_hours = 792.0,
                carrier          = "Maersk / CMA CGM",
            ),
            RouteOption(
                route_id         = "SHA→LGB-Direct-CapEx",
                mode             = "Sea (West Coast capacity boost)",
                cost_per_teu     = 1_800.0,  # same lane, extra berth cost absorbed
                delay_hours      = 630.0,    # current congested time
                risk_factor      = 0.35,     # still passing through SHA
                capacity_teu     = 4_000,
                base_delay_hours = 504.0,
                carrier          = "COSCO",
            ),
            RouteOption(
                route_id         = "SHA→SIN-Sea→LGB-Air",
                mode             = "Sea + Air",
                cost_per_teu     = 9_200.0,
                delay_hours      = 192.0,    # 8 days sea + 2 days air
                risk_factor      = 0.06,
                capacity_teu     = 800,
                base_delay_hours = 168.0,
                carrier          = "OOCL + Singapore Airlines Cargo",
            ),
        ],
    )


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="SentinelFlow OR-Tools Rerouting Optimiser Demo"
    )
    parser.add_argument(
        "--cargo", type=int, default=12_000,
        help="Total cargo in TEU to redistribute (default: 12000)"
    )
    parser.add_argument(
        "--risk-weight", type=float, default=1.0,
        help="Weight applied to risk-delay cost term (default: 1.0)"
    )
    parser.add_argument(
        "--throughput", type=float, default=0.90,
        help="Minimum throughput fraction in [0,1] (default: 0.90)"
    )
    parser.add_argument(
        "--exact", action="store_true",
        help="Require exact demand satisfaction (no partial delivery)"
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  SentinelFlow — LP Rerouting Optimiser")
    print("═" * 60)
    print(f"  Disrupted node  : PORT-CN-SHA (Shanghai)")
    print(f"  Total cargo     : {args.cargo:,} TEU")
    print(f"  Risk weight     : {args.risk_weight}")
    print(f"  Throughput floor: {args.throughput * 100:.0f}%")
    print(f"  Exact demand    : {args.exact}")
    print("═" * 60 + "\n")

    graph = build_shanghai_reroute_graph()
    graph.min_throughput_pct = args.throughput

    print("  Route options:")
    for r in graph.routes:
        print(
            f"    [{r.mode:25s}]  cap={r.capacity_teu:>6,} TEU  "
            f"cost=${r.cost_per_teu:>7,.0f}/TEU  "
            f"delay={r.delay_hours:.0f}h  risk={r.risk_factor:.2f}  "
            f"composite_cost={r.composite_cost:.2f}"
        )
    print(f"\n  Total network capacity : {graph.total_capacity():,} TEU")
    print(f"  Minimum required flow  : {int(graph.min_throughput_pct * args.cargo):,} TEU\n")

    result = optimize_rerouting(
        graph_data        = graph,
        disrupted_node_id = "PORT-CN-SHA",
        total_cargo       = args.cargo,
        risk_weight       = args.risk_weight,
        allow_partial     = not args.exact,
    )

    print(result)

    # ── Sensitivity demo: vary risk_weight ─────────────────────────────────────
    if "--sensitivity" in sys.argv:
        print("\n  Sensitivity Analysis — varying risk_weight:\n")
        header = f"  {'risk_weight':>12}  {'total_cost':>14}  {'avg_delay':>10}  {'throughput':>12}"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for rw in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]:
            r = optimize_rerouting(graph, "PORT-CN-SHA", args.cargo,
                                   risk_weight=rw, allow_partial=not args.exact)
            if r.status in ("OPTIMAL", "FEASIBLE"):
                print(
                    f"  {rw:>12.2f}  ${r.total_cost_usd:>13,.2f}  "
                    f"{r.weighted_delay_hours:>9.1f}h  "
                    f"{r.throughput_pct:>11.1f}%"
                )
            else:
                print(f"  {rw:>12.2f}  {r.status}")
        print()
