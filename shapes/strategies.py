import json
import sys
import warnings
from collections import defaultdict, deque, namedtuple
from dataclasses import dataclass
from enum import Enum
from functools import partial
from operator import itemgetter
from pprint import pprint
from typing import List, Dict

import networkx as nx
import numpy as np
import pandas as pd
from ncls import NCLS
from ortools.graph import pywrapgraph
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model
from pyscipopt import (
    Model,
    quicksum,
    Conshdlr,
    SCIP_RESULT,
    SCIP_PRESOLTIMING,
    SCIP_PROPTIMING,
    SCIP_PARAMSETTING,
    SCIP_HEURTIMING,
    Heur,
    Branchrule,
    Eventhdlr,
    SCIP_RESULT,
    SCIP_EVENTTYPE,
    SCIP_PARAMSETTING,
)
from treed import TreeD

# from ortools.graph import pywrapgraph
# from ortools.linear_solver import pywraplp
# from ortools.sat.python import cp_model
from z3 import Optimize, Bools, Not, And

from shapes.plotting import make_memory_map


def valid_add(a, b):
    return a + b >= a


def valid_sub(a, b):
    return a >= b


# open interval overlap
def overlap(a, b, c, d):
    assert a <= b
    assert c <= d

    outer_len = max(b, d) - min(a, c)
    interval_len_1 = b - a
    interval_len_2 = d - c

    if not valid_add(interval_len_1, interval_len_2) or not valid_sub(
        outer_len, interval_len_1 + interval_len_2
    ):
        return True
    else:
        return False


@dataclass(repr=True, eq=True, order=True, unsafe_hash=True, frozen=False)
class LiveRange:
    begin: int
    end: int

    def __len__(self):
        return self.end - self.begin + 1

    def overlap(self, other):
        return overlap(self.begin, self.end + 1, other.begin, other.end + 1)


@dataclass(repr=True, eq=True, order=True, unsafe_hash=False, frozen=True)
class MemRegion:
    offset: int
    size: int

    def __len__(self):
        return self.size - self.offset + 1

    def overlap(self, other):
        return overlap(
            self.offset, self.next_free_addr, other.offset, other.next_free_addr
        )

    @property
    def next_free_addr(self):
        return self.offset + self.size


import hashlib


class RequiredAlloc:
    lvr: LiveRange
    size: int
    ptr_addr = 0

    def __init__(self, lvr, size, ptr_addr):
        self.lvr = lvr
        self.size = size
        self.ptr_addr = ptr_addr

    def __str__(self):
        return f"{self.lvr}:({self.size}, '{self.ptr_addr}')"

    def __hash__(self):
        return (
            int(
                hashlib.sha2
                ** np.random.randint(5, 10)(str(self).encode("utf-8")).hexdigest(),
                16,
            )
            % 10 ** 8
        )


def make_ncls_from_tuples(tups):
    return NCLS(
        np.array([t for (t, _, _) in tups], dtype=np.int64),
        np.array([t for (_, t, _) in tups], dtype=np.int64),
        np.array([t for (_, _, t) in tups], dtype=np.int64),
    )


def make_ncls_from_reqs(reqs: List[RequiredAlloc]):
    return NCLS(
        np.array([r.lvr.begin for r in reqs], dtype=np.int64),
        np.array([r.lvr.end for r in reqs], dtype=np.int64),
        np.array([r.size for r in reqs], dtype=np.int64),
    )


def make_df_from_reqs(reqs: List[RequiredAlloc]):
    begins = np.array([r.lvr.begin for r in reqs], dtype=np.int64)
    ends = np.array([r.lvr.end for r in reqs], dtype=np.int64)
    sizes = np.array([r.size for r in reqs], dtype=np.int64)
    lifetimes = np.array([len(r.lvr) for r in reqs], dtype=np.int64)
    lvr_index = pd.IntervalIndex.from_arrays(
        left=begins, right=ends, closed="both"
    ).set_names("live_range")
    df = pd.DataFrame(
        {"begin": begins, "end": ends, "mem_size": sizes, "lifetime": lifetimes}
    )
    df.index.rename("alloc_id", inplace=True)
    df = df.set_index(lvr_index, append=True)
    return df


@dataclass(eq=True)
class PlannedAlloc:
    lvr: LiveRange
    mem_region: MemRegion

    @classmethod
    def from_req_row(cls, req_row, offset: int):
        return PlannedAlloc(
            LiveRange(req_row.begin, req_row.end), MemRegion(offset, req_row.mem_size)
        )

    @classmethod
    def from_req(cls, req_alloc: RequiredAlloc, offset: int):
        return PlannedAlloc(req_alloc.lvr, MemRegion(offset, req_alloc.size))

    def __str__(self):
        return f"{self.lvr}:{self.mem_region}"

    def __repr__(self):
        return str(self)

    def overlap(self, other):
        return self.lvr.overlap(other.lvr) and self.mem_region.overlap(other.mem_region)


def make_interval_df_from_plans(reqs: List[PlannedAlloc]):
    lvrs = np.array([r.lvr for r in reqs], dtype=np.object)
    sizes = np.array([r.mem_region.size for r in reqs], dtype=np.int64)
    offsets = np.array([r.mem_region.offset for r in reqs], dtype=np.int64)
    index = pd.IntervalIndex.from_arrays(
        left=offsets, right=offsets + sizes, closed="left"
    )
    return pd.DataFrame({"lvr": lvrs}, index=index)


class GapPriority(Enum):
    SMALLEST = 1
    FIRST = 2


def find_gap(
    live_range: pd.Interval,
    req_row,
    current_allocs: pd.DataFrame,  # this one has offset index
    *,
    GAP_PRIORITY: GapPriority = GapPriority.SMALLEST,
):
    best_gap = float("inf")
    best_offset = None
    prev_offset = 0

    overlapping_allocs_idx = current_allocs.index.get_level_values(
        "live_range"
    ).overlaps(live_range)
    overlapping_allocs = current_allocs[overlapping_allocs_idx]

    for alloc_row in overlapping_allocs.itertuples():
        # offset_x will be ahead of the previous block
        # while prev_offset will be just in front
        # this looks for small gap ahead of a block
        gap = alloc_row.offset - prev_offset
        if req_row.mem_size <= gap < best_gap:
            best_offset = prev_offset
            if GAP_PRIORITY == GapPriority.FIRST:
                break
            best_gap = gap

        prev_offset = max(prev_offset, alloc_row.offset + alloc_row.mem_size)

    if best_offset is None:
        best_offset = prev_offset
    return best_offset


def get_new_sorted_order_mult_index(loc, item, orig_index):
    arr = np.asarray(orig_index)

    # Use Index constructor to ensure we get tuples cast correctly.
    item = pd.Index([item], dtype=orig_index.dtype)._values
    idx = np.concatenate((arr[:loc], item, arr[loc:]))
    return pd.Index(idx, name=pd.name)


def make_current_allocs_multi_index(begins, ends, offsets, sizes):
    live_range_index = pd.IntervalIndex.from_arrays(
        left=begins, right=ends, closed="both"
    ).set_names("live_range")
    mem_region_index = pd.IntervalIndex.from_arrays(
        left=offsets, right=np.array(offsets) + np.array(sizes), closed="left"
    ).set_names("mem_region")
    index = pd.MultiIndex(
        levels=[mem_region_index, live_range_index],
        codes=[np.arange(len(begins)), np.arange(len(begins))],
        sortorder=0,
        names=["mem_region", "live_range"],
    )
    return index


def make_alloc_df(begin, end, offset, size, alloc_id):
    index = make_current_allocs_multi_index([begin], [end], [offset], [size])
    alloc_df = pd.DataFrame(
        data={
            "begin": begin,
            "end": end,
            "offset": offset,
            "mem_size": size,
            "alloc_id": alloc_id,
        },
        columns=["begin", "end", "offset", "mem_size", "alloc_id"],
        index=index,
    )
    alloc_df = alloc_df.set_index("alloc_id", append=True)
    return alloc_df


def _greedy_by_size(
    sorted_req_mem_allocs: pd.DataFrame,
    gap_finder,
):
    first_alloc = sorted_req_mem_allocs.iloc[0]
    current_allocs = make_alloc_df(
        first_alloc.begin, first_alloc.end, 0, first_alloc.mem_size, first_alloc.name[0]
    )

    inorder_of_decision_allocs: List[PlannedAlloc] = []
    for req_row in sorted_req_mem_allocs.iloc[1:].itertuples(name="RequiredAlloc"):
        alloc_id, live_range, _ = req_row.Index
        best_offset = gap_finder(live_range, req_row, current_allocs)

        p = PlannedAlloc.from_req_row(req_row, best_offset)
        inorder_of_decision_allocs.append(p)
        new_mem_region = pd.Interval(
            p.mem_region.offset, p.mem_region.offset + p.mem_region.size, closed="left"
        )
        insert_idx = current_allocs.index.get_level_values("mem_region").searchsorted(
            new_mem_region
        )
        p_df = make_alloc_df(
            p.lvr.begin, p.lvr.end, p.mem_region.offset, p.mem_region.size, alloc_id
        )

        current_allocs = pd.concat(
            [current_allocs.iloc[:insert_idx], p_df, current_allocs.iloc[insert_idx:]]
        )

    return inorder_of_decision_allocs


def greedy_by_size(
    req_mem_allocs: pd.DataFrame,
    *,
    gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.SMALLEST),
):
    print("greedy by size", file=sys.stderr)
    # biggest size first but break ties deterministically
    req_mem_allocs = req_mem_allocs.set_index(
        "mem_size", drop=False, append=True
    ).sort_index(level="mem_size", ascending=False)
    return _greedy_by_size(req_mem_allocs, gap_finder)


def greedy_by_longest(
    req_mem_allocs: pd.DataFrame,
    *,
    gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.SMALLEST),
):
    print("greedy by longest", file=sys.stderr)
    req_mem_allocs = req_mem_allocs.set_index(
        "lifetime", drop=False, append=True
    ).sort_index(level="lifetime", ascending=False)
    return _greedy_by_size(req_mem_allocs, gap_finder)


def save_planned_allocs(allocs: List[PlannedAlloc], name):
    json.dump(
        [str(a) for a in allocs], open(f"planned_allocs/{name}", "w"), indent=True
    )


MemEvent = namedtuple("MemEvent", "ptr_addr size ts")


def solve_z3():
    a, b, c = Bools("a b c")
    o = Optimize()
    o.add(a == c)
    o.add(Not(And(a, b)))
    o.add_soft(a, 2)
    o.add_soft(b, 3)
    o.add_soft(c, 1)
    print(o.check())
    print(o.model())


def make_mem_events_from_required_allocs(reqs: List[RequiredAlloc]) -> List[MemEvent]:
    mem_events = {}
    for req in reqs:
        mem_events[req.lvr.begin] = MemEvent(req.ptr_addr, req.size, req.lvr.begin)
        mem_events[req.lvr.end] = MemEvent(req.ptr_addr, -req.size, req.lvr.end)

    return [m[1] for m in sorted(mem_events.items(), key=itemgetter(0))]


def bump_allocator(req_mem_allocs: List[RequiredAlloc]) -> List[PlannedAlloc]:
    mem_events = make_mem_events_from_required_allocs(req_mem_allocs)
    print("bump_allocator", file=sys.stderr)
    mem_events.sort(key=lambda r: r.ts)
    planned_allocations: Dict[str, PlannedAlloc] = {}
    next_offset = 0
    curr_allocs = 0
    for ptr_addr, size, ts in mem_events:
        if size > 0:
            planned_allocations[ptr_addr] = PlannedAlloc(
                LiveRange(ts, -1), MemRegion(next_offset, size)
            )
            next_offset += size
            curr_allocs += 1
        elif size < 0:
            assert ptr_addr in planned_allocations
            p_alloc = planned_allocations[ptr_addr]
            assert p_alloc.mem_region.size == -size
            assert p_alloc.lvr.begin < ts
            planned_allocations[ptr_addr] = PlannedAlloc(
                LiveRange(p_alloc.lvr.begin, ts), p_alloc.mem_region
            )
            curr_allocs -= 1
        if curr_allocs == 0:
            next_offset = 0

    return list(planned_allocations.values())


def first_available(color_list):
    """Return smallest non-negative integer not in the given list of colors."""
    color_set = set(color_list)
    count = 0
    while True:
        if count not in color_set:
            return count
        count += 1


def greedy_color(G, order):
    color = dict()
    for node in order:
        used_neighbour_colors = [color[nbr] for nbr in G[node] if nbr in color]
        color[node] = first_available(used_neighbour_colors)
    return color


def tuple_intersection(A, B):
    _nrows, ncols = A.shape
    assert ncols == 2
    dtype = {
        "names": ["f{}".format(i) for i in range(ncols)],
        "formats": ncols * [A.dtype],
    }
    C = np.intersect1d(A.view(dtype).T[0], B.view(dtype).T[0])
    return C.T.view(A.dtype).T.reshape(-1, ncols)


def tuple_set_diff(A, B):
    _nrows, ncols = A.shape
    assert ncols == 2
    dtype = {
        "names": ["f{}".format(i) for i in range(ncols)],
        "formats": ncols * [A.dtype],
    }
    C = np.setdiff1d(A.view(dtype).T[0], B.view(dtype).T[0])
    return C.T.view(A.dtype).T.reshape(-1, ncols)


def gergov(required_allocs: pd.DataFrame) -> List[PlannedAlloc]:
    w = [0]
    l = [required_allocs.begin.min()]
    r = [required_allocs.end.max()]
    lvr_index = pd.IntervalIndex.from_arrays(left=l, right=r, closed="both").set_names(
        "live_range"
    )
    H = pd.DataFrame(
        data={
            "w": w,
            "l": l,
            "r": r,
        },
        columns=["w", "l", "r"],
        index=lvr_index,
    )

    s = required_allocs.mem_size.values
    r = required_allocs.begin.values
    c = required_allocs.end.values
    lvr_index = pd.IntervalIndex.from_arrays(left=r, right=c, closed="both").set_names(
        "live_range"
    )
    J = pd.DataFrame(
        data={
            "s": s,
            "r": r,
            "c": c,
        },
        columns=["s", "r", "c"],
        index=lvr_index,
    )

    def if_there_exists(pick_idx):
        # without sorting by start of interval this does poorly
        JJ = J[J.index.overlaps(pick_idx)]
        if JJ.empty:
            return None
        if H.empty:
            return next(JJ.itertuples())

        for row in JJ.itertuples():
            # there exists row that doesn't overlap with any in H
            if H.index.overlaps(row.Index).sum() == 0:
                return row
        return None

    alphap = {}
    V = []
    while not J.empty:
        pick_idx = H["w"].idxmin()
        (w, xl, xr) = H.loc[pick_idx]
        H = H.drop(pick_idx)
        j = if_there_exists(pick_idx)
        if j is not None:
            s, r, c = j.s, j.r, j.c
            J = J.drop(j.Index)
            V.append(j)
            alphap[j.Index] = w

            ma, mi = max(xl, r), min(c, xr)
            H = H.append(
                pd.DataFrame(
                    data={"w": w + s, "l": ma, "r": mi},
                    index=pd.IntervalIndex([pd.Interval(ma, mi, closed="both")]),
                    columns=H.columns,
                )
            )

            if xl < r:
                H = H.append(
                    pd.DataFrame(
                        data={"w": w, "l": xl, "r": r},
                        index=pd.IntervalIndex([pd.Interval(xl, r, closed="both")]),
                        columns=H.columns,
                    )
                )
            if c < xr:
                H = H.append(
                    pd.DataFrame(
                        data={"w": w, "l": c, "r": xr},
                        index=pd.IntervalIndex([pd.Interval(c, xr, closed="both")]),
                        columns=H.columns,
                    )
                )

    VV = J.index.to_frame().append(V).dropna(axis=1).set_index("Index")
    VV.index = VV.index.rename("live_range")
    alphapp = {
        v.Index: pd.Interval(alphap[v.Index], alphap[v.Index] + v.s, "left")
        for v in VV.itertuples()
    }
    live_range = pd.IntervalIndex(list(alphapp.keys()), name="live_range")
    mem_region = pd.IntervalIndex(list(alphapp.values()), name="mem_region")
    live_range_ncls = NCLS(
        starts=live_range.left.values,
        ends=(live_range.right + 1).values,
        ids=np.arange(live_range.size),
    )
    mem_region_ncls = NCLS(
        starts=mem_region.left.astype(np.int64),
        ends=mem_region.right.astype(np.int64),
        ids=np.arange(mem_region.size),
    )
    live_range_edge_list = np.array(
        live_range_ncls.all_overlaps_both(
            live_range.left.values,
            (live_range.right + 1).values,
            np.arange(live_range.size),
        )
    )
    live_range_edge_list = live_range_edge_list.T.astype(
        order="C", dtype=live_range_edge_list.dtype
    )
    mem_region_edge_list = np.array(
        mem_region_ncls.all_overlaps_both(
            mem_region.left.values.astype(np.int64),
            mem_region.right.values.astype(np.int64),
            np.arange(mem_region.size),
        )
    )
    mem_region_edge_list = mem_region_edge_list.T.astype(
        order="C", dtype=mem_region_edge_list.dtype
    )

    _E = tuple_intersection(live_range_edge_list, mem_region_edge_list)

    E = defaultdict(list)
    for u, v in _E:
        E[live_range[u]].append(live_range[v])

    color = greedy_color(E, V)
    max_j = max([alphap[itrvl] + s for itrvl, s, r, c in V])
    alpha = {}
    for (itrvl, s, r, c), cc in color.items():
        alpha[s, r, c] = alphap[itrvl] + cc * max_j

    planned_allocs = []
    for (s, r, c), w in alpha.items():
        lvr = LiveRange(r, c)
        mem_region = MemRegion(w, s)
        planned_allocs.append(PlannedAlloc(lvr, mem_region))
    return planned_allocs


def shortest_path_faster(E, W, s):
    d = {}
    for g in E:
        d[g] = float("inf")
    d[s] = 0
    Q = deque([s])
    while Q:
        u = Q.popleft()
        for v in E[u]:
            if d[u] + W[u, v] < d[v]:
                d[v] = d[u] + W[u, v]
                if v not in Q:
                    Q.append(v)


import matplotlib.pyplot as plt


def draw_nx(G):
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(
        G,
        pos,
        edge_color="black",
        width=1,
        linewidths=1,
        node_size=500,
        node_color="pink",
        alpha=0.9,
        labels={node: node for node in G.nodes()},
    )
    edge_labels = {}
    for u, vs in G.adj.items():
        for v in vs:
            edge_labels[u, v] = f"{vs[v]['weight']}, {vs[v]['capacity']}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
    plt.axis("off")
    plt.show()


def ortools_mincost_flow(required_allocs: pd.DataFrame):
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    S = "S"
    T = "T"
    node_to_idx = {S: 0, T: 1}
    idx_to_node = {0: S, 1: T}
    edges_strs = {}
    for row in required_allocs.itertuples():
        L = f"L{row.Index[0]}"
        node_to_idx[L] = len(node_to_idx)
        idx_to_node[node_to_idx[L]] = L
        R = f"R{row.Index[0]}"
        node_to_idx[R] = len(node_to_idx)
        idx_to_node[node_to_idx[R]] = R

        edges_strs[S, R] = (row.mem_size, 1)
        edges_strs[S, L] = (0, 1)
        edges_strs[R, T] = (0, 1)

    ids = np.arange(len(required_allocs))
    live_range_ncls = NCLS(
        starts=required_allocs.begin.values,
        ends=required_allocs.end.values + 1,
        ids=ids,
    )
    edge_list = np.array(
        live_range_ncls.all_overlaps_both(
            starts=required_allocs.begin.values,
            ends=required_allocs.end.values + 1,
            indexes=ids,
        )
    )
    edge_list = edge_list.T.astype(order="C", dtype=edge_list.dtype)
    all_pairs = np.array(np.meshgrid(ids, ids)).T.reshape(-1, 2)
    non_overlap = tuple_set_diff(all_pairs, edge_list)
    directed_edges = non_overlap[non_overlap[:, 0] < non_overlap[:, 1]]

    us = directed_edges[:, 0]
    vs = directed_edges[:, 1]

    L = np.array(["L"])
    R = np.array(["R"])
    Ls = L.astype(object) + us.astype(str)
    Rs = R.astype(object) + vs.astype(str)

    diff = (
        required_allocs.mem_size.iloc[vs].values
        - required_allocs.mem_size.iloc[us].values
    )
    diff[diff < 0] = 0
    costs_per_edge = np.array([Ls, Rs, diff]).T
    for L, R, c in costs_per_edge:
        edges_strs[L, R] = (c, 1)

    for (L, R), (w, c) in edges_strs.items():
        start, end = node_to_idx[L], node_to_idx[R]
        min_cost_flow.AddArcWithCapacityAndUnitCost(start, end, c, w)

    # Add node supply.
    min_cost_flow.SetNodeSupply(node_to_idx[S], len(required_allocs))
    min_cost_flow.SetNodeSupply(node_to_idx[T], -len(required_allocs))
    for idx, node in idx_to_node.items():
        if node in {S, T}:
            continue
        min_cost_flow.SetNodeSupply(idx, 0)

    status = min_cost_flow.Solve()

    if status != min_cost_flow.OPTIMAL:
        warnings.warn("There was an issue with the min cost flow input.")
        warnings.warn(f"Status: {status}")
        return []

    flow_dict = defaultdict(dict)
    for i in range(min_cost_flow.NumArcs()):
        cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
        u, v = idx_to_node[min_cost_flow.Tail(i)], idx_to_node[min_cost_flow.Head(i)]
        flow = min_cost_flow.Flow(i)
        capacity = min_cost_flow.Capacity(i)
        flow_dict[u][v] = flow

    total_flow = 0
    allocs = {}
    reuses = {}

    for v, flow in flow_dict[S].items():
        if v[0] == R and flow == 1:
            total_flow += 1
            allocs[v[1:]] = required_allocs.iloc[int(v[1:])].mem_size

    for u, vs in flow_dict.items():
        if u[0] != L:
            continue
        for v, flow in vs.items():
            if v[0] != R or flow != 1:
                continue

            shared_object = u[1:]
            while shared_object not in allocs:
                shared_object = reuses[shared_object]
            reuses[v[1:]] = shared_object

            avail = allocs[shared_object]
            necessary = required_allocs.iloc[int(v[1:])].mem_size
            if avail < necessary:
                allocs[shared_object] = necessary

    planned_allocs = []
    planned_allocs_dict = {}
    offset = 0
    for alloc, size in allocs.items():
        req = required_allocs.iloc[int(alloc)]
        lvr = LiveRange(req.begin, req.end)
        mem_region = MemRegion(offset, size)
        offset += size
        pl_alloc = PlannedAlloc(lvr, mem_region)
        planned_allocs.append(pl_alloc)
        planned_allocs_dict[alloc] = pl_alloc

    for tens, shared_obj in reuses.items():
        req = required_allocs.iloc[int(tens)]
        lvr = LiveRange(req.begin, req.end)
        avail = planned_allocs_dict[shared_obj]
        assert not lvr.overlap(avail.lvr)
        assert req.mem_size <= avail.mem_region.size, (
            req.mem_size,
            avail.mem_region.size,
        )

        reuse_start = planned_allocs_dict[shared_obj].mem_region.offset
        mem_region = MemRegion(reuse_start, req.mem_size)
        planned_allocs.append(PlannedAlloc(lvr, mem_region))

    return planned_allocs


def mincost_flow(required_allocs: pd.DataFrame):
    G = nx.DiGraph()
    G.add_node("s", demand=-len(required_allocs))
    G.add_node("t", demand=len(required_allocs))

    for row in required_allocs.itertuples():
        G.add_node(f"l,{row.Index[0]}", demand=0)
        G.add_node(f"r,{row.Index[0]}", demand=0)

        G.add_edge("s", f"r,{row.Index[0]}", weight=row.mem_size, capacity=1)
        G.add_edge("s", f"l,{row.Index[0]}", weight=0, capacity=1)
        G.add_edge(f"r,{row.Index[0]}", "t", weight=0, capacity=1)

    ids = np.arange(len(required_allocs))
    live_range_ncls = NCLS(
        starts=required_allocs.begin.values,
        ends=required_allocs.end.values + 1,
        ids=ids,
    )
    edge_list = np.array(
        live_range_ncls.all_overlaps_both(
            starts=required_allocs.begin.values,
            ends=required_allocs.end.values + 1,
            indexes=ids,
        )
    )
    edge_list = edge_list.T.astype(order="C", dtype=edge_list.dtype)
    all_pairs = np.array(np.meshgrid(ids, ids)).T.reshape(-1, 2)
    non_overlap = tuple_set_diff(all_pairs, edge_list)
    directed_edges = non_overlap[non_overlap[:, 0] < non_overlap[:, 1]]

    l = np.array(["l,"])
    r = np.array(["r,"])
    us = directed_edges[:, 0]
    vs = directed_edges[:, 1]
    ls = l.astype(object) + us.astype(np.str)
    rs = r.astype(object) + vs.astype(np.str)
    diff = (
        required_allocs.mem_size.iloc[vs].values
        - required_allocs.mem_size.iloc[us].values
    )
    diff[diff < 0] = 0
    diffs = [{"weight": d, "capacity": 1} for d in diff]
    weighted_edges = np.array([ls, rs, diffs]).T
    G.add_edges_from(weighted_edges)

    flow_dict = nx.min_cost_flow(G)
    total_flow = 0
    allocs = {}
    reuses = {}

    for v, flow in flow_dict["s"].items():
        if v[0] == "r" and flow == 1:
            total_flow += 1
            allocs[v[2:]] = required_allocs.iloc[int(v[2:])].mem_size

    for u, vs in flow_dict.items():
        if u[0] != "l":
            continue
        for v, flow in vs.items():
            if v[0] != "r" or flow != 1:
                continue

            shared_object = u[2:]
            while shared_object not in allocs:
                shared_object = reuses[shared_object]
            reuses[v[2:]] = shared_object

            avail = allocs[shared_object]
            necessary = required_allocs.iloc[int(v[2:])].mem_size
            if avail < necessary:
                allocs[shared_object] = necessary

    planned_allocs = []
    planned_allocs_dict = {}
    offset = 0
    for alloc, size in allocs.items():
        req = required_allocs.iloc[int(alloc)]
        lvr = LiveRange(req.begin, req.end)
        mem_region = MemRegion(offset, size)
        offset += size
        pl_alloc = PlannedAlloc(lvr, mem_region)
        planned_allocs.append(pl_alloc)
        planned_allocs_dict[alloc] = pl_alloc

    for tens, shared_obj in reuses.items():
        req = required_allocs.iloc[int(tens)]
        lvr = LiveRange(req.begin, req.end)
        avail = planned_allocs_dict[shared_obj]
        assert not lvr.overlap(avail.lvr)
        assert req.mem_size <= avail.mem_region.size, (
            req.mem_size,
            avail.mem_region.size,
        )

        reuse_start = planned_allocs_dict[shared_obj].mem_region.offset
        mem_region = MemRegion(reuse_start, req.mem_size)
        planned_allocs.append(PlannedAlloc(lvr, mem_region))

    return planned_allocs


def solve_csp(required_allocs: pd.DataFrame):
    model = cp_model.CpModel()

    max_size = sum(r.mem_size for r in required_allocs.itertuples())

    live_ranges = []
    offsets = []
    offsets_plus_sizes = []
    regions = []
    for row in required_allocs.itertuples():
        live_range = model.NewIntervalVar(
            row.begin,
            row.end + 1 - row.begin,
            row.end + 1,
            "live_range_%i" % row.Index[0],
        )
        live_ranges.append(live_range)

        offset = model.NewIntVar(0, max_size * 2, "offset_%i" % row.Index[0])
        offset_plus_size = model.NewIntVar(
            0, max_size * 2, "offset_plus_size_%i" % row.Index[0]
        )
        region = model.NewIntervalVar(
            offset, row.mem_size, offset_plus_size, "region_%i" % row.Index[0]
        )
        # model.Add(offset + size == offset_plus_size)

        offsets.append(offset)
        offsets_plus_sizes.append(offset_plus_size)
        regions.append(region)

    # Main constraint.
    model.AddNoOverlap2D(live_ranges, regions)

    total_size = model.NewIntVar(0, max_size * 2, "u")
    model.AddMaxEquality(total_size, offsets_plus_sizes)
    model.Minimize(total_size)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 1000.0
    status = solver.Solve(model)

    # # https://github.com/google/or-tools/blob/stable/ortools/sat/doc/model.md#model-copy
    # copy = cp_model.CpModel()
    # copy.CopyFrom(model)

    if status == cp_model.OPTIMAL:
        res = []
        for row in required_allocs.itertuples():
            res.append(
                PlannedAlloc.from_req_row(row, solver.Value(offsets[row.Index[0]]))
            )
        return res
    else:
        warnings.warn("csp: no solution")
        return []


def solve_mip(required_allocs: pd.DataFrame):
    ids = np.arange(len(required_allocs))
    live_range_ncls = NCLS(
        starts=required_allocs.begin.values,
        ends=required_allocs.end.values + 1,
        ids=ids,
    )
    edge_list = np.array(
        live_range_ncls.all_overlaps_both(
            starts=required_allocs.begin.values,
            ends=required_allocs.end.values + 1,
            indexes=ids,
        )
    )
    edge_list = sorted(
        [
            (u, v)
            for u, v in edge_list.T.astype(order="C", dtype=edge_list.dtype)
            if u < v
        ]
    )

    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SetSolverSpecificParametersAsString("display/verblevel = 5")
    solver.SetSolverSpecificParametersAsString("display/freq = 1")
    # solver.SetSolverSpecificParametersAsString("display/lpinfo = TRUE")
    solver.EnableOutput()

    max_mem = sum(r.mem_size for r in required_allocs.itertuples())

    total_mem = solver.IntVar(0.0, max_mem, "total_mem")
    offsets = []
    for i, row in enumerate(required_allocs.itertuples()):
        offset = solver.IntVar(0.0, max_mem, f"offset_{row.Index[0]}")
        # offset_i + mem_i <= total_mem
        solver.Add(offset + row.mem_size <= total_mem)
        offsets.append(offset)

    # we encode the non-overlapping constraints using ordering of the blocks
    # if two blocks overlap then their allocations must be ordered
    # z_ij are decision variable. z_ij = 0 if block i has a lower
    # offset than block j i.e. offset_i + mem_i <= offset_j
    # and z_ij = 1 if the converse offset_j + mem_j <= offset_i
    # (note there could be a gap if we stick a block in between them
    for i, j in edge_list:
        inters = solver.IntVar(0, 1, f"z_{{{i},{j}}}")
        # if z_ij = 0 then i < j then offsets[i] + mems[i] <= offsets[j]
        # but if z_ij = 1 then j < i then offsets[i] + mems[i] <= offsets[j] + max_mem
        # otherwise the signs wouldn't be right
        solver.Add(
            offsets[i] + required_allocs.iloc[i].mem_size
            <= offsets[j] + inters * max_mem
        )
        # conversely here
        solver.Add(
            offsets[j] + required_allocs.iloc[j].mem_size
            <= offsets[i] + (1 - inters) * max_mem
        )

    # Minimize u
    solver.Minimize(total_mem)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        return [
            PlannedAlloc.from_req_row(row, offsets[row.Index[0]].solution_value())
            for row in required_allocs.itertuples()
        ]
    else:
        warnings.warn("mip: The problem does not have an optimal solution.")
        return []


def print_lp(model):
    lp_rows = model.getLPRowsData()
    row_names = [r.name for r in lp_rows]
    lp_cols = model.getLPColsData()
    tabl = np.empty((len(lp_rows), len(lp_cols)), dtype=float)
    col_name_to_idx = {}
    for row in lp_rows:
        row_idx = row.getLPPos()
        vals = row.getVals()
        for i, col in enumerate(row.getCols()):
            col_idx = col.getLPPos()
            var_name = col.getVar().name
            if var_name not in col_name_to_idx:
                col_name_to_idx[var_name] = col_idx
            else:
                assert col_name_to_idx[var_name] == col_idx

            tabl[row_idx, col_idx] = vals[i]

    tabl[tabl > 1e20] = float("inf")
    df = pd.DataFrame(
        tabl.round(decimals=10),
        columns=[k for k, v in sorted(col_name_to_idx.items(), key=lambda x: x[1])],
        index=row_names,
    )
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.precision",
        3,
        "display.float_format",
        lambda x: "%.3f" % x,
    ):
        print(df)


events_types_ = dict(
    BESTSOLFOUND=67108864,
    DISABLED=0,
    FIRSTLPSOLVED=8388608,
    GHOLEADDED=2048,
    GHOLEREMOVED=4096,
    GLBCHANGED=32,
    GUBCHANGED=64,
    IMPLADDED=32768,
    LBRELAXED=256,
    LBTIGHTENED=128,
    LHOLEADDED=8192,
    LHOLEREMOVED=16384,
    LPEVENT=25165824,
    LPSOLVED=16777216,
    NODEBRANCHED=2097152,
    NODEFEASIBLE=524288,
    NODEFOCUSED=262144,
    NODEINFEASIBLE=1048576,
    NODESOLVED=3670016,
    OBJCHANGED=16,
    POORSOLFOUND=33554432,
    PRESOLVEROUND=131072,
    ROWADDEDLP=536870912,
    ROWADDEDSEPA=134217728,
    ROWCOEFCHANGED=2147483648,
    ROWCONSTCHANGED=4294967296,
    ROWDELETEDLP=1073741824,
    ROWDELETEDSEPA=268435456,
    ROWSIDECHANGED=8589934592,
    SYNC=17179869184,
    UBRELAXED=1024,
    UBTIGHTENED=512,
    VARADDED=1,
    VARDELETED=2,
    VARFIXED=4,
    VARUNLOCKED=8,
)

events_types = {v: k for k, v in events_types_.items()}


class NodeEventHandler(Eventhdlr):
    def __init__(self):
        self.calls = []

    def eventinit(self):
        for evt_type in events_types:
            self.model.catchEvent(evt_type, self)

    def eventexit(self):
        for evt_type in events_types:
            self.model.dropEvent(evt_type, self)

    def eventexec(self, event):
        print(events_types[event.getType()])
        print(self.model.getNLPRows(), self.model.getNLPCols())
        if (
            self.model.getPrimalbound() == self.model.getDualbound()
            and self.model.getNLPRows()
        ):
            print_lp(self.model)
        # self.calls.append("eventexec")
        # assert event.getType() == SCIP_EVENTTYPE.NODEFOCUSED
        node = event.getNode()

        # if node.getDepth() == 0:
        #     assert node.getParent() is None
        #     assert node.getParentBranchings() is None
        #     return
        #
        # variables, branchbounds, boundtypes = node.getParentBranchings()
        # assert len(variables) == 1
        # assert len(branchbounds) == 1
        # assert len(boundtypes) == 1
        # domain_changes = node.getDomchg()
        # bound_changes = domain_changes.getBoundchgs()
        # assert len(bound_changes) == 1


def solve_mip_scip(required_allocs: pd.DataFrame):
    ids = np.arange(len(required_allocs))
    live_range_ncls = NCLS(
        starts=required_allocs.begin.values,
        ends=required_allocs.end.values + 1,
        ids=ids,
    )
    edge_list = np.array(
        live_range_ncls.all_overlaps_both(
            starts=required_allocs.begin.values,
            ends=required_allocs.end.values + 1,
            indexes=ids,
        )
    )
    edge_list = sorted(
        [
            (u, v)
            for u, v in edge_list.T.astype(order="C", dtype=edge_list.dtype)
            if u < v
        ]
    )
    solver = Model("memory_planning")
    solver.setPresolve(SCIP_PARAMSETTING.OFF)
    # solver.hideOutput()
    node_eventhdlr = NodeEventHandler()
    solver.includeEventhdlr(
        node_eventhdlr, "NodeEventHandler", "python event handler to catch NODEFOCUSED"
    )
    max_mem = sum(r.mem_size for r in required_allocs.itertuples())
    total_mem = solver.addVar(vtype="I", lb=0, ub=max_mem, name="total_mem")

    offsets = []
    for i, row in enumerate(required_allocs.itertuples()):
        offset = solver.addVar(
            vtype="I", lb=0, ub=max_mem, name=f"offset_{row.Index[0]}"
        )
        # offset_i + mem_i <= total_mem
        solver.addCons(offset + row.mem_size <= total_mem)
        offsets.append(offset)

    # we encode the non-overlapping constraints using ordering of the blocks
    # if two blocks overlap then their allocations must be ordered
    # z_ij are decision variable. z_ij = 0 if block i has a lower
    # offset than block j i.e. offset_i + mem_i <= offset_j
    # and z_ij = 1 if the converse offset_j + mem_j <= offset_i
    # (note there could be a gap if we stick a block in between them
    for i, j in edge_list:
        inters = solver.addVar(vtype="B", lb=0, ub=1, name=f"z_{{{i},{j}}}")
        # if z_ij = 0 then i < j then offsets[i] + mems[i] <= offsets[j]
        # but if z_ij = 1 then j < i then offsets[i] + mems[i] <= offsets[j] + max_mem
        # otherwise the signs wouldn't be right
        solver.addCons(
            offsets[i] + required_allocs.iloc[i].mem_size
            <= offsets[j] + inters * max_mem
        )
        # conversely here
        solver.addCons(
            offsets[j] + required_allocs.iloc[j].mem_size
            <= offsets[i] + (1 - inters) * max_mem
        )

    # Minimize u
    solver.setObjective(total_mem, "minimize")

    # treed = TreeD(scip_model=solver, nodelimit=2000, showcuts=True)
    # treed.solve()
    # fig = treed.draw()
    # fig.show(renderer='notebook')
    # fig.show()

    solver.optimize()

    if solver.getStatus() == "optimal":
        return [
            PlannedAlloc.from_req_row(row, solver.getVal(offsets[row.Index[0]]))
            for row in required_allocs.itertuples()
        ]
    else:
        warnings.warn("mip: The problem does not have an optimal solution.")
        return []


# def greedy_by_breadth(mem_events_per_op):
#     mem_events_per_op_ls = list(mem_events_per_op.items())
#     mem_events_per_op_ls.sort(key=lambda x: sum([y[1] for y in x[1]]), reverse=True)
#
#     mem_allocs = []
#     for _, mem_events in mem_events_per_op_ls:
#         mem_events.sort(key=lambda x: x[1], reverse=True)
#         mem_allocs.extend(mem_events)
#
#     ordered_allocs = []
#     inorder_of_decision_allocs = []
#     total_consumption = 0
#     for record in mem_allocs:
#         best_offset = find_gap(record, ordered_allocs)
#
#         (begin_t, end_t), size_t = record
#         total_consumption = max(total_consumption, best_offset + size_t)
#
#         inorder_of_decision_allocs.append(((begin_t, end_t), (best_offset, size_t)))
#         ordered_allocs.append(((begin_t, end_t), (best_offset, size_t)))
#         ordered_allocs.sort(key=lambda x: x[1][0])
#
#     return inorder_of_decision_allocs, total_consumption


# import memory_planning
#
#
# def memory_planning_cpp(
#         required_allocs: pd.DataFrame, strategy: memory_planning.Strategy
# ):
#     sorted_live_range_map = {}
#     for row in required_allocs.itertuples():
#         lvr = memory_planning.LiveRange(row.begin, row.end)
#         ulvr = memory_planning.UniqueLiveRange(lvr, str(row.Index[0]))
#         size = row.mem_size
#         sorted_live_range_map[ulvr] = size
#
#     return memory_planning.planMemory(sorted_live_range_map, strategy)


def calculate_high_watermark(allocs: List[PlannedAlloc]):
    allocs.sort(key=lambda p: p.lvr.begin)
    peak = 0
    for a in allocs:
        peak = max(peak, a.mem_region.next_free_addr)
    return int(peak)


def verify_allocation(allocations):
    for i, alloc1 in enumerate(allocations):
        for j, alloc2 in enumerate(allocations):
            if i == j:
                continue
            if alloc1.overlap(alloc2):
                # print("invalid", alloc1, alloc2)
                return False
    return True


def test(req_mem_allocs: pd.DataFrame):
    print("LSTMGreedyBySizeWithSmallestGap")
    res = greedy_by_size(
        req_mem_allocs,
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.SMALLEST),
    )
    assert verify_allocation(res)
    print(calculate_high_watermark(res))

    print("LSTMGreedyBySizeWithFirstGap")
    res = greedy_by_size(
        req_mem_allocs,
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST),
    )
    assert verify_allocation(res)
    print(calculate_high_watermark(res))

    print("LSTMGreedyByLongestAndSizeWithSmallestGap")
    res = greedy_by_longest(
        req_mem_allocs,
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.SMALLEST),
    )
    assert verify_allocation(res)
    print(calculate_high_watermark(res))

    print("LSTMGreedyByLongestAndSizeWithFirstGap")
    res = greedy_by_longest(
        req_mem_allocs,
        gap_finder=partial(find_gap, GAP_PRIORITY=GapPriority.FIRST),
    )
    assert verify_allocation(res)
    print(calculate_high_watermark(res))

    print("csp")
    res = solve_csp(req_mem_allocs)
    assert verify_allocation(res)
    print(calculate_high_watermark(res))

    print("mip")
    res = solve_mip(req_mem_allocs)
    assert verify_allocation(res)
    print(calculate_high_watermark(res))

    print("gergov")
    res = gergov(req_mem_allocs)
    assert verify_allocation(res)
    print(calculate_high_watermark(res))

    print("mincost_flow")
    res = mincost_flow(req_mem_allocs)
    assert verify_allocation(res)
    print(calculate_high_watermark(res))


def test_resnet18():
    # from profile_models import get_required_mem_allocs

    trace_json = json.load(open("traces/resnet18.1x3x100x100.json"))
    req_mem_allocs = get_required_mem_allocs(trace_json)
    req_mem_allocs = make_df_from_reqs(req_mem_allocs)
    res = gergov(req_mem_allocs)
    assert verify_allocation(res)


def test_lstm():
    lvrs = {
        # (0, 3): 1024,
        # (1, 3): 1024,
        # (2, 4): 1024,
        (5, 10): 2 ** np.random.randint(1, 5),
        (6, 14): 2 ** np.random.randint(1, 5),
        (7, 10): 2 ** np.random.randint(1, 5),
        (8, 10): 2 ** np.random.randint(1, 5),
        (8, 12): 2 ** np.random.randint(1, 5),
        (9, 12): 2 ** np.random.randint(1, 5),
        (13, 14): 2 ** np.random.randint(1, 5),
    }
    # lvrs = {}
    # for _ in range(7):
    #     start = np.random.randint(1, 20)
    #     end = np.random.randint(start + 1, 21)
    #     lvrs[start, end] = 2 ** np.random.randint(1, 5)
    #
    # lvrs = {}
    # for start, end in [(5, 14), (13, 14), (12, 19), (3, 14), (3, 7), (12, 20), (2, 4)]:
    #     lvrs[start, end] = 2 ** np.random.randint(1, 5)

    req_mem_allocs = make_df_from_reqs(
        [
            RequiredAlloc(LiveRange(begin, end), size, str(i))
            for i, ((begin, end), size) in enumerate(lvrs.items())
        ]
    )
    res = solve_mip_scip(req_mem_allocs)
    res.sort(key=lambda r: r.mem_region.offset)
    pprint([(r.lvr.begin, r.lvr.end) for r in res])
    mip_high_water_mark = res[-1].mem_region.offset + res[-1].mem_region.size
    fig = make_memory_map(res, "mip", save=False, rescale_to_mb=False)
    fig.show()

    # res = solve_csp(req_mem_allocs)
    # res.sort(key=lambda r: r.mem_region.offset)
    # # pprint(res)
    # csp_high_water_mark = res[-1].mem_region.offset + res[-1].mem_region.size
    # if int(mip_high_water_mark) != int(csp_high_water_mark):
    #     print("different", mip_high_water_mark, csp_high_water_mark)
    # print("csp high water mark", res[-1].mem_region.offset + res[-1].mem_region.size)
    # fig = make_memory_map(res, "csp", save=False, rescale_to_mb=False)
    # fig.show()

    # res = gergov(req_mem_allocs)
    # assert verify_allocation(res)
    # test(req_mem_allocs)


if __name__ == "__main__":
    for i in range(1):
        test_lstm()
        # print("\n" + 10 * "*" + "\n")
    # test_resnet18()
