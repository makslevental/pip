import warnings
from dataclasses import dataclass
from itertools import combinations
from pprint import pprint
from typing import List
from IPython.display import display

import numpy as np
import pandas as pd
from ncls import NCLS
from networkx import find_cliques, node_clique_number
from networkx.algorithms.approximation import max_clique
from pyscipopt import Model, Eventhdlr, SCIP_PARAMSETTING
from pyscipopt.scip import PY_SCIP_EVENTTYPE
import inspect
import hashlib
import networkx as nx

from shapes.plotting import make_memory_map
from shapes.strategies import solve_csp
from treed import TreeD


def make_max_clique_graph(G):
    B = G.__class__()
    cliques = list(enumerate(set(c) for c in find_cliques(G)))
    # Add a numbered node for each clique.
    B.add_nodes_from((i, {"nodes": c}) for i, c in cliques)
    # Join cliques by an edge if they share a node.
    clique_pairs = combinations(cliques, 2)
    B.add_edges_from(
        (i, j, {"nodes": c1 & c2}) for (i, c1), (j, c2) in clique_pairs if c1 & c2
    )
    return B


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
            % 10**8
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
        # print(df)
        display(df)


events_types_ = dict(
    inspect.getmembers(PY_SCIP_EVENTTYPE, predicate=lambda x: isinstance(x, int))
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
        # if (
        #     self.model.getPrimalbound() == self.model.getDualbound()
        #     and self.model.getNLPRows()
        # ):
        #     print_lp(self.model)

        evt_type = events_types[event.getType()]
        print(evt_type)
        if evt_type.startswith("ROW"):
            return
        try:
            print(self.model.getPrimalbound(), self.model.getDualbound())
            if event.getNode() and event.getNode().getParentBranchings():
                print(event.getNode().getParentBranchings())
                # variables, branchbounds, boundtypes = event.getNode().getParentBranchings()
        except Exception as e:
            print(e)

        # if event.getType() in {
        #     SCIP_EVENTTYPE.LPSOLVED,
        #     SCIP_EVENTTYPE.FIRSTLPSOLVED,
        #     SCIP_EVENTTYPE.NODEBRANCHED,
        #     SCIP_EVENTTYPE.NODESOLVED,
        #     SCIP_EVENTTYPE.NODEFOCUSED,
        #     SCIP_EVENTTYPE.NODEINFEASIBLE,
        #     SCIP_EVENTTYPE.ROWDELETEDLP
        #     SCIP_EVENTTYPE.OBJCHANGED
        #     SCIP_EVENTTYPE.BESTSOLFOUND,
        # }:
        #     # print(events_types[event.getType()])
        #     node = event.getNode()
        #
        #     if node.getDepth() == 0:
        #         if node.getParent() is not None:
        #             pass
        #         assert node.getParentBranchings() is None
        #         return
        #     variables, branchbounds, boundtypes = node.getParentBranchings()
        #     print(variables)
        #     # print([f"{len([v for v in variables if '_z' in v.name])}z"] + [v for v in variables if "_z" not in v.name])
        #     # print(variables, end=" ")
        #     # print()
        #     # if rows := self.model.getLPRowsData():
        #     #     print(", ".join([r.name for r in rows]))
        #     #     # print(", ".join(map(str, [r.isRemovable() for r in rows])))
        #     #     # print(", ".join(map(str, [r.isIntegral() for r in rows])))
        #     #     # print(", ".join([r.getBasisStatus() for r in rows]))
        #     #     print([(r.getLhs() == r.getRhs()) for r in rows])
        #     #
        #     # # if cols := self.model.getLPColsData():
        #     # #     print(", ".join([r.getVar().name for r in cols]))
        #     # #     print(", ".join([r.getBasisStatus() for r in cols]))
        #     #
        #     # # print(self.model.getPrimalbound(), self.model.getDualbound())
        #


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
    solver.hideOutput()

    solver.setPresolve(SCIP_PARAMSETTING.OFF)
    node_eventhdlr = NodeEventHandler()
    # solver.includeEventhdlr(node_eventhdlr, "", "")

    max_mem = sum(r.mem_size for r in required_allocs.itertuples())
    total_mem = solver.addVar(vtype="I", lb=0, ub=max_mem, name="total_mem")

    offsets = []
    for i, row in enumerate(required_allocs.itertuples()):
        offset = solver.addVar(
            vtype="I", lb=0, ub=max_mem, name=f"offset_{row.Index[0]}"
        )
        solver.chgVarBranchPriority(offset, -1e6)
        solver.addCons(offset + row.mem_size <= total_mem, name=f"o_{i} + m_{i} <= t")
        offsets.append(offset)

    zs = {}
    for i, j in edge_list:
        zs[i, j] = z = solver.addVar(vtype="B", lb=0, ub=1, name=f"z_{{{i},{j}}}")
        solver.chgVarBranchPriority(z, 1e6)
        solver.addCons(
            offsets[i] + required_allocs.iloc[i].mem_size <= offsets[j] + z * max_mem,
            name=f"o_{i} +m_{i} <= o_{j} + z*M",
        )
        # conversely here
        solver.addCons(
            offsets[j] + required_allocs.iloc[j].mem_size
            <= offsets[i] + (1 - z) * max_mem,
            name=f"o_{j} +m_{j} <= o_{i} + (1-z)*M",
        )

    # Minimize u
    solver.setObjective(total_mem, "minimize")

    # treed = TreeD(scip_model=solver, nodelimit=2000, showcuts=True)
    # treed.solve()
    # fig = treed.draw2d()
    # fig.write_html(file="mip.html", include_plotlyjs=True)
    # fig.show()

    solver.optimize()

    if solver.getStatus() == "optimal":
        return [
            PlannedAlloc.from_req_row(row, solver.getVal(offsets[row.Index[0]]))
            for row in required_allocs.itertuples()
        ], {(i, j): solver.getVal(z) for (i, j), z in zs.items()}
    else:
        warnings.warn("mip: The problem does not have an optimal solution.")
        return []


def schedule_cliques(lvrs):
    G = nx.interval_graph(lvrs.keys())
    max_cliques = make_max_clique_graph(G)
    for clique, nodes in max_cliques.nodes(data=True):
        nodes = nodes["nodes"]
        req_mem_allocs = make_df_from_reqs(
            [
                RequiredAlloc(LiveRange(begin, end), lvrs[begin, end], str(i))
                for i, (begin, end) in enumerate(nodes)
            ]
        )
        planned_allocs, zs = solve_mip_scip(req_mem_allocs)
        planned_allocs.sort(key=lambda r: r.mem_region.offset)
        # print([v for v in sorted(zs.values())])
        # print([(r.lvr.begin, r.lvr.end) for r in planned_allocs])
        make_memory_map(planned_allocs, "mip", save=False).show()

        mip_high_water_mark = (
            planned_allocs[-1].mem_region.offset + planned_allocs[-1].mem_region.size
        )
        print(f"clique {clique}", mip_high_water_mark)


def schedule_whole(lvrs):
    req_mem_allocs = make_df_from_reqs(
        [
            RequiredAlloc(LiveRange(begin, end), size, str(i))
            for i, ((begin, end), size) in enumerate(lvrs.items())
        ]
    )
    planned_allocs, zs = solve_mip_scip(req_mem_allocs)
    planned_allocs.sort(key=lambda r: r.mem_region.offset)
    # print([v for v in sorted(zs.values())])
    # print([(r.lvr.begin, r.lvr.end) for r in planned_allocs])
    make_memory_map(planned_allocs, "mip", save=False).show()

    mip_high_water_mark = (
        planned_allocs[-1].mem_region.offset + planned_allocs[-1].mem_region.size
    )
    print("whole", mip_high_water_mark)


def test_lstm():
    k = 2
    lvrs = {
        # (0, 3): 1024,
        # (1, 3): 1024,
        # (2, 4): 1024,
        # (5, 10): 2 ** np.random.randint(1, 5),
        # (6, 14): 2 ** np.random.randint(1, 5),
        # (7, 10): 2 ** np.random.randint(1, 5),
        # (8, 10): 2 ** np.random.randint(1, 5),
        # (8, 12): 2 ** np.random.randint(1, 5),
        # (9, 12): 2 ** np.random.randint(1, 5),
        # (13, 14): 2 ** np.random.randint(1, 5),
        (1, 7): 2 ** np.random.randint(1, 5),
        (2, 7): 2 ** np.random.randint(1, 5),
        (3, 7): 2 ** np.random.randint(1, 5),
        (4, 7): 2 ** np.random.randint(1, 5),
        (5, 7): 2 ** np.random.randint(1, 5),
        (6, 7): 2 ** np.random.randint(1, 5),
        (6, 15): 2 ** np.random.randint(1, 5),
        (10, 15): 2 ** np.random.randint(1, 5),
        (11, 15): 2 ** np.random.randint(1, 5),
        (12, 15): 2 ** np.random.randint(1, 5),
        (13, 15): 2 ** np.random.randint(1, 5),
        # (1, 3): 2 ** np.random.randint(1, 5),
        # (2, 4): 2 ** np.random.randint(1, 5),
        # (3, 5): 2 ** np.random.randint(1, 5),
        # (4, 6): 2 ** np.random.randint(1, 5),
        # (5, 7): 2 ** np.random.randint(1, 5),
        # (6, 8): 2 ** np.random.randint(1, 5),
    }
    # print("lvrs", lvrs)

    schedule_whole(lvrs)
    schedule_cliques(lvrs)

    # print(mip_high_water_mark)
    #
    # planned_allocs = solve_csp(req_mem_allocs)
    # planned_allocs.sort(key=lambda r: r.mem_region.offset)
    # pprint([(r.lvr.begin, r.lvr.end) for r in planned_allocs])
    # csp_high_water_mark = planned_allocs[-1].mem_region.offset + planned_allocs[-1].mem_region.size
    # print(csp_high_water_mark)


if __name__ == "__main__":
    for i in range(10):
        test_lstm()
        print("\n" + 10 * "*" + "\n")
    # test_resnet18()
