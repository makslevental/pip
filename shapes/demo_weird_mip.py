import warnings
from dataclasses import dataclass
from pprint import pprint
from typing import List
from IPython.display import display

import numpy as np
import pandas as pd
from ncls import NCLS
from pyscipopt import Model, Eventhdlr, SCIP_PARAMSETTING


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
        # print(events_types[event.getType()])
        # print(self.model.getNLPRows(), self.model.getNLPCols())
        if (
            self.model.getPrimalbound() == self.model.getDualbound()
            and self.model.getNLPRows()
        ):
            print_lp(self.model)


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
    solver.includeEventhdlr(node_eventhdlr, "", "")

    max_mem = sum(r.mem_size for r in required_allocs.itertuples())
    total_mem = solver.addVar(vtype="I", lb=0, ub=max_mem, name="total_mem")

    offsets = []
    for i, row in enumerate(required_allocs.itertuples()):
        offset = solver.addVar(
            vtype="I", lb=0, ub=max_mem, name=f"offset_{row.Index[0]}"
        )
        solver.addCons(offset + row.mem_size <= total_mem)
        offsets.append(offset)

    for i, j in edge_list:
        inters = solver.addVar(vtype="B", lb=0, ub=1, name=f"z_{{{i},{j}}}")
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

    treed = TreeD(scip_model=solver, nodelimit=2000, showcuts=True)
    treed.solve()
    fig = treed.draw()
    fig.show(renderer='notebook')
    fig.show()

    # solver.optimize()

    if solver.getStatus() == "optimal":
        return [
            PlannedAlloc.from_req_row(row, solver.getVal(offsets[row.Index[0]]))
            for row in required_allocs.itertuples()
        ]
    else:
        warnings.warn("mip: The problem does not have an optimal solution.")
        return []


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
    # print("lvrs", lvrs)
    req_mem_allocs = make_df_from_reqs(
        [
            RequiredAlloc(LiveRange(begin, end), size, str(i))
            for i, ((begin, end), size) in enumerate(lvrs.items())
        ]
    )
    res = solve_mip_scip(req_mem_allocs)
    res.sort(key=lambda r: r.mem_region.offset)
    # print("res")
    pprint([(r.lvr.begin, r.lvr.end) for r in res])


if __name__ == "__main__":
    for i in range(10):
        test_lstm()
        # print("\n" + 10 * "*" + "\n")
    # test_resnet18()
