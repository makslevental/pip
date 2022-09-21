import time
from pprint import pformat, pprint

import numpy as np
import sympy as sp
import cvxpy as cp
import matplotlib.pyplot as plt
from ncls import NCLS

import torch
from torch import nn

from shapes.shape_infer import (
    make_shape_stuff,
    simplify_with_mathematica,
    get_tensor_live_ranges,
)


def test_conv2d():
    torch._C._jit_set_symbolic_shapes_test_mode(True)
    model = nn.Conv2d(in_channels=16, out_channels=33, kernel_size=3, stride=2).eval()
    ss_map = {
        "input_1[0]": "a",
        "input_1[1]": "b",
        "input_1[2]": "c",
        "input_1[3]": "d",
    }
    inp_shapes = [None, [None, None, None, None]]
    (
        partial_shape_eval_graph,
        sym_to_shape_inputs,
        convs_to_greeks,
        tensor_to_sizes,
        frozen_graph,
    ) = make_shape_stuff(model, ss_map, inp_shapes)
    print(f"{partial_shape_eval_graph=}")

    print("orig:", sym_to_shape_inputs)
    simplified = simplify_with_mathematica(sym_to_shape_inputs, ss_map, convs_to_greeks)

    print("simplified:")
    for shape_sym, shape_formula in simplified.items():
        print(shape_sym, "&=", shape_formula, r"\\")


def test_resnet():
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18")
    # sym1 = torch._C._new_symbolic_shape_symbol()
    # sym2 = torch._C._new_symbolic_shape_symbol()
    sym3 = torch._C._new_symbolic_shape_symbol()
    sym4 = torch._C._new_symbolic_shape_symbol()
    ssa_to_param_map = {
        "input_1[0]": "a",
        "input_1[1]": "b",
        "input_1[2]": "c",
        "input_1[3]": "d",
    }
    inp_shapes = [None, [1, 3, sym3, sym4]]
    (
        partial_shape_eval_graph,
        sym_ssa_to_shape_inputs,
        shape_sym_to_val_debug_name,
        shape_ssa_to_greeks,
        tensor_ssa_to_sizes,
        frozen_graph,
    ) = make_shape_stuff(model, ssa_to_param_map, inp_shapes)
    sym_ssa_to_sym = {v: k for k, v in shape_sym_to_val_debug_name.items()}
    print(frozen_graph.str(False))
    print(partial_shape_eval_graph.str(False))

    print("orig:", pformat(sym_ssa_to_shape_inputs))
    simplified = simplify_with_mathematica(sym_ssa_to_shape_inputs, ssa_to_param_map)

    shape_sym_to_formula = {}
    for shape_ssa, shape_formula in simplified.items():
        shape_sym = sym_ssa_to_sym[shape_ssa]
        shape_sym_to_formula[shape_sym] = shape_formula
        # print(shape_sym, "&=", shape_formula, r"\\")

    live_ranges = get_tensor_live_ranges(frozen_graph)
    assert set(live_ranges.keys()) == set(tensor_ssa_to_sizes.keys())
    return tensor_ssa_to_sizes, shape_sym_to_formula, live_ranges


def test_resnet_symbolic_planning():
    tensor_ssa_to_sizes, shape_sym_to_formula, live_ranges = test_resnet()
    tensor_ssa_to_sympy_expr = {}
    for tensor, sizes in tensor_ssa_to_sizes.items():
        sympy_expr = np.prod(
            [sp.sympify(shape_sym_to_formula.get(s, s), rational=True) for s in sizes]
        )
        tensor_ssa_to_sympy_expr[tensor] = sympy_expr

    ids = {
        id: tensor
        for id, tensor in zip(np.arange(len(live_ranges)), live_ranges.keys())
    }
    starts = np.array([live_ranges[tensor][0] for tensor in ids.values()])
    ends = np.array([live_ranges[tensor][1] for tensor in ids.values()])
    live_range_ncls = NCLS(
        starts=starts,
        ends=ends + 1,
        ids=np.array(list(ids.keys())),
    )
    edge_list = np.array(
        live_range_ncls.all_overlaps_both(
            starts=starts,
            ends=ends + 1,
            indexes=np.array(list(ids.keys())),
        )
    )
    edge_list = edge_list.T.astype(order="C", dtype=edge_list.dtype)

    params = {}
    constraints = []
    for id, tensor in ids.items():
        size_expr = tensor_ssa_to_sympy_expr[tensor]
        for sym in size_expr.free_symbols:
            params[sym] = cp.Parameter(
                name=str(sym), nonneg=True, value=np.random.randint(1, 10)
            )
            # constraints.append(params[sym] >= 0)

    Z = {}
    # offset_i - offset_j - z_ij * M <= -mem_i
    # offset_j - offset_j - (1 - z_ij) * M <= -mem_j
    offsets = {}
    for id in ids.keys():
        offsets[id] = cp.Variable(integer=True, name=f"offset_{id}")
        constraints.append(offsets[id] >= 0)

    M = 1e6
    for i, j in edge_list:
        if i >= j:
            continue

        Z[i, j] = cp.Variable(name=f"z_({i},{j})", boolean=True)
        mem_i = tensor_ssa_to_sympy_expr[ids[i]]
        mem_j = tensor_ssa_to_sympy_expr[ids[j]]
        constraints.append(
            offsets[i] - offsets[j] - Z[i, j] * M
            <= -np.prod([params.get(m, m) for m in mem_i.args])
        )
        constraints.append(
            offsets[j] - offsets[i] - (1 - Z[i, j]) * M
            <= -np.prod([params.get(m, m) for m in mem_j.args])
        )

    obj = cp.Minimize(cp.sum(list(offsets.values())))
    assert obj.is_dpp()

    prob = cp.Problem(obj, constraints)
    assert prob.is_dcp(dpp=True)
    print(cp.installed_solvers())
    times = []
    new_problem_times = []

    xs = range(20)
    for _ in xs:
        for p in params.values():
            p.value = np.random.randint(1, 10)

        start = time.time()
        prob.solve(solver=cp.GUROBI, verbose=True, warm_start=True)
        end = time.time()
        times.append(end - start)
        new_problem = cp.Problem(obj, constraints)
        start = time.time()
        new_problem.solve(solver=cp.GUROBI, verbose=True, warm_start=True)
        end = time.time()
        new_problem_times.append(end - start)

    plt.rc("font", family="serif")
    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(xs, times, label="Re-solving a DPP problem")
    plt.plot(xs, new_problem_times, label="Solving a new problem")
    plt.xlabel(r"iter", fontsize=16)
    plt.ylabel(r"time (s)", fontsize=16)
    plt.legend()
    plt.yscale("log")
    plt.savefig("../docs/dpp.pdf")


if __name__ == "__main__":
    test_resnet_symbolic_planning()
