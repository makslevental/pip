import time
from pprint import pformat, pprint

import numpy as np
import sympy as sp
import cvxpy as cp
import matplotlib.pyplot as plt
from ncls import NCLS

import torch
from torch import nn
import torch.nn.functional as F

from shapes.shape_infer import (
    make_shape_stuff,
    simplify_with_mathematica,
    get_tensor_live_ranges,
    get_constraints,
)
from symbolics.symbol import SymbolicTableau, solve
from symbolics.util import build_tableau_from_eqns


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


def test_vision_model(model):
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


def test_resnet_ddp():
    model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18")
    tensor_ssa_to_sizes, shape_sym_to_formula, live_ranges = test_vision_model(model)
    tensor_ssa_to_sympy_expr, ids, edge_list = get_constraints(
        tensor_ssa_to_sizes, shape_sym_to_formula, live_ranges
    )

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
        # prob.solve(solver=cp.GLPK_MI, verbose=True, warm_start=True)
        end = time.time()
        times.append(end - start)
        new_problem = cp.Problem(obj, constraints)
        start = time.time()
        new_problem.solve(solver=cp.GUROBI, verbose=True, warm_start=True)
        # new_problem.solve(solver=cp.GLPK_MI, verbose=True, warm_start=True)
        end = time.time()
        new_problem_times.append(end - start)

    plt.rc("font", family="serif")
    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot(xs, times, label="Re-solving a DPP problem")
    plt.plot(xs, new_problem_times, label="Solving a new problem")
    plt.xticks(xs, list(map(int, xs)))
    plt.xlabel(r"trials", fontsize=16)
    plt.ylabel(r"time (s)", fontsize=16)
    plt.legend()
    plt.yscale("log")
    plt.savefig("../docs/dpp1.png")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_net_pip():
    model = Net()
    tensor_ssa_to_sizes, shape_sym_to_formula, live_ranges = test_vision_model(model)
    tensor_ssa_to_sympy_expr, ids, edge_list = get_constraints(
        tensor_ssa_to_sizes, shape_sym_to_formula, live_ranges
    )

    params = {}
    constraints = []
    domain_vars = set()
    symbol_vars = set()
    for id, tensor in ids.items():
        size_expr = tensor_ssa_to_sympy_expr[tensor]
        for sym in size_expr.free_symbols:
            params[sym] = sym
            symbol_vars.add(sym)

    Z = {}
    # offset_i - offset_j - z_ij * M <= -mem_i
    # offset_j - offset_j - (1 - z_ij) * M <= -mem_j
    offsets = {}
    for id in ids.keys():
        offsets[id] = sp.Symbol(integer=True, name=f"offset_{id}")
        domain_vars.add(offsets[id])
        constraints.append(offsets[id] >= 0)

    M = 1e6
    for i, j in edge_list:
        if i >= j:
            continue

        Z[i, j] = sp.Symbol(name=f"z_{i}{j}", boolean=True)
        domain_vars.add(Z[i, j])
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

    result = sp.Symbol(name="result")
    constraints.append(sp.Eq(result, sum(list(offsets.values()))))
    tableau, domain_vars, symbol_vars = build_tableau_from_eqns(
        constraints,
        tuple(domain_vars),
        range_var=result,
        symbol_vars=tuple(symbol_vars),
        minimize=True,
        use_symbols=True,
    )

    tab = SymbolicTableau(tableau, set(domain_vars), set(symbol_vars))
    # TODO: need to build the dual here since solve is in terms of symbolic objective
    solve(tab)

    # obj = cp.Minimize(cp.sum(list(offsets.values())))
    # assert obj.is_dpp()
    #
    # prob = cp.Problem(obj, constraints)
    # assert prob.is_dcp(dpp=True)
    # print(cp.installed_solvers())
    # times = []
    # new_problem_times = []
    #
    # xs = range(20)
    # for _ in xs:
    #     for p in params.values():
    #         p.value = np.random.randint(1, 10)
    #
    #     start = time.time()
    #     prob.solve(solver=cp.GUROBI, verbose=True, warm_start=True)
    #     # prob.solve(solver=cp.GLPK_MI, verbose=True, warm_start=True)
    #     end = time.time()
    #     times.append(end - start)
    #     new_problem = cp.Problem(obj, constraints)
    #     start = time.time()
    #     new_problem.solve(solver=cp.GUROBI, verbose=True, warm_start=True)
    #     # new_problem.solve(solver=cp.GLPK_MI, verbose=True, warm_start=True)
    #     end = time.time()
    #     new_problem_times.append(end - start)
    #
    # plt.rc("font", family="serif")
    # plt.figure(figsize=(6, 6), dpi=300)
    # plt.plot(xs, times, label="Re-solving a DPP problem")
    # plt.plot(xs, new_problem_times, label="Solving a new problem")
    # plt.xticks(xs, list(map(int, xs)))
    # plt.xlabel(r"trials", fontsize=16)
    # plt.ylabel(r"time (s)", fontsize=16)
    # plt.legend()
    # plt.yscale("log")
    # plt.savefig("../docs/dpp1.png")


if __name__ == "__main__":
    test_net_pip()
