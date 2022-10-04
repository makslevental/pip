import itertools
from pprint import pprint

import sympy as sp
import numpy as np

from symbolics.util import build_tableau_from_eqns, big_M


def collect_problem_data(
    live_range_ids_to_tensor_ssa,
    tensor_ssa_to_sympy_expr,
    overlapping_live_ranges_edge_list,
):
    rhss = {}
    symbol_vars = set()
    for id, tensor in live_range_ids_to_tensor_ssa.items():
        size_expr = tensor_ssa_to_sympy_expr[tensor]
        if size_expr.free_symbols:
            rhs = sp.Symbol(name=f"φ_{id}")
            rhss[size_expr] = rhs
            symbol_vars.add(rhs)

    for size_expr, rhs in rhss.items():
        print(f"{size_expr=} {rhs=}")
    domain_vars = []
    offsets = {}
    for id in live_range_ids_to_tensor_ssa.keys():
        offsets[id] = sp.Symbol(integer=True, name=f"offset_{id}")
        domain_vars.append(offsets[id])

    Z = {}
    for i, j in overlapping_live_ranges_edge_list:
        if i >= j:
            continue

        # Z.append((i, j))
        Z[i, j] = sp.Symbol(name=f"z_{i}{j}", boolean=True)
        symbol_vars.add(Z[i, j])
    symbol_vars.add(big_M)
    return offsets, Z, rhss, tuple(domain_vars), tuple(symbol_vars)


def build_dual_problem(offsets, Z, mems, rhss, domain_vars, symbol_vars):
    # Maximize c'x subject to Ax ≤ b, x ≥ 0
    num_constraints = 2 * len(Z)
    num_vars = len(offsets)  # + len(Z)
    A = sp.zeros(num_constraints, num_vars)
    b = sp.zeros(1, num_constraints)
    for constr_idx, ((i, j), z) in enumerate(Z.items()):

        mem_i, mem_j = mems[i], mems[j]
        # offset_i - offset_j <= -mem_i + z_ij * M
        (
            A[2 * constr_idx, i],
            A[2 * constr_idx, j],
        ) = (offsets[i], -offsets[j])
        b[2 * constr_idx] = -rhss.get(mem_i, mem_i) + z * big_M
        # offset_j - offset_j <= -mem_j + (1 - z_ij) * M
        (
            A[2 * constr_idx + 1, j],
            A[2 * constr_idx + 1, i],
        ) = (offsets[j], -offsets[i])
        b[2 * constr_idx + 1] = -rhss.get(mem_j, mem_j) + (1 - z) * big_M

    # originally we were minimizing (hence just sum), but now we're maximizing (hence negative sum)
    c = sp.zeros(num_vars, 1)
    for i, offset in offsets.items():
        c[i] = -offset

    # set domain vars to 1
    A = A.subs([(d, 1) for d in domain_vars])
    b = b.subs([(d, 1) for d in domain_vars])
    c = c.subs([(d, 1) for d in domain_vars])

    # Minimize b'y subject to A'y ≥ c, y ≥ 0
    y = sp.Matrix(num_constraints, 1, lambda i, j: sp.Symbol(f"y_{i}"))
    dual_domain_vars = y.values()
    result_var = sp.Symbol(name="result")
    objective = sp.Eq(result_var, (b @ y)[0])
    dual_constraints = [objective]
    for dual_row_idx in range(A.T.rows):
        dual_row = A.T.row(dual_row_idx)
        dual_constraints.append((dual_row @ y)[0] >= c[dual_row_idx])

    tableau, *_ = build_tableau_from_eqns(
        dual_constraints,
        domain_vars=tuple(dual_domain_vars),
        range_var=result_var,
        symbol_vars=tuple(symbol_vars),
        minimize=False,
        use_symbols=False,
    )

    return tableau


def build_dual_problems(
    live_range_ids_to_tensor_ssa,
    tensor_ssa_to_sympy_expr,
    overlapping_live_ranges_edge_list,
):
    offsets, Z, rhss, domain_vars, symbol_vars = collect_problem_data(
        live_range_ids_to_tensor_ssa,
        tensor_ssa_to_sympy_expr,
        overlapping_live_ranges_edge_list,
    )

    mems = {}
    for i, j in Z:
        mems[i] = tensor_ssa_to_sympy_expr[live_range_ids_to_tensor_ssa[i]]
        if j not in mems:
            mems[j] = tensor_ssa_to_sympy_expr[live_range_ids_to_tensor_ssa[j]]

    tableau = build_dual_problem(offsets, Z, mems, rhss, domain_vars, symbol_vars)

    return tableau, symbol_vars
