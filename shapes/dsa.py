from pprint import pprint

import sympy as sp
import numpy as np

from symbolics.util import build_tableau_from_eqns


def build_dual_problem(
    live_range_ids_to_tensor_ssa,
    tensor_ssa_to_sympy_expr,
    overlapping_live_ranges_edge_list,
):
    rhss = {}
    symbol_vars = set()
    for id, tensor in live_range_ids_to_tensor_ssa.items():
        size_expr = tensor_ssa_to_sympy_expr[tensor]
        if size_expr.free_symbols:
            rhs = sp.Symbol(name=f"θ_{id}")
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
    opp_Z = {}
    for i, j in overlapping_live_ranges_edge_list:
        if i >= j:
            continue

        Z[i, j] = sp.Symbol(name=f"z_{i}{j}", boolean=True)
        opp_Z[i, j] = sp.Symbol(name=f"oppz_{i}{j}", boolean=True)
        domain_vars.append(Z[i, j])
        domain_vars.append(opp_Z[i, j])

    # Maximize c'x subject to Ax ≤ b, x ≥ 0
    big_M = np.iinfo(np.int32).max
    # big_M = sp.Symbol("M")
    num_constraints = 2 * (len(Z) + len(opp_Z))
    num_vars = len(offsets) + len(Z) + len(opp_Z)
    A = sp.zeros(num_constraints, num_vars)
    b = sp.zeros(1, num_constraints)
    for constr_idx, ((i, j), z) in enumerate(Z.items()):
        mem_i = tensor_ssa_to_sympy_expr[live_range_ids_to_tensor_ssa[i]]

        # offset_i - offset_j - z_ij * M <= -mem_i
        A[constr_idx, i], A[constr_idx, j], A[constr_idx, len(offsets) + constr_idx] = (
            offsets[i],
            -offsets[j],
            -z * big_M,
        )
        b[constr_idx] = -rhss.get(mem_i, mem_i)

    for constr_idx, ((i, j), oppz) in enumerate(opp_Z.items()):
        mem_j = tensor_ssa_to_sympy_expr[live_range_ids_to_tensor_ssa[j]]
        # offset_j - offset_j - (1 - z_ij) * M <= -mem_j
        # <=>
        # offset_j - offset_j - opp_z_ij * M <= -mem_j
        (
            A[len(Z) + constr_idx, j],
            A[len(Z) + constr_idx, i],
            A[len(Z) + constr_idx, len(offsets) + len(Z) + constr_idx],
        ) = (offsets[j], -offsets[i], -oppz * big_M)
        b[len(Z) + constr_idx] = -rhss.get(mem_j, mem_j)

    for constr_idx, ((i, j), z) in enumerate(Z.items()):
        oppz = opp_Z[i, j]
        # opp_z_ij <= 1 - z_ij
        # opp_z_ij + z_ij <= 1
        (
            A[2 * len(Z) + constr_idx, len(offsets) + len(Z) + constr_idx],
            A[2 * len(Z) + constr_idx, len(offsets) + constr_idx],
        ) = (oppz, z)
        b[2 * len(Z) + constr_idx] = 1

    for constr_idx, ((i, j), z) in enumerate(Z.items()):
        oppz = opp_Z[i, j]
        # opp_z_ij >= 1 - z_ij
        # -opp_z_ij - z_ij <= -1
        (
            A[3 * len(Z) + constr_idx, len(offsets) + len(Z) + constr_idx],
            A[3 * len(Z) + constr_idx, len(offsets) + constr_idx],
        ) = (-oppz, -z)
        b[3 * len(Z) + constr_idx] = -1

    # originally we were minimizing (hence just sum), but now we're maximizing (hence negative sum)
    c = sp.zeros(num_vars, 1)
    for i, offset in offsets.items():
        c[i] = -offset

    a = np.array(A)
    # set domain vars to 1
    pprint(domain_vars)
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
        minimize=True,
        use_symbols=False,
    )

    return tableau, tuple(domain_vars), tuple(dual_domain_vars), tuple(symbol_vars)
