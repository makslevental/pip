import itertools as it
import re
import sys
from collections import defaultdict
from contextlib import suppress
from functools import reduce
from pprint import pprint

import numpy as np
import sympy as sp
from IPython.display import Latex, display
from sympy import init_printing, latex, N
from sympy import pprint as sym_pprint
from sympy import solve_univariate_inequality, Symbol, Interval

# init_session(quiet=True)
init_printing(use_latex=True)


def pprint_latex(expr):
    display(Latex("$$" + latex(expr) + "$$"))


class SymplexError(Exception):
    pass


class UnboundedProblem(SymplexError):
    pass


def _normalize_tableau(tab, m=None):
    tab = sp.Matrix(tab)
    tab = sp.Matrix([sp.nsimplify(x) for x in tab]).reshape(*tab.shape)
    # m is largest row index
    return tab, (m if m is not None else tab.rows - 1)


def linprog(minimize=None, maximize=None, subject_to=None):
    # minimization only - convert max to min negative
    if minimize is None:
        return linprog(minimize=-sp.Matrix(maximize), subject_to=subject_to)
    # add objective row
    tab = sp.Matrix(list(minimize) + [0]).T
    subject_to = list(subject_to)
    for a, ineq, b in subject_to:
        tab = tab.row_insert(
            -1,
            # canonical form is <=
            # multiply by -1 if constraint is >= (in order to reverse)
            # multiply by 1 if equality or already in canonical form
            {"<=": 1, ">=": -1, "==": 1}[ineq] * sp.Matrix(list(a) + [b]).T,
        )
    normed_tableau = _normalize_tableau(
        # insert rows for slack variables corresponding to inequality constraints
        tab.col_insert(
            -1,
            sp.eye(tab.rows)[:, [i for i, s in enumerate(subject_to) if s[1] != "=="]],
        )
    )[0]
    return normed_tableau


def find_basis(tab, m=None):
    # find columns that correspond to basic variables (i.e., basis columns)
    tab, m = _normalize_tableau(tab, m)
    identity = sp.eye(m)
    basis = []
    for i in range(m):
        for j in range(tab.cols - 1):
            if tab[:m, :].col(j) == identity.col(i):
                basis.append(j)
                break
        else:
            basis.append(None)
    return basis


def sweep(tab, pivot):
    # gauss eliminate
    tab = _normalize_tableau(tab)[0]
    # divide pivot row by pivot column element (entering variable?)
    tab[pivot[0], :] /= tab[pivot]
    # subtract corresponding columns for all other rows
    for k in set(range(tab.rows)) - {pivot[0]}:
        tab[k, :] -= tab[k, pivot[1]] * tab[pivot[0], :]
    return tab


def find_pivot(tab, m=None):
    tab, m = _normalize_tableau(tab, m)
    # if any of the right hand sides are negative then
    #
    poses = [a >= 0 for a in tab[:m, -1] if not a.atoms(Symbol)]
    assert not poses or all(b >= 0 for b in tab[:m, -1])
    if tab[-1, :-1].atoms(Symbol):
        idx_sym_exprs = {(j, e) for j, e in enumerate(tab[-1, :-1]) if e.atoms(Symbol)}
        # smarter way to pick entering variable is probably in terms of what's already been explored
        j = np.random.choice([j for j, e in idx_sym_exprs], 1)[0]
    else:
        # -z + (-1)*x0 + 4*x1 = 0
        # -z - 1*x0 + 4*x1 = 0
        # -z = 1*x0 - 4*x1
        # z = -1*x0 + 4*x1
        # the largest negative coefficient pushes z down the most
        j = np.argmin(tab[-1, :-1])
        if tab[-1, j] >= 0:
            # if all coefficients are positive then objective is minimum
            # e.g. cf. above
            # -z + 10*x0 + 4*x1 = 0
            # z = 10*x0 + 4*x1
            # and z is most negative at the mins of the domains of x1, x2
            return  # success
    negs = [a < 0 for a in tab[:m, j] if not a.atoms(Symbol)]
    if negs and all(negs):
        raise UnboundedProblem
    # find in order to not violate other constraints;
    # 1*x1 + 1*x2 <= 12
    # 2*x1 + 1*x2 <= 16
    # if x1 is the pivot column then we cannot choose row 1 as the pivot row
    # if row 1 is the pivot column then we will produce
    # then the implied upper bound for x1 is 12 (1*x1 + 0*x2 <= 12)
    # but then the second constraint would be violated (2*12 + 0*x2 !<= 16)
    idx_sym_exprs = {(j, e) for j, e in enumerate(tab[:m, j]) if e.atoms(Symbol)}
    idx_consts = {(j, e) for j, e in enumerate(tab[:m, j]) if not e.atoms(Symbol)}
    if len(idx_consts):
        i = min([(tab[j, -1] / a, j) if a > 0 else (sp.oo, j) for j, a in idx_consts])
        i = i[1]
    else:
        assert idx_sym_exprs
        i = np.random.choice([j for j, e in idx_sym_exprs], 1)[0]
    return i, j


def find_dual_pivot(tab, m=None):
    tab, m = _normalize_tableau(tab, m)
    assert all(c >= 0 for c in tab[-1, :-1])
    i = np.argmin(tab[:-1, -1])
    if tab[i, -1] >= 0:
        return  # success
    if all(a >= 0 for a in tab[i, :-1]):
        raise SymplexError(
            "No feasible solution (rows with all non-negative elements exist)"
        )
    j = np.argmax(
        [c / a if a < 0 else -sp.oo for c, a in zip(tab[-1, :-1], tab[i, :-1])]
    )
    return i, j


def simplex(tab, m=None):
    tab, m = _normalize_tableau(tab, m)
    if any(b < 0 for b in tab[:m, -1]):
        raise SymplexError(
            "No feasible solution (rows with all non-negative elements exist)"
        )
    yield tab.copy()
    while any(c < 0 for c in tab[-1, :-1]):
        pivot = find_pivot(tab, m=m)
        tab = sweep(tab, pivot)
        yield tab.copy()


def dual_simplex(tab):
    tab = _normalize_tableau(tab)[0]
    if any(c < 0 for c in tab[-1, :-1]):
        raise SymplexError("Feasible solution unknown (c_j < 0 exists)")
    yield tab.copy()
    while any(b < 0 for b in tab[:-1, -1]):
        tab = sweep(tab, find_dual_pivot(tab))
        yield tab.copy()


def two_phase_simplex(tab):
    tab = _normalize_tableau(tab)[0]
    tab = tab.col_join(sp.zeros(1, tab.cols))
    for i in (i for i in range(tab.rows - 2) if tab[i, -1] < 0):
        tab[-1, :] += tab[i, :]
        tab[i, :] *= -1
    yield tab
    for tab in it.islice(simplex(tab, m=tab.rows - 2), 1, None):
        yield tab
    if tab[-1, -1] != 0:
        raise SymplexError(
            "No feasible solution (optimal value of first stage is non-zero)"
        )
    for tab in it.islice(simplex(tab[:-1, :]), 1, None):
        yield tab


def solve(tab):
    seq = [i for i, j in enumerate(find_basis(tab)) if j is None]
    tab = reduce(
        lambda tab, i: sweep(
            tab,
            (
                i,
                np.argmax(
                    [
                        c / a if a < 0 else -sp.oo
                        for c, a in zip(tab[-1, :-1], tab[i, :-1])
                    ]
                ),
            ),
        ),
        seq,
        tab,
    )
    assert all(j is not None for j in find_basis(tab))
    if all(b >= 0 for b in tab[:-1, -1]):
        return simplex(tab)
    if all(c >= 0 for c in tab[-1, :-1]):
        return dual_simplex(tab)
    return two_phase_simplex(tab)


def print_solution(tab, use_two_phase=None):
    tab, m = _normalize_tableau(tab)

    tab0 = sp.ImmutableMatrix(tab)
    basis_log = set()

    try:
        for cycle, tab in enumerate(
            two_phase_simplex(tab0) if use_two_phase else solve(tab0)
        ):
            disp_tableau(tab, m)
            basis = tuple(find_basis(tab, m))
            if basis in basis_log:
                raise SymplexError("patrol occurs")
            basis_log.add(basis)
    except SymplexError as e:
        disp_math(str(e))
        return

    tab = tab[: m + 1, :]
    basis = find_basis(tab)
    B = tab0[:-1, basis]
    cB = tab0[-1, basis].T

    x = sp.Matrix(
        tab0.cols - 1, 1, lambda i, j: tab[basis.index(i), -1] if i in basis else 0
    )
    disp_math("(B, B^{-1}, c_B) =", (B, B.inv(), cB))
    disp_math("z =", -tab[-1, -1], inline=True)
    disp_math("x^T =", x.T[:-m], inline=True)
    disp_math("π^T = ", cB.T * B.inv(), inline=True)


def disp_math(
    title=None, expr=None, indent=4, inline=False, end="\n\n", use_latex=True
):
    if expr is None:
        expr = ""
    if use_latex:
        from IPython.display import Latex, display

        display(Latex("$$" + (title or "") + sp.latex(expr) + "$$"))
        return
    if inline:
        if title:
            title += " "
        sys.stdout.write(title + sp.pretty(expr) + end)
        return
    if title:
        sys.stdout.write(title + "\n\n")
    sys.stdout.write(re.sub(r"(^|\n)", r"\1" + " " * indent, sp.pretty(expr)) + end)


def disp_tableau(tab, m=None):
    tab, m = _normalize_tableau(tab, m)
    basis = [
        sp.var("x_{}".format(i + 1)) if i is not None else sp.var(r"μ_{}".format(k + 1))
        for k, i in enumerate(find_basis(tab, m))
    ]
    basis += [-sp.var("z")]
    if tab.rows == m + 2:
        basis += [-sp.var("w")]
    vsep = sp.Matrix([[sp.var("|")] * tab.rows]).T
    hsep = sp.Matrix([[sp.var("--")] * (tab.cols + 3)])
    tab = (
        sp.Matrix([basis])
        .T.row_join(tab)
        .col_insert(1, vsep)
        .col_insert(-1, vsep)
        .row_insert(m, hsep)
    )
    disp_math("", tab)


def handle_parameterized_objective(tab):
    tab, m = _normalize_tableau(tab)
    objective_row = tab[-1, :-1]
    objective_val = tab[-1, -1]
    syms = tab[-1, :-1].atoms(Symbol)
    if not syms:
        return
    assert len(syms) == 1, "only handle one sym for right now"
    sym = syms.pop()
    row_set = {(j, e) for j, e in enumerate(objective_row)}
    idx_sym_exprs = {(j, e) for j, e in row_set if e.atoms(Symbol)}
    # all non-basic variables >= 0 means tableau is optimal
    if all(e >= 0 for _, e in row_set - idx_sym_exprs):
        sol = None
        for j, expr in idx_sym_exprs:
            _sol = solve_univariate_inequality(expr >= 0, sym, relational=False)
            col = tab[:m, j]
            negs = [c < 0 for c in col if not c.atoms(Symbol)]
            if negs and all(negs):
                PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP[
                    Interval(-sp.oo, sp.oo) - _sol
                ] = -sp.oo
            if sol is None:
                sol = _sol
            else:
                sol &= _sol
        sol = sol.simplify()
        if sol != sp.EmptySet:
            PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP[sol] = objective_val


def print_param_sol_dict():
    collected_intervals = []
    for inter, fn in sorted(
        PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP.items(),
        key=lambda inter_fn: inter_fn[0].right,
    ):
        if collected_intervals:
            prev_inter, prev_fn = collected_intervals[-1]
            if (
                inter.intersection(prev_inter) == inter
                or prev_inter.intersection(inter) == prev_inter
            ):
                assert fn == prev_fn
                collected_intervals.pop()
                inter = prev_inter.union(inter)

        collected_intervals.append((inter, fn))

    for inter, fn in collected_intervals:
        sym_pprint({"interval": inter, "soln": fn})
    print()


def check_param_coverage():
    intervals = list(PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP.keys())

    return intervals and reduce(
        lambda acc, val: acc.union(val), intervals[1:], intervals[0]
    ) == Interval(-sp.oo, sp.oo)


PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP = {}


def test_single_symbol_objective():
    global PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP
    PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP = {}

    tableau = linprog(
        minimize=[-3 + 2 * sp.var("θ"), 3 - sp.var("θ"), 1],
        subject_to=[
            ([1, 2, -3], "<=", 5),
            ([2, 1, -4], "<=", 7),
            # ([1, 0, 0], ">=", 0),
            # ([0, 1, 0], ">=", 0),
            # ([0, 0, 1], ">=", 0),
        ],
    )

    tableau_stack = [tableau.copy()]

    while not check_param_coverage():
        tableau = tableau_stack[-1]
        disp_tableau(tableau)
        # sym_pprint(tableau)
        handle_parameterized_objective(tableau)
        try:
            pivot = find_pivot(tableau)
            tableau = sweep(tableau, pivot)
            tableau_stack.append(tableau.copy())
        except UnboundedProblem:
            continue

    assert check_param_coverage(), print_param_sol_dict()
    print_param_sol_dict()


def test_single_symbol_constraints():
    global PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP
    PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP = {}

    tableau = linprog(
        minimize=[-3, 3, 1],
        subject_to=[
            ([1, 2 * sp.var("θ"), -3], "<=", 5),
            ([2, 1, -4 + sp.var("θ")], "<=", 7),
            # ([1, 0, 0], ">=", 0),
            # ([0, 1, 0], ">=", 0),
            # ([0, 0, 1], ">=", 0),
        ],
    )

    tableau_stack = [tableau.copy()]

    while not check_param_coverage():
        tableau = tableau_stack[-1]
        disp_tableau(tableau)
        sym_pprint(tableau)
        handle_parameterized_objective(tableau)
        try:
            pivot = find_pivot(tableau)
            tableau = sweep(tableau, pivot)
            tableau_stack.append(tableau.copy())
        except UnboundedProblem:
            continue

    assert check_param_coverage(), print_param_sol_dict()
    print_param_sol_dict()


if __name__ == "__main__":
    test_single_symbol_objective()
    # test_single_symbol_constraints()

# b = {"c": "d", "d": 3}
# with suppress(Exception, SyntaxError):
#     (
#         c := b["c"],
#         d := b[c],
#     )
# print(c)
# print(d)
