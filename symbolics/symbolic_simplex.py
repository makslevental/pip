import itertools as it
import re
import sys
from functools import reduce

import numpy as np
import sympy as sp
from IPython.display import Latex, display
from sympy import init_printing, latex

# init_session(quiet=True)
from sympy.core.relational import Relational, Eq

from symbolics.util import check_constraints_feasible, big_M

init_printing(use_latex=True)


def pprint_latex(expr):
    display(Latex("$$" + latex(expr) + "$$"))


class SymplexError(Exception):
    pass


class UnboundedProblem(SymplexError):
    def __init__(self, msg):
        super(UnboundedProblem, self).__init__(f"unbouded, msg: {msg}")


class NoFeasibleSolution(SymplexError):
    def __init__(self, message):
        super(NoFeasibleSolution, self).__init__(message)


def normalize_tableau(tab, m=None):
    tab = sp.Matrix(tab)
    tab = sp.Matrix([sp.nsimplify(x) for x in tab]).reshape(*tab.shape)
    # m is largest row index
    return tab, (m if m is not None else tab.rows - 1)


def linprog(minimize=None, maximize=None, subject_to=None):
    if minimize is None:
        return linprog(minimize=-sp.Matrix(maximize), subject_to=subject_to)
    tab = sp.Matrix(list(minimize) + [0]).T
    subject_to = list(subject_to)
    for a, ineq, b in subject_to:
        tab = tab.row_insert(
            -1, {"<=": 1, ">=": -1, "==": 1}[ineq] * sp.Matrix(list(a) + [b]).T
        )
    return normalize_tableau(
        tab.col_insert(
            -1,
            sp.eye(tab.rows)[:, [i for i, s in enumerate(subject_to) if s[1] != "=="]],
        )
    )[0]


def find_basis(tab, m=None):
    # find columns that correspond to basic variables (i.e., basis columns)
    tab, m = normalize_tableau(tab, m)
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
    tab = normalize_tableau(tab)[0]
    # divide pivot row by pivot column element (entering variable?)
    tab[pivot[0], :] /= tab[pivot]
    # subtract corresponding columns for all other rows
    for k in set(range(tab.rows)) - {pivot[0]}:
        tab[k, :] -= tab[k, pivot[1]] * tab[pivot[0], :]
    return tab


branches_taken = []
explored = set()
sym_vars = None


def candidate_pivot_cols_avail(tab):
    cand = None
    for j, c in enumerate(tab[-1, :-1]):
        if c.free_symbols and ((c < 0) not in {c for c, _tab in branches_taken}) and check_constraints_feasible(
            [c for c, _tab in branches_taken] + [c < 0], sym_vars
        ):
            branches_taken.append((c < 0, tab))
            return (j, c), tab
        elif not c.free_symbols and c < 0:
            if cand is None:
                cand = (j, c)
            elif c < cand[1]:
                cand = (j, c)

    if cand is None:
        print("branches_taken", end=" ")
        sp.pprint([c for c, _tab in branches_taken])
        sp.pprint(Eq(sp.Symbol("soln"), -tab[-1, -1]))
        print()

        while branches_taken:
            expr, tab = branches_taken.pop()
            assert isinstance(expr, Relational), expr
            if expr in explored:
                continue
            ge = expr.lhs >= 0
            if (
                expr.rel_op == "<"
                and ge not in explored
                and (ge not in {c for c, _tab in branches_taken})
                and check_constraints_feasible(
                    [c for c, _tab in branches_taken] + [ge], sym_vars
                )
            ):
                branches_taken.append((ge, tab))
                return candidate_pivot_cols_avail(tab)
            elif expr.rel_op == ">=":
                explored.add(expr)

    if cand is None:
        return None
    return cand, tab


def find_pivot(j, tab, m=None):
    tab, m = normalize_tableau(tab, m)
    assert all(b >= 0 for b in tab[:m, -1])
    if all(a <= 0 for a in tab[:m, j]):
        raise UnboundedProblem(tab[:m, j])
    sol_col = tab[:m, -1]
    piv_col = tab[:m, j]
    i = np.argmin([b / a if a > 0 else sp.oo for b, a in zip(sol_col, piv_col)])
    return (i, j), tab


def simplex(tab, m=None):
    tab, m = normalize_tableau(tab, m)
    if any(b < 0 for b in tab[:m, -1]):
        raise NoFeasibleSolution("rows with all non-negative elements exist")
    yield tab.copy()
    while j_c_tab := candidate_pivot_cols_avail(tab):
        j_c, tab = j_c_tab
        j, c = j_c
        pivot, tab = find_pivot(j, tab, m=m)
        tab = sweep(tab, pivot)
        # how to identify a leaf?
        yield tab.copy()


def find_dual_pivot(tab, m=None):
    tab, m = normalize_tableau(tab, m)
    assert all(c >= 0 for c in tab[-1, :-1])
    i = np.argmin(tab[:-1, -1])
    if tab[i, -1] >= 0:
        return  # success
    if all(a >= 0 for a in tab[i, :-1]):
        raise NoFeasibleSolution("rows with all non-negative elements exist")
    j = np.argmax(
        [c / a if a < 0 else -sp.oo for c, a in zip(tab[-1, :-1], tab[i, :-1])]
    )
    return i, j


def dual_simplex(tab):
    tab = normalize_tableau(tab)[0]
    if any(c < 0 for c in tab[-1, :-1]):
        raise SymplexError("Feasible solution unknown (c_j < 0 exists)")
    yield tab.copy()
    while any(b < 0 for b in tab[:-1, -1]):
        tab = sweep(tab, find_dual_pivot(tab))
        yield tab.copy()


def two_phase_simplex(tab):
    tab = normalize_tableau(tab)[0]
    tab = tab.col_join(sp.zeros(1, tab.cols))
    for i in (i for i in range(tab.rows - 2) if tab[i, -1] < 0):
        tab[-1, :] += tab[i, :]
        tab[i, :] *= -1
    yield tab
    for tab in it.islice(simplex(tab, m=tab.rows - 2), 1, None):
        yield tab
    if tab[-1, -1] != 0:
        raise NoFeasibleSolution(
            f"optimal value of first phase is non-zero: {tab[-1, -1]}"
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
    tab, m = normalize_tableau(tab)

    tab0 = sp.ImmutableMatrix(tab)
    basis_log = set()

    for cycle, tab in enumerate(
        two_phase_simplex(tab0) if use_two_phase else solve(tab0)
    ):
        disp_tableau(tab, m)
        basis = tuple(find_basis(tab, m))
        if basis in basis_log:
            raise SymplexError("cycle occurs")
        basis_log.add(basis)

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
    title=None, expr=None, indent=4, inline=False, end="\n\n", use_latex=False
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
    tab, m = normalize_tableau(tab, m)
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


if __name__ == "__main__":
    print_solution(
        linprog(
            minimize=[-4, -5],
            subject_to=[
                ([2.5, 5], "<=", 350),
                ([5, 6], "<=", 450),
                ([3, 2], "<=", 240),
            ],
        )
    )
