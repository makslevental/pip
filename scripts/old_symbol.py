import dataclasses
from dataclasses import dataclass, field
from functools import reduce
from pprint import pformat

import numpy as np
import sympy as sp
from sympy import (
    Interval,
    pprint as sym_pprint,
    Symbol,
    solve_univariate_inequality,
)

from simplex import (
    UnboundedProblem,
    normalize_tableau,
)
from util import check_constraints_feasible, unitize_syms

EPS = 1


def check_param_coverage():
    intervals = list(PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP.keys())

    return intervals and reduce(
        lambda acc, val: acc.union(val), intervals[1:], intervals[0]
    ) == Interval(-sp.oo, sp.oo)


PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP = {}


def handle_parameterized_objective(tab):
    tab, m = normalize_tableau(tab)
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


def find_symbolic_pivot(tab, build_param_tableau):
    tab, m = normalize_tableau(tab)
    # if any of the right hand sides are negative then
    #
    param_tableau = build_param_tableau(0)
    evaled_context_tableau = param_tableau.subs(
        {s: 1 for s in param_tableau.free_symbols}
    )
    # _z, x = get_solution(evaled_context_tableau)
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


def get_symbol_exprs_in_objective(tab):
    return tuple(c for c in tab[-1, :] if c.free_symbols)


EXPLORED: set[sp.Expr] = set()
BRANCHES: list[sp.Expr] = []


@dataclass
class SymbolicTableau:
    tableau: sp.Matrix
    domain_vars: set[sp.Symbol]
    sym_vars: set[sp.Symbol]
    neg_constraints: set[sp.Expr] = field(default_factory=lambda: set())
    pos_constraints: set[sp.Expr] = field(default_factory=lambda: set())
    parent: "SymbolicTableau" = field(default_factory=lambda: None, repr=False)
    lt: "SymbolicTableau" = field(default_factory=lambda: None, repr=False)
    ge: "SymbolicTableau" = field(default_factory=lambda: None, repr=False)
    use_symbols: bool = field(default_factory=lambda: True, repr=False)

    def __str__(self):
        return pformat(self)

    def _check_branch_lt(self, expr):
        assert expr.free_symbols <= self.sym_vars
        pos_constraints = set(sp.simplify(self.pos_constraints))
        neg_constraints = set(sp.simplify(self.neg_constraints | {expr}))
        if check_constraints_feasible(
            list(c >= 0 for c in pos_constraints)
            + list(c <= -EPS for c in neg_constraints),
            list(self.sym_vars),
        ):
            return neg_constraints
        else:
            return None

    def _check_branch_ge(self, expr):
        assert expr.free_symbols <= self.sym_vars
        pos_constraints = set(sp.simplify(self.pos_constraints | {expr}))
        neg_constraints = set(sp.simplify(self.neg_constraints))
        if check_constraints_feasible(
            list(c >= 0 for c in pos_constraints)
            + list(c <= -EPS for c in neg_constraints),
            list(self.sym_vars),
        ):
            return pos_constraints
        else:
            return None

    def _branch_lt(self, expr):
        if constraints := self._check_branch_lt(expr):
            self.lt = dataclasses.replace(
                self,
                neg_constraints=constraints,
                lt=None,
                ge=None,
                parent=self,
            )
            BRANCHES.append((expr, self))
        return self.lt

    def _branch_ge(self, expr):
        if constraints := self._check_branch_ge(expr):
            # assert expr not in self.branches
            self.ge = dataclasses.replace(
                self,
                pos_constraints=constraints,
                lt=None,
                ge=None,
                parent=self,
            )
            BRANCHES.append((expr, self))
        return self.ge

    def _get_obj_coeffs(self):
        objective_coeffs = self.tableau[-1, :-1]
        if self.use_symbols:
            objective_coeffs = unitize_syms(objective_coeffs, self.domain_vars)
        objective_coeffs = [
            (j, sp.simplify(coeff))
            for j, coeff in enumerate(objective_coeffs)
            if coeff != 0
        ]
        return objective_coeffs

    def get_obj_coeff(self, col):
        return next((j, coeff) for j, coeff in self._get_obj_coeffs() if j == col)

    def find_pivot_column(self):
        objective_coeffs = self._get_obj_coeffs()
        for j, coeff in objective_coeffs:
            if (
                coeff not in self.neg_constraints | self.pos_constraints
            ) and self._check_branch_lt(coeff):
                return j, coeff
        return None

    def find_pivot_row(self, col_idx):
        piv_col = self.tableau[:-1, col_idx]
        if self.use_symbols:
            piv_col = unitize_syms(piv_col, self.domain_vars)
        if all(a <= 0 for a in piv_col):
            raise UnboundedProblem
        sol_col = self.tableau[:-1, -1]
        if self.use_symbols:
            sol_col = unitize_syms(sol_col, self.tableau.free_symbols)
        ratios = [a / b if b > 0 else sp.oo for a, b in zip(sol_col, piv_col)]
        piv_row_idx = np.argmin(ratios)
        assert ratios[piv_row_idx] != sp.oo
        return piv_row_idx

    def pivot(self, piv_row_idx, piv_col_idx):
        def _pivot():
            M = self.tableau.copy()
            piv_val = M[piv_row_idx, piv_col_idx]
            piv_row = M[piv_row_idx, :] / piv_val
            for i in range(M.rows):
                if i == piv_row_idx:
                    continue
                M[i, :] -= piv_row * M[i, piv_col_idx]
                if M[i, -1].could_extract_minus_sign():
                    M[i, :] *= -1
                M[i, :] = sp.simplify(M[i, :])

            M[piv_row_idx, :] = piv_row * (
                piv_val.free_symbols.pop() if piv_val.free_symbols else 1
            )
            return M

        tableau = _pivot()
        # self.tableau = lcm_tableau(tableau)
        self.tableau = tableau

    def __getitem__(self, item):
        return self.tableau[item]

    def copy(self):
        return dataclasses.replace(self)

    def backup(self):
        last_expr, tab = BRANCHES.pop()
        while BRANCHES and EXPLORED and last_expr in EXPLORED:
            last_expr, tab = BRANCHES.pop()
        right = tab.branch_ge(last_expr)
        EXPLORED.add(last_expr)
        return dataclasses.replace(right)


