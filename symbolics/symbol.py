import dataclasses
import random
from dataclasses import dataclass, field
from pprint import pformat

import numpy as np
import sympy as sp
from sympy import Add
from sympy.core.numbers import IntegerConstant
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean

from symbolics.simplex import UnboundedProblem
from symbolics.util import (
    symbolic_pivot,
    check_constraints_feasible,
    unitize_syms,
)

EPS = 1


@dataclass
class SymbolicTableau:
    tableau: sp.Matrix
    domain_vars: set[sp.Symbol]
    sym_vars: set[sp.Symbol]
    constraints: dict[sp.Expr] = field(default_factory=lambda: {})
    parent: "SymbolicTableau" = field(default_factory=lambda: None, repr=False)
    use_symbols: bool = field(default_factory=lambda: True, repr=False)

    def __str__(self):
        return pformat(self)

    def __getitem__(self, item):
        return self.tableau[item]

    def copy(self):
        return dataclasses.replace(self)

    def _check_branch(self, new_constraint):
        assert new_constraint.free_symbols <= self.sym_vars, new_constraint
        if new_constraint in self.constraints:
            return None

        constraints = dict(self.constraints)
        constraints[new_constraint] = True
        if check_constraints_feasible(
            list(constraints.keys()),
            list(self.sym_vars),
        ):
            return constraints
        else:
            return None

    def branch_lt(self, expr):
        expr = expr.rewrite(Add, evaluate=False)
        if constraints := self._check_branch(expr <= -EPS):
            lt = dataclasses.replace(
                self,
                constraints=constraints,
                parent=self,
            )
            return lt
        else:
            return None

    def branch_ge(self, expr):
        expr = expr.rewrite(Add, evaluate=False)
        if constraints := self._check_branch(expr >= 0):
            ge = dataclasses.replace(
                self,
                constraints=constraints,
                parent=self,
            )
            return ge
        else:
            return None

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
        random.shuffle(objective_coeffs)
        for j, coeff in objective_coeffs:
            coeff = coeff.rewrite(Add, evaluate=False)
            is_neg = coeff <= -EPS
            if isinstance(is_neg, Relational):
                if self._check_branch(is_neg):
                    return j, coeff
            elif isinstance(is_neg, Boolean):
                return (j, coeff) if bool(is_neg) else None
            else:
                raise NotImplementedError

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
        tableau = symbolic_pivot(self.tableau, piv_row_idx, piv_col_idx)
        self.tableau = tableau


def solve(tableau):
    explored: set[sp.Expr] = set()
    branches: list[sp.Expr] = []

    for i in range(100):
        # TODO there needs to be some randomization in the pivot column selection
        # but it needs to play with backup
        if piv_col_idx_coeff := tableau.find_pivot_column():
            piv_col_idx, coeff = piv_col_idx_coeff
            if coeff.free_symbols:
                branches.append((coeff, tableau))
                tableau = tableau.branch_lt(coeff)
            piv_row_idx = tableau.find_pivot_row(piv_col_idx)
            tableau.pivot(piv_row_idx, piv_col_idx)
            # print("tab", tab)
        else:
            print(
                "answer",
                tableau.constraints.keys(),
                tableau[-1, -1],
            )

            # backtrack
            if branches:
                last_expr, tab = branches.pop()
                while branches and explored and last_expr in explored:
                    last_expr, tab = branches.pop()

                explored.add(last_expr)
                branches.append((last_expr, tab))
                right = tab.branch_ge(last_expr)
                if right is not None:
                    tableau = right
