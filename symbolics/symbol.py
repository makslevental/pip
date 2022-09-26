import dataclasses
import random
import warnings
from dataclasses import dataclass, field
from pprint import pformat

import numpy as np
import sympy as sp
from numpy.linalg import pinv, inv
from sympy import Add, pprint
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean

from symbolics.simplex import UnboundedProblem
from symbolics.util import (
    check_constraints_feasible,
    unitize_syms,
    big_M,
)

EPS = 0


def symbolic_pivot(tableau, piv_row_idx, piv_col_idx):
    M = tableau.copy()
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


@dataclass
class AugmentedSymbolicTableau:
    tableau: sp.Matrix
    domain_vars: tuple[sp.Symbol]
    sym_vars: tuple[sp.Symbol]
    constraints: dict[sp.Expr] = field(default_factory=lambda: {})
    use_symbols: bool = field(default_factory=lambda: True, repr=False)

    def __str__(self):
        return pformat(self)

    def __getitem__(self, item):
        return self.tableau[item]

    def copy(self):
        return dataclasses.replace(self)

    def _check_branch(self, new_constraint):
        assert new_constraint.free_symbols <= set(self.sym_vars) | {
            big_M
        }, new_constraint
        if new_constraint in self.constraints:
            return None

        constraints = dict(self.constraints)
        constraints[new_constraint] = True
        if check_constraints_feasible(
            list(constraints.keys()),
            self.sym_vars,
        ):
            return constraints
        else:
            return None

    def branch_lt(self, expr):
        expr = expr.rewrite(Add, evaluate=False)
        if constraints := self._check_branch(expr < -EPS):
            lt = dataclasses.replace(
                self,
                constraints=constraints,
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
        # random.shuffle(objective_coeffs)
        for j, coeff in objective_coeffs:
            coeff = coeff.rewrite(Add, evaluate=False)
            if coeff.free_symbols == {big_M}:
                continue
            is_neg = coeff < -EPS
            if isinstance(is_neg, Relational):
                if self._check_branch(is_neg):
                    return j, coeff
            elif isinstance(is_neg, Boolean):
                if bool(is_neg):
                    return j, coeff
            else:
                raise NotImplementedError

        return None

    def find_pivot_row(self, col_idx):
        piv_col = self.tableau[:-1, col_idx]
        piv_col = piv_col.applyfunc(lambda x: sp.limit(x, big_M, sp.oo))
        if self.use_symbols:
            piv_col = unitize_syms(piv_col, self.domain_vars)
        if all(a < 0 for a in piv_col):
            raise UnboundedProblem(piv_col)
        sol_col = self.tableau[:-1, -1]
        if self.use_symbols:
            sol_col = unitize_syms(sol_col, self.tableau.free_symbols)
        ratios = [a / b if b > 0 else sp.oo for a, b in zip(sol_col, piv_col)]
        piv_row_idx = np.argmin(ratios)
        if ratios[piv_row_idx] == sp.oo:
            return None
        if ratios[piv_row_idx] == 0:
            warnings.warn("0 pivot ratio")
        return piv_row_idx

    def pivot(self, piv_row_idx, piv_col_idx):
        self.tableau = symbolic_pivot(self.tableau, piv_row_idx, piv_col_idx)

    @staticmethod
    def _check_basic_col(col):
        nonz = col.nonzero()[0]
        if len(nonz) == 1 and col[nonz[0]] == 1:
            return nonz[0]
        else:
            return -1

    def _get_basic_columns(self):
        A = np.array(self.tableau[:-1, :-1])
        basic_cols = np.apply_along_axis(self._check_basic_col, 0, A)
        assert sum(basic_cols > -1) == A.shape[0]
        return {
            col_idx: row_idx
            for col_idx, row_idx in enumerate(basic_cols)
            if row_idx != -1
        }

    def get_current_soln(self):
        basic_cols = self._get_basic_columns()
        soln = []
        for i, _domain_var in enumerate(self.domain_vars):
            if i in basic_cols:
                val = self.tableau[basic_cols[i], -1]
            else:
                val = 0
            soln.append(val)

        return soln


def solve(tableau):
    print()
    sp.pprint(tableau.tableau, wrap_line=False)
    print()
    explored: set[sp.Expr] = set()
    branches: list[sp.Expr] = []

    for i in range(1000):
        # TODO there needs to be some randomization in the pivot column selection
        # but it needs to play with backup
        if piv_col_idx_coeff := tableau.find_pivot_column():
            piv_col_idx, coeff = piv_col_idx_coeff
            if coeff.free_symbols:
                branches.append((coeff, tableau))
                tableau = tableau.branch_lt(coeff)
            piv_row_idx = tableau.find_pivot_row(piv_col_idx)
            if piv_row_idx is None:
                continue
            else:
                tableau.pivot(piv_row_idx, piv_col_idx)
            # print("tab", tab)
        else:
            print()
            sp.pprint(tableau.tableau, wrap_line=False)
            print()
            print(
                "answer",
                list(tableau.constraints.keys()),
                tableau[-1, -1],
            )
            # https://math.stackexchange.com/questions/2818217/how-obtain-the-dual-variables-value-given-a-primal-solution
            # https://www.matem.unam.mx/~omar/math340/comp-slack.html
            # curr_tableau, curr_b, curr_c = map(
            #     lambda x: to_int(np.array(x)), tableau.get_unaug_tableau()
            # )
            # dual_soln = tableau.tableau[-1, num_dual_vars:-1]
            # print(str(dual_soln).replace("2147483647", "M"))
            # basic_cols = tableau._get_basic_columns()
            # orig_submatrix = to_int(orig_tableau[:-1, list(basic_cols.keys())])
            # orig_sub_c = orig_tableau[-1, list(basic_cols.keys())]
            # dual_soln = sp.Matrix(orig_sub_c @ inv(orig_submatrix))
            # pprint(dual_soln)
            # print(current_soln)
            # dual_tableau = orig_tableau.T
            #
            # # complementary_slackness => non-zero => no slack in dual constraints
            # tight_dual_constraints = dual_tableau[current_soln > 0]
            #
            # # complementary_slackness => slack original constraint => dual variable = 0
            # # thus drop those columns
            # slack_orig_constraints = ((orig_tableau @ current_soln) < orig_b.T)[0]
            # reduced_dual_tableau = tight_dual_constraints[:, ~slack_orig_constraints]
            #
            # dual_soln = pinv(reduced_dual_tableau) @ orig_c[0]
            # print(dual_soln)
            # backtrack
            if branches:
                last_expr, tab = branches.pop()
                while branches and explored and last_expr in explored:
                    last_expr, tab = branches.pop()
                if last_expr in explored:
                    break

                explored.add(last_expr)
                branches.append((last_expr, tab))
                right = tab.branch_ge(last_expr)
                if right is not None:
                    tableau = right
