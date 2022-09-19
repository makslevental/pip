import dataclasses
from dataclasses import dataclass, field
from functools import reduce
from pprint import pformat

import numpy as np
import sympy as sp
from scipy.optimize import linprog as scipy_linprog
from sympy import (
    parse_expr,
    Interval,
    pprint as sym_pprint,
    Symbol,
    solve_univariate_inequality,
    Equality,
    Add,
    Matrix,
)
from sympy.parsing.sympy_parser import standard_transformations, convert_equals_signs
from sympy.solvers.solveset import linear_coeffs

from simplex import (
    linprog,
    disp_tableau,
    find_pivot,
    sweep,
    UnboundedProblem,
    normalize_tableau,
)

EPS = 1


def lcm_tableau(tableau):
    lcm = set()
    for el in tableau:
        if el.free_symbols:
            print()
        coeffs = list(el.simplify().as_coefficients_dict().values())
        for c in coeffs:
            if c.is_Rational and -1 < c < 1 and c != 0:
                lcm.add(c)
    if lcm:
        tableau *= np.lcm.reduce(list(c.denominator for c in lcm))
    return tableau


def get_var_coeffs(eqn, vars):
    var_coeffs = sorted(
        tuple((d, eqn.coeff(d)) for d in vars),
        key=lambda d: d[0].name if d[0].has_free() else float("inf"),
    )
    constant = reduce(
        lambda x, y: x + y,
        [float(term) for term in eqn.as_ordered_terms() if not term.free_symbols],
        0,
    )
    return var_coeffs, (1, constant)


def eqn_to_tuple(eqn, vars):
    all_one_side = eqn.lhs - eqn.rhs
    return get_var_coeffs(all_one_side, vars)


def make_constraints_from_eqns(eqns, domain_vars, range_var, use_symbols):
    constraints = []
    for eqn in eqns:
        if range_var in eqn.free_symbols:
            assert eqn.is_Equality
            continue

        coeffs, constant = eqn_to_tuple(eqn, domain_vars)
        if use_symbols:
            coeffs = [sym * coeff for sym, coeff in coeffs]
        else:
            coeffs = [coeff for sym, coeff in coeffs]
        constraints.append((coeffs, eqn.rel_op, -constant[1]))

    return constraints


def build_tableau_from_eqns(
    eqns_str, domain_vars, range_var, symbol_vars, minimize=True, use_symbols=False
):
    domain_vars = {d: sp.Symbol(d, real=True) for d in sorted(domain_vars)}
    range_var = {range_var: sp.Symbol(range_var, real=True)}
    symbol_vars = {d: sp.Symbol(d, real=True) for d in sorted(symbol_vars)}

    def _parse_expr(exp):
        return parse_expr(
            exp,
            transformations=standard_transformations + (convert_equals_signs,),
            local_dict=domain_vars | range_var | symbol_vars,
        )

    eqns = [
        _parse_expr(eqn_str.strip())
        for eqn_str in eqns_str.strip().splitlines()
        if "#" not in eqn_str
    ]

    domain_vars = tuple(domain_vars.values())
    range_var = range_var.popitem()[1]
    symbol_vars = tuple(symbol_vars.values())

    objective = next(
        get_var_coeffs(eqn.rhs, domain_vars)[0]
        for eqn in eqns
        if range_var in eqn.free_symbols
    )
    if use_symbols:
        objective = [sym * coeff for sym, coeff in objective]
    else:
        objective = [coeff for sym, coeff in objective]
    constraints = make_constraints_from_eqns(eqns, domain_vars, range_var, use_symbols)

    tableau = linprog(
        **{
            "minimize" if minimize else "maximize": objective,
            "subject_to": constraints,
            "use_symbols": use_symbols,
        }
    )

    return tableau, domain_vars, symbol_vars


def check_param_coverage():
    intervals = list(PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP.keys())

    return intervals and reduce(
        lambda acc, val: acc.union(val), intervals[1:], intervals[0]
    ) == Interval(-sp.oo, sp.oo)


PARAMETER_DOMAIN_OBJECTIVE_VAL_MAP = {}


def test_single_symbol_objective():
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


def test_build_tableau_from_eqns():
    tableau_true = linprog(
        minimize=[-3 + 2 * sp.var("θ"), 3 - sp.var("θ"), 1],
        subject_to=[
            ([1, 2, -3], "<=", 5),
            ([2, 1, -4], "<=", 7),
            # ([1, 0, 0], ">=", 0),
            # ([0, 1, 0], ">=", 0),
            # ([0, 0, 1], ">=", 0),
        ],
    )
    tableau_test = build_tableau_from_eqns(
        f"""
    z = (-3 + 2*θ)*x0 + (3 - θ)*x1 + x2
    x0 + 2*x1 - 3*x2 <= 5
    2*x0 + 1*x1 - 4*x2 <= 7
    """,
        ["x0", "x1", "x2"],
        "z",
        ["θ"],
        use_symbols=False,
    )
    assert tableau_test == tableau_true
    sym_pprint(tableau_test)


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


def linear_ineq_to_matrix(inequalities, symbols):
    inequalities = list(sp.sympify(inequalities))
    for i, ineq in enumerate(inequalities):
        inequalities[i] = ineq.func(ineq.lhs.as_expr() - ineq.rhs.as_expr(), 0)

    A, b = [], []
    for i, f in enumerate(inequalities):
        if isinstance(f, (Equality, sp.LessThan, sp.GreaterThan)):
            f = f.rewrite(Add, evaluate=False)
        if isinstance(f, sp.GreaterThan):
            f = f.reversedsign
        coeff_list = linear_coeffs(f.lhs, *symbols)
        b.append(-coeff_list.pop())
        A.append(coeff_list)
    A, b = map(Matrix, (A, b))
    return A, b


def check_constraints_feasible(constraints, sym_vars):
    A, b = linear_ineq_to_matrix(constraints, sym_vars)
    A = np.array(A).astype(int)
    b = np.array(b).astype(int)
    # 3rd version here
    # https://en.wikipedia.org/wiki/Farkas%27_lemma#Variants
    # if there's a solution with obj value >= 0.0 then
    # a negative solution doesn't exist (since we're mining)
    # hence the original system is feasible
    # if there's a solution and it's negative then the original system
    # isn't feasible
    # if there's no solution because it's negative unbounded then
    # there exists a negative solution and therefore
    # the original solution isn't feasible
    res = scipy_linprog(
        b.T,
        A_eq=A.T,
        b_eq=np.zeros(len(A.T)),
        method="highs",
        bounds=[(0, None)],
        integrality=np.ones(len(b)),
    )
    if res.success:
        if res.fun >= 0.0:
            return True
        else:
            return False
    else:
        assert "At lower/fixed bound" in res.message
        return False


def unitize_syms(expr, vars):
    return expr.subs({s: 1 for s in vars})


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
        right = tab._branch_ge(last_expr)
        EXPLORED.add(last_expr)
        return dataclasses.replace(right)


def test_check_constraint_checker():
    x1 = sp.Symbol("x1", real=True)
    x2 = sp.Symbol("x2", real=True)
    print(
        check_constraints_feasible(
            [x1 - x2 <= -EPS, x1 - x2 >= 0, x1 <= -EPS], (x1, x2)
        )
    )
    print()
    print(check_constraints_feasible([x1 <= -EPS, x1 <= 0], (x1, x2)))
    print()
    print(check_constraints_feasible([x1 <= -EPS, x1 >= 0], (x1, x2)))


def test_manual_tree():
    tableau, domain_vars, (x1, x2) = build_tableau_from_eqns(
        eqns_str=f"""
    z = x1*l1 + x2*l2 + 1
    l1 + l2 <= 5
    -l1 <= 1
    -l2 <= 2
    -l1 + l2 <= 0
    # l1 >= 0
    # l2 >= 0
    """,
        domain_vars=["l1", "l2"],
        range_var="z",
        symbol_vars=["x1", "x2"],
        use_symbols=False,
    )

    tab1 = SymbolicTableau(tableau, set(domain_vars), {x1, x2})
    print("start tab", tab1)

    piv_col_idx, coeff = tab1.get_obj_coeff(col=0)
    tab2 = tab1._branch_lt(coeff)
    piv_row_idx = tab2.find_pivot_row(piv_col_idx)
    tab2.pivot(piv_row_idx, piv_col_idx)
    print("tab2", tab2)

    piv_col_idx, coeff = tab2.get_obj_coeff(col=1)
    tab3 = tab2._branch_lt(coeff)
    piv_row_idx = tab3.find_pivot_row(piv_col_idx)
    tab3.pivot(piv_row_idx, piv_col_idx)
    print("tab3", tab3)
    if piv_col_idx_coeff := tab3.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        piv_row_idx = tab3.find_pivot_row(piv_col_idx)
        tab3.pivot(piv_row_idx, piv_col_idx)
        print("tab3", tab3)
    else:
        print("answer", tab3[-1, -1])

    piv_col_idx, coeff = tab2.get_obj_coeff(col=1)
    tab4 = tab2._branch_ge(coeff)
    if piv_col_idx_coeff := tab4.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        piv_row_idx = tab4.find_pivot_row(piv_col_idx)
        tab4.pivot(piv_row_idx, piv_col_idx)
        print("tab4", tab4)
    else:
        print("answer", tab4[-1, -1])

    _, coeff = tab1.get_obj_coeff(col=0)
    tab5 = tab1._branch_ge(coeff)
    print("tab5", tab5)
    if piv_col_idx_coeff := tab5.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        piv_row_idx = tab5.find_pivot_row(piv_col_idx)
        tab6 = tab1._branch_lt(coeff)
        tab6.pivot(piv_row_idx, piv_col_idx)
        print("tab6", tab6)
    else:
        print("answer", tab5[-1, -1])

    if piv_col_idx_coeff := tab6.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        tab7 = tab6._branch_lt(coeff)
        piv_row_idx = tab7.find_pivot_row(piv_col_idx)
        tab7.pivot(piv_row_idx, piv_col_idx)
        print("tab7", tab7)
    else:
        print("answer", tab6[-1, -1])

    if piv_col_idx_coeff := tab7.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        tab8 = tab7._branch_lt(coeff)
        piv_row_idx = tab8.find_pivot_row(piv_col_idx)
        tab8.pivot(piv_row_idx, piv_col_idx)
        print("tab8", tab8)
    else:
        print("answer", tab7[-1, -1])

    _, coeff = tab6.get_obj_coeff(col=0)
    tab7 = tab6._branch_ge(coeff)
    print("tab7", tab7)
    if piv_col_idx_coeff := tab7.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        tab7 = tab7._branch_lt(coeff)
        piv_row_idx = tab7.find_pivot_row(piv_col_idx)
        tab7.pivot(piv_row_idx, piv_col_idx)
        print("tab7", tab7)
    else:
        print("answer", tab7[-1, -1])

    _, coeff = tab5.get_obj_coeff(col=1)
    tab8 = tab5._branch_ge(coeff)
    print("tab8", tab8)
    if piv_col_idx_coeff := tab8.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        tab8 = tab8._branch_lt(coeff)
        piv_row_idx = tab8.find_pivot_row(piv_col_idx)
        tab8.pivot(piv_row_idx, piv_col_idx)
        print("tab8", tab8)
    else:
        print("answer", tab8[-1, -1])


def test_auto_tree():
    tableau, domain_vars, (x1, x2) = build_tableau_from_eqns(
        eqns_str=f"""
    z = x1*l1 + x2*l2 + 1
    l1 + l2 <= 5
    -l1 <= 1
    -l2 <= 2
    -l1 + l2 <= 0
    # l1 >= 0
    # l2 >= 0
    """,
        domain_vars=["l1", "l2"],
        range_var="z",
        symbol_vars=["x1", "x2"],
        use_symbols=False,
    )

    tab = SymbolicTableau(tableau, set(domain_vars), {x1, x2})

    for i in range(10):
        if piv_col_idx_coeff := tab.find_pivot_column():
            piv_col_idx, coeff = piv_col_idx_coeff
            print(i, coeff)
            tab = tab._branch_lt(coeff)
            piv_row_idx = tab.find_pivot_row(piv_col_idx)
            tab.pivot(piv_row_idx, piv_col_idx)
            print("tab", tab)
        else:
            print(
                "answer",
                tab.neg_constraints,
                "< 0",
                tab.pos_constraints,
                ">= 0",
                tab[-1, -1],
            )
            tab = tab.backup()


if __name__ == "__main__":
    # test_check_constraint_checker()
    # test_build_tableau_from_eqns()
    # test_manual_tree()
    test_auto_tree()
