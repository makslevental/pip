import dataclasses
import sys
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

EPS = 100 * sys.float_info.epsilon


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
    _z, x = get_solution(evaled_context_tableau)
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


def linear_ineq_to_matrix(inequalities, *symbols):
    inequalities = sp.sympify(inequalities)
    for i, ineq in enumerate(inequalities):
        inequalities[i] = ineq.func(ineq.lhs.as_expr() - ineq.rhs.as_expr(), 0)

    A, b = [], []
    for i, f in enumerate(inequalities):
        if isinstance(f, (Equality, sp.LessThan, sp.GreaterThan)):
            f = f.rewrite(Add, evaluate=False)
        coeff_list = linear_coeffs(f.lhs, *symbols)
        b.append(-coeff_list.pop())
        A.append(coeff_list)
    A, b = map(Matrix, (A, b))
    return A, b


@dataclass
class ParamTree:
    expr: sp.Expr
    param_vars: tuple[sp.Symbol]
    # param_constraints: dict[str, Set] = None
    param_constraints: list[sp.Expr] = field(default_factory=lambda: [])
    parent: "ParamTree" = None
    lt: "ParamTree" = None
    ge: "ParamTree" = None
    use_symbols: bool = True

    # def __post_init__(self):
    #     if self.param_constraints is None:
    #         self.param_constraints = {v: Reals for v in self.param_vars}

    def clone_new_expr(self, expr):
        return ParamTree(
            expr,
            tuple(sorted(self.param_vars, key=lambda v: v.name)),
            self.param_constraints,
            self,
        )

    def branch_lt(self):
        # sol = solve_linear_inequalities((self.expr < 0,), self.param_vars)
        self.lt = ParamTree(
            self.expr,
            self.param_vars,
            self.param_constraints + [self.expr <= -EPS],
            # {
            #     v: ((sol[v] if v in sol else self.param_constraints[v])
            #         & self.param_constraints[v]).simplify()
            #     for v in self.param_vars
            # },
            self,
        )
        return self.lt

    def branch_ge(self):
        # sol = solve_linear_inequalities((self.expr >= 0,), self.param_vars)
        self.ge = ParamTree(
            self.expr,
            self.param_vars,
            self.param_constraints + [self.expr >= 0],
            # {
            #     v: ((sol[v] if v in sol else self.param_constraints[v])
            #         & self.param_constraints[v]).simplify()
            #     for v in self.param_vars
            # },
            self,
        )
        return self.ge

    def solve_constraints(self):
        pass

    def __str__(self):
        return f"ParamTree(expr={self.expr}, param_vars={self.param_vars}, constraints={self.param_constraints})"


def check_constraints_feasible(constraints, sym_vars):
    A, b = linear_ineq_to_matrix(constraints, sym_vars)
    A = np.array(A)
    b = np.array(b)
    c = [0] * len(sym_vars)
    res = scipy_linprog(c, A_ub=A, b_ub=b, method="highs", bounds=[(None, None)])
    return res.success


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
        constraints = self.neg_constraints | {expr <= -EPS}
        if check_constraints_feasible(constraints, self.sym_vars):
            return constraints
        else:
            return None

    def _check_branch_ge(self, expr):
        assert expr.free_symbols <= self.sym_vars
        constraints = self.neg_constraints | {expr >= 0}
        if check_constraints_feasible(constraints, self.sym_vars):
            return constraints
        else:
            return None

    def _branch_lt(self, expr):
        assert self.lt is None
        if constraints := self._check_branch_lt(expr):
            self.lt = dataclasses.replace(self, neg_constraints=constraints)
        return self.lt

    def _branch_ge(self, expr):
        assert self.ge is None
        if constraints := self._check_branch_ge(expr):
            self.ge = dataclasses.replace(self, neg_constraints=constraints)
        return self.ge

    def find_pivot_column(self):
        objective_coeffs = tuple(map(lambda e: get_var_coeffs(e, self.domain_vars), self.tableau[-1, :]))
        print(objective_coeffs)



if __name__ == "__main__":
    # test_build_tableau_from_eqns()
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
        use_symbols=True,
    )

    tab = SymbolicTableau(tableau, set(domain_vars), {x1, x2})
    print(tab)
    tab.find_pivot_column()

    # x1 < 0
    lt = tab._branch_lt(x1)
    print(lt)

    # x1 < 0 => l1 entering
    # pivot = find_symbolic_pivot(tableau)

    # diff = lt.clone_new_expr(x2 - x1)
    # sym_pprint(diff)
    #
    # diff_lt = diff.branch_lt()
    # sym_pprint(diff_lt)
    #
    # A, b = linear_ineq_to_matrix(diff_lt.param_constraints, *diff_lt.param_vars)
    # A = np.array(A)
    # b = np.array(b)
    # c = [0] * len(diff_lt.param_vars)
    # res = scipy_linprog(c, A_ub=A, b_ub=b, method="highs", bounds=[(None, None)])
    # print(res.success)

    # A = sp.Matrix([
    #     [5, 4],
    #     [1, 5]
    # ])
    # sym_pprint(A)
    # sym_pprint(A.inv())
    # b = sp.Matrix([14, 7])
    # sym_pprint(A.inv() * b)

    #
    # test_single_symbol_objective()
    # ineq1 = parse_expr(
    #     "l1 + l2 <= 5",
    #     transformations=standard_transformations + (convert_equals_signs,),
    # )
    # ineq2 = parse_expr(
    #     "-l1 <= 1",
    #     transformations=standard_transformations + (convert_equals_signs,),
    # )
    # print()

    # expr = parse_expr('z <= x1*l1 + x2*l2', evaluate=False)
    # expr = parse_latex('z \leq x1*l1 + x2*l2')
    # tab = sp.Matrix([
    #     [],
    #     [],
    #     [],
    #
    # ])
    # test_single_symbol_constraints()

# b = {"c": "d", "d": 3}
# with suppress(Exception, SyntaxError):
#     (
#         c := b["c"],
#         d := b[c],
#     )
# print(c)
# print(d)
