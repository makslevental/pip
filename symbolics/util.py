from functools import reduce

import numpy as np
import sympy as sp
from scipy.optimize import linprog as scipy_linprog
from sympy import Equality, Add, Matrix, parse_expr
from sympy.parsing.sympy_parser import standard_transformations, convert_equals_signs
from sympy.solvers.solveset import linear_coeffs

from symbolics.simplex import linprog


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


def build_tableau_from_eqns(
    eqns, domain_vars, range_var, symbol_vars, minimize, use_symbols
):
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
            "use_symbols": False,
        }
    )

    return tableau, domain_vars, symbol_vars


def build_tableau_from_eqns_str(
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

    return build_tableau_from_eqns(
        eqns, domain_vars, range_var, symbol_vars, minimize, use_symbols
    )


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


def make_constraints_from_eqns(eqns, domain_vars, range_var, use_symbols):
    constraints = []
    for eqn in eqns:
        if range_var in eqn.free_symbols:
            assert eqn.is_Equality
            continue

        all_one_side = eqn.lhs - eqn.rhs
        coeffs, constant = get_var_coeffs(all_one_side, domain_vars)
        if use_symbols:
            coeffs = [sym * coeff for sym, coeff in coeffs]
        else:
            coeffs = [coeff for sym, coeff in coeffs]
        constraints.append((coeffs, eqn.rel_op, -constant[1]))

    return constraints


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
