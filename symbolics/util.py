import time
from functools import reduce

import numpy as np
import sympy as sp
from scipy.optimize import linprog as scipy_linprog
from sympy import Equality, Matrix, parse_expr
from sympy.core import Mul, Expr, Add, Pow, Symbol, Number
from sympy.core.relational import Relational
from sympy.parsing.sympy_parser import standard_transformations, convert_equals_signs
from sympy.solvers.solveset import linear_coeffs
from z3 import Sqrt, Int, Solver, UDiv

from symbolics.simplex import linprog

# big_M = 1e9
big_M = sp.Symbol("M")


def _sympy_to_z3_rec(var_map, e):
    "recursive call for sympy_to_z3()"
    rv = None

    if not isinstance(e, (Expr, Relational)):
        raise RuntimeError("Expected sympy Expr: " + repr(e))

    if isinstance(e, Symbol):
        rv = var_map.get(e.name)
        if rv is None:
            raise RuntimeError("No var was corresponds to symbol '" + str(e) + "'")
    elif isinstance(e, Number):
        rv = int(e)
    elif isinstance(e, Relational):
        lhs = _sympy_to_z3_rec(var_map, e.lhs)
        rhs = _sympy_to_z3_rec(var_map, e.rhs)
        rv = {
            "<=": lhs <= rhs,
            "<": lhs < rhs,
            ">=": lhs >= rhs,
            ">": lhs > rhs,
        }[e.rel_op]
    elif isinstance(e, Mul):
        if isinstance(e.args[0], Pow) and e.args[0].args[1] == -1:
            assert not isinstance(e.args[1], Pow)
            e = e.args[1] / e.args[0]
        rv = _sympy_to_z3_rec(var_map, e.args[0])
        for child in e.args[1:]:
            if isinstance(child, Pow) and child.args[1] == -1:
                rv /= _sympy_to_z3_rec(var_map, child.args[0])
            else:
                rv *= _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, Add):
        rv = _sympy_to_z3_rec(var_map, e.args[0])
        for child in e.args[1:]:
            rv += _sympy_to_z3_rec(var_map, child)
    # elif isinstance(e, Pow):
    #     term = _sympy_to_z3_rec(var_map, e.args[0])
    #     exponent = _sympy_to_z3_rec(var_map, e.args[1])
    #
    #     if exponent == 0.5:
    #         # sqrt
    #         rv = Sqrt(term)
    #     else:
    #         rv = term ** exponent

    if rv is None:
        raise RuntimeError(
            "Type '"
            + str(type(e))
            + "' is not yet implemented for convertion to a z3 expresion. "
            + "Subexpression was '"
            + str(e)
            + "'."
        )

    return rv


def sympy_to_z3(sympy_var_list, sympy_exp):
    "convert a sympy expression to a z3 expression. This returns (z3_vars, z3_expression)"

    z3_vars = {}
    z3_var_map = {}

    for var in sympy_var_list:
        name = var.name
        z3_var = Int(name)
        z3_var_map[name] = z3_var
        z3_vars[name] = z3_var

    result_exp = _sympy_to_z3_rec(z3_var_map, sympy_exp)

    return z3_vars, result_exp


def check_constraints_feasible(constraints, sym_vars):
    s = Solver()
    z3_vars, result_exp1 = sympy_to_z3(sym_vars + (big_M,), constraints[0])
    big_M_constraint = sum([z for z_name, z in z3_vars.items() if z_name != big_M.name])
    s.add(big_M_constraint < z3_vars[big_M.name])

    s.add(result_exp1)
    for c in constraints[1:]:
        result_exp = _sympy_to_z3_rec(z3_vars, c)
        s.add(result_exp)

    start = time.monotonic()
    sat = s.check()
    print("len cons", len(constraints), "check time", time.monotonic() - start)
    return sat.r == 1


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
        }
    )

    return tableau, domain_vars, symbol_vars


def build_tableau_from_eqns_str(
    eqns_str,
    domain_var_names,
    range_var,
    symbol_var_names,
    minimize=True,
    use_symbols=False,
):
    domain_var_names = {d: sp.Symbol(d, real=True) for d in sorted(domain_var_names)}
    range_var = {range_var: sp.Symbol(range_var, real=True)}
    symbol_vars = {
        d: sp.Symbol(d, real=True) for d in sorted(symbol_var_names) if d != "M"
    }
    if "M" in symbol_var_names:
        symbol_vars["M"] = big_M

    def _parse_expr(exp):
        return parse_expr(
            exp,
            transformations=standard_transformations + (convert_equals_signs,),
            local_dict=domain_var_names | range_var | symbol_vars,
        )

    eqns = [
        _parse_expr(eqn_str.strip())
        for eqn_str in eqns_str.strip().splitlines()
        if "#" not in eqn_str
    ]

    domain_var_names = tuple(domain_var_names.values())
    range_var = range_var.popitem()[1]
    symbol_vars = tuple(symbol_vars.values())

    return build_tableau_from_eqns(
        eqns, domain_var_names, range_var, symbol_vars, minimize, use_symbols
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
