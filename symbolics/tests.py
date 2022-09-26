import sympy as sp
from sympy import pprint as sym_pprint

from symbol import EPS, AugmentedSymbolicTableau
from symbolics.symbolic_simplex import linprog, solve
import symbolics.symbolic_simplex
from util import check_constraints_feasible, build_tableau_from_eqns_str


def test_build_tableau_from_eqns():
    theta = sp.var("θ")
    tableau_true = linprog(
        minimize=[-3 + 2 * theta, 3 - theta, 1],
        subject_to=[
            ([1, 2, -3], "<=", 5),
            ([2, 1, -4], "<=", 7),
            # ([1, 0, 0], ">=", 0),
            # ([0, 1, 0], ">=", 0),
            # ([0, 0, 1], ">=", 0),
        ],
    )
    tableau_test, _domain_vars, _symbol_vars = build_tableau_from_eqns_str(
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
    assert (tableau_test - tableau_true).simplify() is None
    sym_pprint(tableau_test)


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
    tableau, domain_vars, (x1, x2) = build_tableau_from_eqns_str(
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

    tab1 = AugmentedSymbolicTableau(tableau, domain_vars, (x1, x2))
    print("start tab", tab1)

    piv_col_idx, coeff = tab1.get_obj_coeff(col=0)
    tab2 = tab1.branch_lt(coeff)
    piv_row_idx = tab2.find_pivot_row(piv_col_idx)
    tab2.pivot(piv_row_idx, piv_col_idx)
    # print("tab2", tab2)

    piv_col_idx, coeff = tab2.get_obj_coeff(col=1)
    tab3 = tab2.branch_lt(coeff)
    piv_row_idx = tab3.find_pivot_row(piv_col_idx)
    tab3.pivot(piv_row_idx, piv_col_idx)
    # print("tab3", tab3)
    if piv_col_idx_coeff := tab3.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        piv_row_idx = tab3.find_pivot_row(piv_col_idx)
        tab3.pivot(piv_row_idx, piv_col_idx)
        # print("tab3", tab3)
    else:
        print("answer", list(tab3.constraints.keys()), tab3[-1, -1])

    piv_col_idx, coeff = tab2.get_obj_coeff(col=1)
    tab4 = tab2.branch_ge(coeff)
    if piv_col_idx_coeff := tab4.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        piv_row_idx = tab4.find_pivot_row(piv_col_idx)
        tab4.pivot(piv_row_idx, piv_col_idx)
        # print("tab4", tab4)
    else:
        print("answer", list(tab4.constraints.keys()), tab4[-1, -1])

    _, coeff = tab1.get_obj_coeff(col=0)
    tab5 = tab1.branch_ge(coeff)
    # print("tab5", tab5)
    if piv_col_idx_coeff := tab5.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        piv_row_idx = tab5.find_pivot_row(piv_col_idx)
        tab6 = tab5.branch_lt(coeff)
        tab6.pivot(piv_row_idx, piv_col_idx)
        # print("tab6", tab6)
    else:
        print("answer", list(tab5.constraints.keys()), tab5[-1, -1])

    if piv_col_idx_coeff := tab6.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        tab7 = tab6.branch_lt(coeff)
        piv_row_idx = tab7.find_pivot_row(piv_col_idx)
        tab7.pivot(piv_row_idx, piv_col_idx)
        # print("tab7", tab7)
    else:
        print("answer", list(tab6.constraints.keys()), tab6[-1, -1])

    if piv_col_idx_coeff := tab7.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        tab8 = tab7.branch_lt(coeff)
        piv_row_idx = tab8.find_pivot_row(piv_col_idx)
        tab8.pivot(piv_row_idx, piv_col_idx)
        # print("tab8", tab8)
    else:
        print("answer", list(tab7.constraints.keys()), tab7[-1, -1])

    _, coeff = tab6.get_obj_coeff(col=0)
    tab7 = tab6.branch_ge(coeff)
    # print("tab7", tab7)
    if piv_col_idx_coeff := tab7.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        tab7 = tab7.branch_lt(coeff)
        piv_row_idx = tab7.find_pivot_row(piv_col_idx)
        tab7.pivot(piv_row_idx, piv_col_idx)
        # print("tab7", tab7)
    else:
        print("answer", list(tab7.constraints.keys()), tab7[-1, -1])

    _, coeff = tab5.get_obj_coeff(col=1)
    tab8 = tab5.branch_ge(coeff)
    # print("tab8", tab8)
    if piv_col_idx_coeff := tab8.find_pivot_column():
        piv_col_idx, coeff = piv_col_idx_coeff
        tab8 = tab8.branch_lt(coeff)
        piv_row_idx = tab8.find_pivot_row(piv_col_idx)
        tab8.pivot(piv_row_idx, piv_col_idx)
        # print("tab8", tab8)
    else:
        print("answer", list(tab8.constraints.keys()), tab8[-1, -1])


def test_auto_tree():
    tableau, domain_vars, (x1, x2) = build_tableau_from_eqns_str(
        eqns_str=f"""
    z = x1*l1 + x2*l2
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
        minimize=True,
    )

    symbolics.symbolic_simplex.sym_vars = (x1, x2)
    for sol in solve(tableau):
        pass


if __name__ == "__main__":
    # test_build_tableau_from_eqns()
    # test_check_constraint_checker()
    # test_manual_tree()
    print()
    test_auto_tree()
