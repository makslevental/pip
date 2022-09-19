# public solve_linear_inequalities and LP (to get a better name)

############ code for maximizing linear expression under linear constraints

from sympy import *
from sympy.abc import *


def _positive_exprs(inequalities, symset):
    """return set of expressions that are positive from the list
    of inequalities and expressions (assumed to be positive).

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> _positive_exprs([x, y > z, z < x + y], {x, y, z})
    [x, y - z, x + y - z]
    """
    from sympy.core.relational import Relational
    from sympy import Expr, Basic
    from sympy.solvers.solveset import linear_coeffs
    from sympy.core.sympify import _sympify
    assert type(symset) is set
    eqs = []
    for i in inequalities:
        if isinstance(i, Relational):
            if i.gts != i.lhs:
                i = i.reversed
            # not sure this is a necessary constraint
            # if i.rel_op != ">":
            #    raise ValueError('inequality must be strict')
            i = i.lhs - i.rhs
        elif not isinstance(i, Expr):
            if isinstance(i, Basic):
                raise ValueError('expecting Relational or expression')
            else:
                i = _sympify(i)
        x = list(i.free_symbols & symset)
        c = linear_coeffs(i, *x)
        eqs.append(Add(*[i * j for i, j in zip(c, x + [1])]))
    return eqs


class SimplexError(Exception):
    "error raised in _simplex"
    pass


def _pivot(M, i, j):
    Mi, Mj, Mij = M[i, :], M[:, j], M[i, j]
    MM = M - Mj * (Mi / Mij)
    MM[i, :] = Mi / Mij
    MM[:, j] = -Mj / Mij
    MM[i, j] = 1 / Mij
    return MM


def _simplex(M, R, S, random_seed=0):
    import random
    rand = random.Random(x=random_seed)
    while True:
        B = M[:-1, -1]
        A = M[:-1, :-1]
        C = M[-1, :-1]
        if all(B[i] >= 0 for i in range(B.rows)):
            piv_cols = []
            for j in range(C.cols):
                if C[j] < 0:
                    piv_cols.append(j)
            if not piv_cols:
                return M, R, S
            rand.shuffle(piv_cols)
            j0 = piv_cols[0]
            piv_rows = []
            for i in range(A.rows):
                if A[i, j0] > 0:
                    ratio = B[i] / A[i, j0]
                    piv_rows.append((ratio, i))
            if not piv_rows:
                raise SimplexError(filldedent('''
                    Objective function can assume arbitrarily
                    large values at feasible vectors'''))
            _ = sorted(piv_rows)
            r0 = _[0][0]
            piv_rows = []
            for i in _:
                if i[0] != r0:
                    break
                piv_rows.append(i)
            rand.shuffle(piv_rows)
            _, i0 = piv_rows[0]
            M = _pivot(M, i0, j0)
            R[j0], S[i0] = S[i0], R[j0]
        else:
            for k in range(B.rows):
                if B[k] < 0:
                    break
            piv_cols = []
            for j in range(A.cols):
                if A[k, j] < 0:
                    piv_cols.append(j)
            if not piv_cols:
                raise SimplexError('empty set of constraints')
            rand.shuffle(piv_cols)
            j0 = piv_cols[0]
            ratio = B[k] / A[k, j0]
            piv_rows = [(ratio, k)]
            for i in range(A.rows):
                if A[i, j0] > 0 and B[i] > 0:
                    ratio = B[i] / A[i, j0]
                    piv_rows.append((ratio, i))
            piv_rows = sorted(piv_rows, key=lambda x: (x[0], x[1]))
            piv_rows = [(ratio, i) for ratio, i in piv_rows if ratio == piv_rows[0][0]]
            rand.shuffle(piv_rows)
            _, i0 = piv_rows[0]
            M = _pivot(M, i0, j0)
            R[j0], S[i0] = S[i0], R[j0]


def linear_programming(M):
    """
    When x is a column vector of variables, y is a row vector of dual variables,
    and when the objective is to either

    a) maximize `F = C.T*x + D` constrained by `A*x <= B` and `x >= 0`, or
    b) minimize `f = y.T*B + D` constrained by `y.T*A >= C.T` and `y >= 0`,

    this method returns a tuple ``(o, A, a)`` where ``o`` is the
    maximum value of ``F`` (found at ``x`` point ``A``) and ``a``
    is the ``y`` point that minimize the ``f``.
    The input ``M`` is ``Matrix([[A, B], [-C.T, D]])``.

    Examples
    ========

    >>> from sympy.matrices.dense import Matrix
    >>> from sympy.solvers.inequalities import linear_programming
    >>> from sympy import symbols
    >>> from sympy.abc import x, y

    >>> A = Matrix([[1, 2], [4, 2], [-1, 1]])
    >>> B = Matrix([4, 12, 1])
    >>> C = Matrix([3, 5])
    >>> D = Matrix([7])
    >>> x = Matrix([x, y])
    >>> y = Matrix(symbols('y:3'))

    The function and constraints:

    >>> F = (C.T*x + D)[0]; F
    3*x + 5*y + 7
    >>> [i <= 0 for i in A*x - B]
    [x + 2*y - 4 <= 0, 4*x + 2*y - 12 <= 0, -x + y - 1 <= 0]

    The corresponding dual system and constraints:

    >>> f = (y.T*B + D)[0]; f
    4*y0 + 12*y1 + y2 + 7
    >>> [i>=0 for i in y.T*A-C.T]
    [y0 + 4*y1 - y2 - 3 >= 0, 2*y0 + 2*y1 + y2 - 5 >= 0]

    >>> M = Matrix([[A, B], [-C.T, D]]); M
    Matrix([[1, 2, 4], [4, 2, 12], [-1, 1, 1], [-3, -5, 7]])
    >>> t = max, argmax, argmin_dual = linear_programming(M); t
    (55/3, [8/3, 2/3], [7/3, 1/6, 0])
    >>> F.subs(dict(zip(x, t[1]))) == f.subs(dict(zip(y, t[-1]))) == t[0]
    True
    """
    r_orig = ['x_{}'.format(j) for j in range(M.cols - 1)]
    s_orig = ['y_{}'.format(i) for i in range(M.rows - 1)]
    M, r, s = _simplex(M, r_orig[:], s_orig[:])
    argmax = []
    argmin_dual = []
    for _ in r_orig:
        for i, x in enumerate(s):
            if _ == x:
                argmax.append(M[i, -1])
                break
        else:
            argmax.append(Integer(0))
    for _ in s_orig:
        for i, x in enumerate(r):
            if _ == x:
                argmin_dual.append(M[-1, i])
                break
        else:
            argmin_dual.append(Integer(0))
    return M[-1, -1], argmax, argmin_dual


def LP(func, inequalities, syms, unbound=None):
    """
    Return the maximum value of func under constraints of the
    given inequalities on the nonnegative symbols, the values
    of the symbols giving that maximum, and the values of the
    symbols giving the minimum of the related dual of the
    system.

    Parameters
    ==========

    func - the expression to be maximized
    inequalities - relationals or expressions (assumed to be nonnegative)
    sysm - symbols assumed to be nonnegative
    unbound - symbols without bound

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.solvers.inequalities import LP

    For nonnegative values of x1 and x2 under constraints

        x1 + 2*x2   <= 4
        4*x1 + 2*x2 <= 12
        -x1 + x2    <= 1

    maximize 3*x1 + 5*x2 + 7 or, equivalently, minimize
    4*y1 + 12*y2 + y3 + 7 under the constraints of nonnegative
    values of yi and

        y1 + 4*y2 - y3 >= 3
        2*y1 + 2*y2 + y3 >= 5

    >>> x1, x2 = symbols('x1, x2')
    >>> func = 3*x1 + 5*x2 + 7
    >>> constr = [
    ... x1 + 2*x2 <= 4,     # inequalities can be written
    ... -4*x1 >= 12 + 2*x2, # in any format; expressions are
    ... 1 + x1 - x2]        # assumed to be nonnegative
    ...
    >>> LP(func, constr, [x1, x2])
    (55/3,
    {x: 8/3, y: 2/3},
    4*_y0 + 12*_y1 + _y2 + 7,
    [_y0 + 4*_y1 - _y2 - 3 >= 0, 2*_y0 + 2*_y1 + _y2 - 5 >= 0],
    {_y0: 7/3, _y1: 1/6, _y2: 0})
    >>> opt, max, dual, constr, min = _

    >>> func.subs(max)
    55/3
    >>> dual.subs(min)
    55/3

    If a variable is not restricted to being nonnegative, pass it
    as unbound.

    >>> LP(func.subs(x2, -x2), [i.subs(x2, -x2) for i in const], [x1], [x2])
    (55/3,
    {x1: 8/3, x2: -2/3},
    4*_y0 + 12*_y1 + _y2 + 7,
    [-2*_y0 - 2*_y1 - _y2 + 5 >= 0, 2*_y0 + 2*_y1 + _y2 - 5 >= 0, _y0 + 4*_y1 - _y2 - 3 >= 0],
    {_y0: 7/3, _y1: 1/6, _y2: 0})

    References
    ==========

    [1] Ferguson, Thomas S., "Linear Programming - A Concise Introduction",
    https://www.math.ucla.edu/~tom/LP.pdf.
    """
    from sympy.solvers.solveset import linear_coeffs, linear_eq_to_matrix
    from sympy.utilities.iterables import sift
    from sympy.core.symbol import symbols
    unbound = unbound or []
    symset = set(syms + unbound)
    assert len(symset) == len(syms + unbound), 'duplicate symbols'
    in_syms = list(ordered(symset))
    nonpos = []
    defs, inequalities = sift(inequalities, lambda x: x.is_Equality, binary=True)
    for i in _positive_exprs(inequalities, symset):
        if not i or i == True:
            continue
        if i == False:
            return  # no solution
        if i.is_number:
            assert i.is_zero is not None
            return  # no solution
        nonpos.append(-i)
    if defs:
        # eliminate defs
        sol = solve(defs, symset, dict=True)
        assert len(sol) == 1
        sol = sol[0]
        symset -= set(sol)
        nonpos = [i.xreplace(sol) for i in nonpos]
        if any(i.is_negative for i in nonpos):
            return  # no solution
    if unbound:
        nn = numbered_symbols('nn')
        for u in unbound:
            reps = {u: next(nn) - next(nn)}
            symset -= {u}
            symset.update(reps[u].free_symbols)
            nonpos = [expand_mul(i.xreplace(reps)) for i in nonpos]
            func = func.xreplace(reps)
    x = list(func.free_symbols & symset)
    c = linear_coeffs(func, *x)
    nonpos.append(-Add(*[i * j for i, j in zip(c, x + [1])]))
    syms = list(ordered(symset))
    a, rhs = linear_eq_to_matrix(nonpos, syms)
    M = Matrix([[a, rhs]])
    A = M[:-1, :-1]
    B = M[:-1, -1]
    C = M[-1, :-1]
    D = M[-1:, -1:]
    ys = symbols('y:%s' % (a.rows - 1), cls=Dummy)
    y = Matrix(ys)
    a, amax, dmin = linear_programming(M)
    r = dict(zip(syms, amax))
    if unbound:
        new = {}
        for i in in_syms:
            if i in r:
                new[i] = r.pop(i)
            else:
                new[i] = reps[i].xreplace(r)
        r = new
    dfunc = list(y.T * B + D)[0]
    dual = [i[0] >= 0 for i in (A.T * y + C.T).tolist()]
    return a, r, dfunc, dual, dict(zip(ys, dmin))




#################### solving inequalities ##############################

def _find_pivot(inequalities, symbols):
    """
    Return a variable that has at least two coefficients with opposite
    sign in a system of inequalities.

    Examples
    ========

    >>> from sympy.solvers.inequalities import _find_pivot
    >>> from sympy.abc import x, y, z
    >>> symbols = {x, y, z}
    >>> eq1 = 2*x - 3*y + z + 1
    >>> eq2 = x - y + 2*z - 2
    >>> eq3 = x + y + 3*z + 4
    >>> eq4 = x - z
    >>> inequalities = [eq1, eq2, eq3, eq4]
    >>> _find_pivot(inequalities, symbols)
    y
    """
    memory = {}
    symset = set(symbols)
    for eq in inequalities:
        for s in ordered(eq.free_symbols & symset):
            if not s in memory:
                memory[s] = [False, False]
            coeff = eq.coeff(s)
            if coeff > 0:
                memory[s][0] = True
            else:
                memory[s][1] = True
            if memory[s] == [True, True]:
                return s


def _split_min_max(inequalities, pivot):
    """return expressions that are less than or greater than the pivot
    (have a coefficient on the pivot that is negative or positive).
    Inequalities that do not contain a pivot are returned as a list.

    Examples
    ========

    >>> from sympy.solvers.inequalities import _split_min_max
    >>> from sympy.abc import x, y, z
    >>> eq1 = 2*x - 3*y + z + 1
    >>> eq2 = x - y + 2*z - 2
    >>> eq3 = x + y + 3*z + 4
    >>> eq4 = x - z
    >>> inequalities = [eq1, eq2, eq3, eq4]
    >>> pivot = y
    >>> _split_min_max(inequalities, pivot)
    (Min(2*x/3 + z/3 + 1/3, x + 2*z - 2), -x - 3*z - 4, [x - z])
    """
    greater_than = []
    lower_than = []
    extra = []
    for eq in inequalities:
        coeff = eq.coeff(pivot)
        a = (-eq + (pivot * coeff)) / coeff
        if coeff > 0:
            greater_than.append(a)
        elif coeff < 0:
            lower_than.append(a)
        else:
            extra.append(eq)
    return Min(*lower_than), Max(*greater_than), extra


def _merge_mins_maxs(mins, maxs, symbols):
    """Build the system of inequalities which verify that all equations
    of maxs are greater than those of mins.

    Examples
    ========

    >>> from sympy.solvers.inequalities import _merge_mins_maxs
    >>> from sympy.abc import x, z
    >>> from sympy import Min
    >>> symbols = {x, z}
    >>> maxs = -x - 3*z - 4
    >>> mins = Min((2*x + z + 1)/3, x + 2*z - 2)
    >>> _merge_mins_maxs(mins, maxs, symbols)
    [2*x + 5*z + 2, 5*x/3 + 10*z/3 + 13/3]
    """
    if not isinstance(mins, Min):
        mins = [mins]
    else:
        mins = mins.args

    if not isinstance(maxs, Max):
        maxs = [maxs]
    else:
        maxs = maxs.args
    return [collect(i - j, symbols) for i in mins for j in maxs]


def _fourier_motzkin(inequalities, symbols):
    """Eliminate variables of system of linear inequalities by using
    Fourier-Motzkin elimination algorithm

    Examples
    ========

    >>> from sympy.solvers.inequalities import _fourier_motzkin
    >>> from sympy.abc import x, y, z
    >>> symbols = {x, y, z}
    >>> eq1 = 2*x - 3*y + z + 1
    >>> eq2 = x - y + 2*z - 2
    >>> eq3 = x + y + 3*z + 4
    >>> eq4 = x - z
    >>> ie, d = _fourier_motzkin([eq1, eq2, eq3, eq4], symbols)
    >>> ie
    [3*x/2 + 13/10, 7*x/5 + 2/5]
    >>> assert set(d) == set([y, z])
    >>> d[y]
    (Min(2*x/3 + z/3 + 1/3, x + 2*z - 2) > y, y > -x - 3*z - 4)
    >>> d[z]
    (x > z, z > Max(-x/2 - 13/10, -2*x/5 - 2/5))
    """
    pivot = _find_pivot(inequalities, symbols)
    res = {}
    while pivot != None:
        mins, maxs, extra = _split_min_max(inequalities, pivot)
        res[pivot] = (mins > pivot, pivot > maxs)
        inequalities = _merge_mins_maxs(mins, maxs, symbols) + extra
        pivot = _find_pivot(inequalities, symbols)
    return inequalities, res


def _pick_var(inequalities, symbols):
    """Return a free variable of the system of inequalities

    Examples
    ========

    >>> from sympy.solvers.inequalities import _pick_var
    >>> from sympy.abc import x, y, z
    >>> symbols = {x, y, z}
    >>> eq1 = 2*x - 3*y + z + 1
    >>> eq2 = x - y + 2*z - 2
    >>> eq3 = x + y + 3*z + 4
    >>> eq4 = x - z
    >>> inequalities = [eq1, eq2, eq3, eq4]
    >>> _pick_var(inequalities, symbols)
    x
    """
    for eq in inequalities:  # should already be in canonical order
        symbols = symbols.intersection(eq.free_symbols)
        if symbols:
            return next((ordered(symbols)))


def _fourier_motzkin_extension(inequalities, symbols):
    """Extension of the Fourier-Motzkin algorithm to the case where
    inequalities do not contain variables that have at least two
    coefficients with opposite sign.

    Examples
    ========

    >>> from sympy.solvers.inequalities import _fourier_motzkin_extension
    >>> from sympy.abc import x, y, z
    >>> symbols = {x, y, z}
    >>> eq1 = 2*x - 3*y + z + 1
    >>> eq2 = x - y + 2*z - 2
    >>> eq3 = x - y + 3*z + 4
    >>> eq4 = x - z
    >>> d = _fourier_motzkin_extension([eq1, eq2, eq3, eq4], symbols)
    >>> assert set(d) == {x}
    >>> d[x]
    (oo > x, x > Max(z, y - 3*z - 4, y - 2*z + 2, 3*y/2 - z/2 - 1/2))
    """
    res = {}
    pivot = _pick_var(inequalities, symbols)
    while pivot and inequalities:
        mins, maxs, extra = _split_min_max(inequalities, pivot)
        with evaluate(False):
            res[pivot] = (mins > pivot, pivot > maxs)
        inequalities = extra
        pivot = _pick_var(inequalities, symbols)
    return res


def repsort(*replace):
    """Return sorted replacement tuples `(o, n)` such that `(o_i, n_i)` will appear before
    `(o_j, n_j)` if `o_j` appears in `n_i`. An error will be raised if o_j appears in n_i and o_i
    appears in n_k if k >= i.

    Examples
    ========

    >>> from sympy.abc import x, y, z, a
    >>> repsort((x, y + 1), (z, x + 2))
    [(z, x + 2), (x, y + 1)]
    >>> repsort((x, y + 1), (z, x**2))
    [(z, x**2), (x, y + 1)]
    >>> repsort(*Tuple((x, y + z), (y, a), (z, 1/y)))
    [(x, y + z), (z, 1/y), (y, a)]

    Any two of the following 3 tuples will not raise an error,
    but together they contain a cycle that raises an error:

    >>> repsort((x, y), (y, z), (z, x))
    Traceback (most recent call last):
    ...
    raise ValueError("cycle detected")
    """
    from itertools import permutations
    from sympy import default_sort_key, topological_sort
    free = {i for i, _ in replace}
    defs, replace = sift(replace,
                         lambda x: x[1].is_number or not x[1].has_free(*free),
                         binary=True)
    edges = [(i, j) for i, j in permutations(replace, 2) if
             i[1].has(j[0]) and (not j[0].is_Symbol or j[0] in i[1].free_symbols)]
    rv = topological_sort([replace, edges], default_sort_key)
    rv.extend(ordered(defs))
    return rv


def solve_linear_inequalities(inequalities, symbols):
    """Solve a system of linear inequalities returning the range
    of each variable as a Tuple of two relationals in a Dict.

    Parameters
    ==========

    inequalities: list of equations
        The system of inequalities to solve. All expressions are
        tested to be linear in the given symbols. Expressions that
        are relational must be strictly so (XXX necessary?); those
        that are passed as Expr are assumed to be positive.

    Examples
    ========

    >>> from sympy.solvers.inequalities import solve_linear_inequalities
    >>> from sympy.abc import x, y, z
    >>> eq1 = 2*x - 3*y + z + 1  # assumed to be positive
    >>> eq2 = x + 2*z > y + 2
    >>> eq3 = x + y + 3*z + 4 > 0
    >>> eq4 = z < x
    >>> symbols = {x, y, z}
    >>> d = solve_linear_inequalities([eq1, eq2, eq3, eq4], symbols)
    >>> keys = list(d)
    >>> assert keys == list(ordered(symbols)) == [x, y, z]
    >>> d[x]
    (oo > x, x > -2/7)
    >>> d[y]
    (Min(x + 1/3, 3*x - 2) > y, y > -4*x - 4)
    >>> d[z]
    (x > z, z > Max(-2*x + 3*y - 1, -x/2 + y/2 + 1, -x/3 - y/3 - 4/3))

    To find a valid point that satisfies the system, pick successive values.
    The value of ``x`` must be greater than -2/7, so let's pick 1. This
    determines the value of ``y``:

    >>> reps = {x: 1}
    >>> d[y].subs(reps)
    (1 > y, y > -8)

    So -1 is a valid value for ``y``. Those values of ``x`` and ``y`` give
    a range for ``z``:

    >>> reps[y] = -1
    >>> d[z].subs(reps)
    (1 > z, z > 0)

    So the point `(x, y, z) = (1, -1, 1/2) satisfies the equations since these
    values satisfy all the ranges for the variables:

    >>> reps[z] = S(1)/2
    >>> set(flatten(d.subs(reps).values()))
    {True}

    Thus, they make the equations themselves either True (or positive, if given as
    an expression):

    >>>> [i.xreplace(reps) for i in (eq1, eq2, eq3, eq4)]
    [13/2, True, True, True]

    See Also
    ========
    linear_programming - provides a solution to a linear problem of optimizing an objective under given constraints.

    """
    eqs = []
    symbols = list(symbols)
    symset = set(symbols)
    poseq = list(ordered(_positive_exprs(inequalities, symset)))
    eqs, res1 = _fourier_motzkin(poseq, symset)
    res2 = _fourier_motzkin_extension(eqs, symset)
    it = {k: Interval(v[1].rhs, v[0].lhs) for k, v in {**res1, **res2}.items()}.items()
    return Dict(*reversed(repsort(*it)))


if __name__ == "__main__":
    eq1 = 2 * x - 3 * y + z + 1
    eq2 = x - y + 2 * z - 2
    eq3 = x + y + 3 * z + 4 > 0
    eq4 = z < x
    loi = (eq1, eq2, eq3, eq4)
    ys = (x, y, z)
    sol = solve_linear_inequalities(loi, set(ys))
    print(sol)
    from time import time

    t = time()
    u1, u2, x1, x2 = symbols('u1 u2 x1 x2')
    ys = y1, y2, y3, y4, y5 = symbols('y1 y2 y3 y4 y5')
    loi = [50 * u2 - y5, -x2 + y5, 0, -u2 + 1, -35 * u2 + y4 + y5, 35 * u2 + x1 + x2 - y4 - y5 - 35, 35 * u2 - y4 - y5,
           -35 * u2 - x1 - x2 + y4 + y5 + 35, 50 * y1 - y4, 50 * u1 - x1 - 50 * y1 + y4, -50 * y1 + y4,
           -50 * u1 + x1 + 50 * y1 - y4, u2 - y1, -u1 - u2 + y1 + 1, 50 * u2 - y5, -50 * u2 - x2 + y5 + 50, 65 * y1,
           65 * u1 - 65 * y1, 35 * u2 + 65 * y1 - y4 - y5]
    print('solving %s inequalities...' % len(loi))
    print(solve_linear_inequalities(loi, set(ys)))
    print('time to find solution:', round(time() - t, 1))
