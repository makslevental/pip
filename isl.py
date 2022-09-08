import islpy as isl
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as pt
import numpy as np


# ENDEXAMPLE


def plot_basic_set(bset, *args, **kwargs):
    # This is a total hack. But it works for what it needs to do. :)

    plot_vert = kwargs.pop("plot_vert", False)

    vertices = []
    bset.compute_vertices().foreach_vertex(vertices.append)

    vertex_pts = []

    for v in vertices:
        points = []
        myset = isl.BasicSet.from_multi_aff(v.get_expr())
        myset.foreach_point(points.append)
        (point,) = points
        vertex_pts.append(
            [
                point.get_coordinate_val(isl.dim_type.set, i).to_python()
                for i in range(2)
            ]
        )

    vertex_pts = np.array(vertex_pts)

    center = np.average(vertex_pts, axis=0)

    from math import atan2

    vertex_pts = np.array(
        sorted(vertex_pts, key=lambda x: atan2(x[1] - center[1], x[0] - center[0]))
    )

    if plot_vert:
        pt.plot(vertex_pts[:, 0], vertex_pts[:, 1], "o")

    Path = mpath.Path  # noqa

    codes = [Path.LINETO] * len(vertex_pts)
    codes[0] = Path.MOVETO

    pathdata = [(code, tuple(coord)) for code, coord in zip(codes, vertex_pts)]
    pathdata.append((Path.CLOSEPOLY, (0, 0)))

    codes, verts = zip(*pathdata)
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, **kwargs)
    pt.gca().add_patch(patch)


def basic_set():
    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=["x", "y"])

    bset = (
        isl.BasicSet.universe(space)
        .add_constraint(isl.Constraint.ineq_from_names(space, {1: -1, "x": 1}))
        .add_constraint(isl.Constraint.ineq_from_names(space, {1: 5, "x": -1}))
        .add_constraint(isl.Constraint.ineq_from_names(space, {1: -1, "y": 1}))
        .add_constraint(isl.Constraint.ineq_from_names(space, {1: 5, "y": -1}))
    )
    print("set 1 %s:" % bset)

    bset2 = isl.BasicSet("{[x, y] : x >= 0 and x < 5 and y >= 0 and y < x+4 }")
    print("set 2: %s" % bset2)

    bsets_in_union = []
    bset.union(bset2).convex_hull().foreach_basic_set(bsets_in_union.append)
    print(bsets_in_union)
    (union,) = bsets_in_union
    print("union: %s" % union)

    plot_basic_set(bset, facecolor="red", edgecolor="black", alpha=0.3)
    plot_basic_set(bset2, facecolor="green", edgecolor="black", alpha=0.2)
    pt.grid()
    pt.xlim([-1, 6])
    pt.ylim([-1, 8])
    # pt.show()
    pt.savefig("before-union.png", dpi=50)

    plot_basic_set(
        union, facecolor="blue", edgecolor="yellow", alpha=0.5, plot_vert=True
    )
    pt.savefig("after-union.png", dpi=50)


def isl_ast_codegen(S):  # noqa: N803
    b = isl.AstBuild.from_context(isl.Set("{:}"))
    m = isl.Map.from_domain_and_range(S, S)
    m = isl.Map.identity(m.get_space())
    m = isl.Map.from_domain(S)
    ast = b.ast_from_schedule(m)
    p = isl.Printer.to_str(isl.DEFAULT_CONTEXT)
    p = p.set_output_format(isl.format.C)
    p.flush()
    p = p.print_ast_node(ast)
    return p.get_str()


def lexmin():
    # space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT, set=["a", "b", "c"])
    #
    # bset = (
    #     isl.BasicSet.universe(space)
    #     # Create a constraint `const + coeff_1 * var_1 +...>= 0`.
    #     .add_constraint(isl.Constraint.ineq_from_names(space, {1: -1, "a": 1}))
    #     .add_constraint(isl.Constraint.ineq_from_names(space, {1: 5, "a": -1}))
    #
    #     .add_constraint(isl.Constraint.ineq_from_names(space, {1: -2, "b": 1}))
    #     .add_constraint(isl.Constraint.ineq_from_names(space, {1: 4, "b": -1}))
    #
    #     .add_constraint(isl.Constraint.ineq_from_names(space, {1: -2, "c": 1}))
    #     .add_constraint(isl.Constraint.ineq_from_names(space, {1: 7, "c": -1}))
    #     .add_constraint(isl.Constraint.eq_from_names())
    #
    #
    # )
    # print("set 1 %s:" % bset)

    # s = isl.Set(
    #     """{ [s] : exists a,b,c :
    #     0 <= a <= 5 and 1 <= b <= 4 and 2 <= c <= 7 and
    #     ((2 <= b and b <= 3) implies (a <= 1 or a >= 3)) and
    #     ((not (c < 5 or b > 3)) implies (a > 2 and c < 3)) and s = a + b + c }
    #     """
    # )
    # print(s.lexmin())

    # s = isl.Set(
    #     """{ [z] : exists x0,x1,x2,t :
    #     0 <= x0 and 0 <= x1 and 0 <= x2 and
    #
    #     1*x0 + 2*x1 + (-3*x2) <= 5 and
    #     2*x0 + 1*x1 + (-4*x2) <= 7 and
    #     z = 2*t*x0 + x1 + x2 }
    #     """
    # )
    # print(s.lexmin())
    # s = isl.Set("[n,m] -> { [i,j] : 0 <= i <= n and i <= j <= m }")
    # print(isl_ast_codegen(s))
    # s = isl.Set("""[z,t] -> { [x0,x1,x2] :
    # 0 <= x0 and 0 <= x1 and 0 <= x2
    # and 1*x0 + 2*x1 + (-3*x2) <= 5
    # and 2*x0 + 1*x1 + (-4*x2) <= 7
    # and z = -3 + t*x0
    #
    # }""")
    #
    # print(
    #     s.lexmin()
    # )
    print(isl.PwQPolynomial("""[n, m] -> {[i, j] -> i * m + j :
                0 <= i < n and 0 <= j < m}""").bound(isl.fold.max))


lexmin()
