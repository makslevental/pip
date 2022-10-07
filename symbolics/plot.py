from sympy import var
from spb import *  # SymPy Plotting Module

var("x, y")

c = 3
# a boolean expression composed of multiple inequalities
expr = (y < x + c) & (y < -x + c) & (y > x - c) & (y > -x - c)

# assuming y is on the LHS of the inequality, here we extract
# the RHS, which are going to create the limiting lines
expressions = []
for a in expr.args:
    rhs = a.args[1]
    # append to expression the tuple (RHS, label)
    expressions.append((rhs, str(a)))

# plot the limiting lines
p1 = plot(*expressions, (x, -5, 5), aspect="equal", line_kw={"linestyle": "--"})
# plot the region represented by the overall expression
p2 = plot_implicit(expr, (x, -5, 5), (y, -5, 5))
# combine the plots
(p1 + p2).show()
