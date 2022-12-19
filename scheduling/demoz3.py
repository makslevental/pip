from z3 import (
    Int,
    solve,
    Function,
    IntSort,
    BoolSort,
    Goal,
    Tactic,
    Solver,
    RealSort,
    Real,
)
import z3

x = Real("x")
y = Real("y")
z = Real("z")
u = Real("u")
v = Real("v")
w = Real("w")

P = Function("P", RealSort(), BoolSort())

cons = [
    x <= y + 2 * z,
    x >= y - z,
    x >= y - 3 - 3 * z,
    x >= 5,
    x <= u,
    x >= v,
    # P(u),
    # P(v),
    P(x),
]

print(cons)
g = Goal()
g.add(*cons)

g2 = Tactic("fm")(g)[0]
print(g2)
