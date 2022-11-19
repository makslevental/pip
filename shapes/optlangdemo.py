# pip install optlang

import numpy as np
from optlang import Model, Variable, Constraint, Objective
from optlang.duality import convert_linear_problem_to_dual

rnd_cst = lambda: np.random.randint(1, 10)

x1 = Variable("x1", lb=0)
x2 = Variable("x2", lb=0)
x3 = Variable("x3", lb=0)

c1 = Constraint(x1 + x2 + x3, ub=100, name="c1")
c2 = Constraint(rnd_cst() * x1 + rnd_cst() * x2 + rnd_cst() * x3, ub=600, name="c2")
c3 = Constraint(rnd_cst() * x1 + rnd_cst() * x2 + rnd_cst() * x3, ub=300, name="c3")

obj = Objective(rnd_cst() * x1 + rnd_cst() * x2 + rnd_cst() * x3, direction="max")

model = Model(name="Primal model")
model.objective = obj
model.add([c1, c2, c3])

status = model.optimize()
print(model.to_lp())

print("status:", model.status)
print("objective value:", model.objective.value)
for var_name, var in model.variables.iteritems():
    print(var_name, "=", var.primal)

# the shadow price associated with a resource tells you how much more profit you would get by
# increasing the amount of that resource by one unit
print("shadow prices:")
for var_name, var in model.shadow_prices.items():
    print(var_name, "=", var)

print("\n----------\n")

dual_model = convert_linear_problem_to_dual(model, prefix="")
dual_model.name = "Dual model"
status = dual_model.optimize()
print(dual_model.to_lp())

print("status:", dual_model.status)
print("objective value:", dual_model.objective.value)
for var_name, var in dual_model.variables.iteritems():
    print(var_name, "=", var.primal)

print("shadow prices:")
for var_name, var in dual_model.shadow_prices.items():
    print(var_name, "=", var)
