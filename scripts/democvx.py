import cvxpy as cp
import numpy
import matplotlib.pyplot as plt
import time

# n = 15
# m = 10
# # numpy.random.seed(1)
# A = numpy.random.rand(n, m)
# b = numpy.random.rand(n)
# # gamma must be nonnegative due to DCP rules.
# gamma = cp.Parameter(nonneg=True)
#
# x = cp.Variable(m, integer=True)
# error = cp.sum(A @ x - gamma)
# obj = cp.Maximize(error)
# problem = cp.Problem(obj)
#
# assert problem.is_dcp(dpp=True)
#
# gamma_vals = numpy.logspace(-4, 1)
# times = []
# new_problem_times = []
# for val in gamma_vals:
#     gamma.value = val
#     start = time.time()
#     problem.solve()
#     print("Status: ", problem.status)
#     print("The optimal value is", problem.value)
#     print("A solution x is")
#     print(x.value)
#
#     end = time.time()
#     times.append(end - start)
#     new_problem = cp.Problem(obj)
#     start = time.time()
#     new_problem.solve()
#     end = time.time()
#     new_problem_times.append(end - start)
#
# plt.rc('text')
# plt.rc('font', family='serif')
# plt.figure(figsize=(6, 6))
# plt.plot(gamma_vals, times, label='Re-solving a DPP problem')
# plt.plot(gamma_vals, new_problem_times, label='Solving a new problem')
# plt.xlabel(r'$\gamma$', fontsize=16)
# plt.ylabel(r'time (s)', fontsize=16)
# plt.legend()
# plt.show()

import cvxpy as cp
import numpy as np

# Generate a random non-trivial linear program.
m = 15
n = 10
s0 = np.random.randn(m)
lamb0 = np.maximum(-s0, 0)
s0 = np.maximum(s0, 0)
x0 = np.random.randn(n)
A = np.random.randn(m, n)
b = A @ x0 + s0
c = -A.T @ lamb0
gamma = cp.Parameter(shape=b.shape, nonneg=True)



# Define and solve the CVXPY problem.
x = cp.Variable(n, integer=True)
constraints = [A @ x <= gamma]
obj = cp.Minimize(c.T @ x)
prob = cp.Problem(obj, constraints)
assert prob.is_dcp(dpp=True)
for i in range(40):
    gamma.value = np.abs(b) * np.random.rand()
    prob.solve()

    # Print result.
    print("\nThe optimal value is", prob.value)
    print("A solution x is")
    print(x.value)
    print("A dual solution is")
    print(prob.constraints[0].dual_value)


