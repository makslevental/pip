from z3 import Int, solve

x = Int('x')
y = Int('y')
print(solve(x > 2, y < 10, x + 2*y == 7))