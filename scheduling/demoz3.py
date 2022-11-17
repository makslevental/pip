from z3 import Int, solve

s = [Int(f's_{i}') for i in range(6)]
print(s)


# to honor target

cycle_time_constraints = [
    s[2] - s[5] <= -1,
    s[1] - s[5] <= -1,
]

print(dependence_constraints)

# solve(x > 2, y < 10, x + 2*y == 7)