#!/usr/bin/env python
# coding: utf-8

# Import LibPoly and utitlities

from sys import stdout
import polypy
import itertools


# Table printout utilities


def output(item, width=15):
    if isinstance(item, tuple):
        stdout.write("({:.2f} {:.2f})".format(item[0], item[1]).center(width))
    elif isinstance(item, float):
        stdout.write("{:.2f}".format(item).center(width))
    else:
        stdout.write("{}".format(item).center(width))


# Compute and print the sign table
# - x: the main variable
# - polys: the polynomials to sign table
# - m: the model for other variables


def sign_table(x, polys, m):
    # Get the roots and print the header
    roots = set()  # Set of roots
    output("poly/int")
    for f in polys:
        output(f)
        f_roots = f.roots_isolate(m)
        roots.update(f_roots)
    stdout.write("\n")
    # Sort the roots and add infinities
    roots = [polypy.INFINITY_NEG] + sorted(roots) + [polypy.INFINITY_POS]
    # Print intervals and signs in the intervals
    root_i, root_j = itertools.tee(roots)
    next(root_j)
    for r1, r2 in zip(root_i, root_j):
        output((r1.to_double(), r2.to_double()))
        # The interval (r1, r2)
        v = r1.get_value_between(r2)
        m.set_value(x, v)
        for f in polys:
            output(f.sgn(m))
        stdout.write("\n")
        # The interval [r2]
        if r2 != polypy.INFINITY_POS:
            output(r2.to_double())
            m.set_value(x, r2)
            for f in polys:
                output(f.sgn(m))
            stdout.write("\n")
    m.unset_value(x)


# Test the sign table on $(x^2 - 2)$ and $(x^2 - 3)$.


x = polypy.Variable("x")


y = polypy.Variable("y")
polys_y = [x ** 2 + y ** 2 - 1, (x - 1) ** 2 + y ** 2 - 1]


polys_x = [x + 1, x + 0, 2 * x - 1, x - 1, x - 2]


m = polypy.Assignment()


sign_table(x, polys_x, m)


m.set_value(x, -1)


sign_table(y, polys_y, m)


# In[ ]:
