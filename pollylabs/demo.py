import islpy as isl
from islplot.plotter import plot_set_points, plot_map, plot_map_as_groups
import matplotlib.pyplot as plt
from islplot_support import get_set_list, get_umap_list, plot_usets, plot_umaps, print_before_after
from islpy import Set

# prop_cycle = plt.rcParams["axes.prop_cycle"]
# colors = prop_cycle.by_key()["color"]
#
# uset = isl.UnionSet("{T[i,j] : 0 < i <= j < 10; S[i,j] : 0 < 2 * j  <  i <= 8}")
#
# umap = isl.UnionMap("{X[i,j] -> Y[i+3, 2 * j + 1]: 0 < i, j < 5}")
#
# usets = get_set_list(uset)
# plot_usets(usets)
#
# umaps = get_umap_list(umap)
# plot_umaps(umaps)
#
# uset = isl.UnionSet("{T[i,j] : 0 < i <= j < 10; S[i,0] : 0 < i <= 20}")
# usets = get_set_list(uset)
# plot_usets(usets)
#
# context, domain, schedule, reads, writes = parse_code

# plot_set_points(
#     Set("{S[x,y]: 0 <= x <= 8 and -7 <= y <= 5}"), marker=".", color="gray", size=3
# )
# import matplotlib as mpl
# # mpl.rcParams['grid'] = True
# print(mpl.rcParams.keys())
#
# plot_set_points(Set("{S[x,y]: 0 < x < 8 and 0 < y + x < 5}"), marker="o")
#
# plot_set_points(Set("{S[x,y]: 4 < x < 8 and 0 < y < 5}"), marker="s", color="red")
#
# plot_set_points(
#     Set("{S[x,y]: 0 < x and y > -6 and y + x < 0}"), marker="D", color="blue"
# )
umap = isl.UnionMap("{X[i,j] -> Y[i+3, 2 * j + 1]: 0 < i, j < 5}")
#
# # umap.foreach_map(lambda m: plot_map(m))
# # plot_map_as_groups(umap)
# plot_map(umap)
#
# plt.grid(True)
# plt.show()

domain = isl.UnionSet("{S[i,j] : 0 <= i,j < 1024 }")
original = isl.UnionMap("{S[i,j] -> [i,j]}")
transformation = isl.UnionMap("{[i,j] -> [floor(i/4), i % 4, floor(j/4), j % 4]}")

transformed = original.apply_range(transformation)
print_before_after(domain, original, transformed)
