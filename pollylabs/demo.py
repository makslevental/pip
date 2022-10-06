import islpy as isl
from islplot_support import get_set_list, get_umap_list, plot_usets, plot_umaps

uset = isl.UnionSet("{T[i,j] : 0 < i <= j < 10; S[i,j] : 0 < 2 * j  <  i <= 8}")

umap = isl.UnionMap("{X[i,j] -> Y[i+3, 2 * j + 1]: 0 < i, j < 5}")

usets = get_set_list(uset)
plot_usets(usets)

umaps = get_umap_list(umap)
plot_umaps(umaps)

uset = isl.UnionSet("{T[i,j] : 0 < i <= j < 10; S[i,0] : 0 < i <= 20}")
usets = get_set_list(uset)
plot_usets(usets)