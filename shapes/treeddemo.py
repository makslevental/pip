from treed import TreeD

treed = TreeD(
    probpath="model.mps",
    nodelimit=2000,
    showcuts=True
)

treed.solve()
fig = treed.draw()
fig.show(renderer='notebook')