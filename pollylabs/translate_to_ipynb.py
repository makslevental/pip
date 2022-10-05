import glob
import json
import os

import nbformat as nbf

for nbjson_fp in glob.glob(
    "/Users/mlevental/dev_projects/pip/pollylabs/playground.pollylabs.org.har.d/playground.pollylabs.org/examples/*.json"
):
    nb_name = os.path.split(os.path.splitext(nbjson_fp)[0])[1]
    print(nb_name)

    nbjson = json.load(open(nbjson_fp))

    nb = nbf.v4.new_notebook()
    nb["cells"] = []
    for _cell_idx, (cell_body, _cell_output, type) in sorted(
        nbjson.items(), key=lambda idx_cell: int(idx_cell[0])
    ):
        if cell_body == "":
            continue
        if type == "markdown":
            nb["cells"].append(nbf.v4.new_markdown_cell(cell_body))
        elif type in {"python", "solution", "error"}:
            nb["cells"].append(nbf.v4.new_code_cell(cell_body))
        else:
            raise Exception(f"wtfbbq {type}")

    fname = f"{nb_name}.ipynb"

    with open(fname, "w") as f:
        nbf.write(nb, f)
