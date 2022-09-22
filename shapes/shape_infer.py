import warnings
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import List

import torch
from ncls import NCLS
from torch import TensorType
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr, wl
import sympy as sp
import numpy as np

torch._C._jit_set_symbolic_shapes_test_mode(True)
torch._C.Graph.set_global_print_source_ranges(False)

GREEKS = [
    r"\\alpha",
    r"\\beta",
    r"\\gamma",
    r"\\delta",
    r"\\epsilon",
    r"\\zeta",
    r"\\eta",
    r"\\theta",
    r"\\iota",
    r"\\kappa",
    r"\\lambda",
    r"\\mu",
    r"\\nu",
    r"\\xi",
    r"\\omicron",
    r"\\pi",
    r"\\rho",
    r"\\sigma",
    r"\\tau",
    r"\\upsilon",
    r"\\phi",
    r"\\chi",
    r"\\psi",
    r"\\omega",
]
GREEKS_TO_IDX = {g: i for i, g in enumerate(GREEKS)}
IDX_TO_GREEKS = {i: g for i, g in enumerate(GREEKS)}


def label_constants(graph):
    for n in graph.nodes():
        if n.kind() == "prim::Constant":
            n.output().setDebugName(n.output().debugName() + ".constant")


def remove_periods(graph):
    for inp in graph.inputs():
        try:
            inp.setDebugName(inp.debugName().replace(".", "_"))
        except:
            pass
    for inp in graph.outputs():
        try:
            inp.setDebugName(inp.debugName().replace(".", "_"))
        except:
            pass
    for n in graph.nodes():
        for inp in n.inputs():
            try:
                inp.setDebugName(inp.debugName().replace(".", "_"))
            except:
                pass
        for inp in n.outputs():
            try:
                inp.setDebugName(inp.debugName().replace(".", "_"))
            except:
                pass


def eval_backtrace_symbolic_outputs(op_shape_graph):
    def dfs(inp):
        val_name = inp.debugName()
        if inp.node().kind() == "prim::Constant":
            return inp.node().i("value")
        elif inp.node().kind() == "prim::Param":
            return val_name
        else:
            res = {}
            for next_inp in inp.node().inputs():
                res[next_inp.debugName()] = dfs(next_inp)

            if inp.node().kind() == "aten::__getitem__":
                obj, item = list(inp.node().inputs())
                obj_name, item_name = obj.debugName(), item.debugName()
                return f"{obj_name}[{res[item_name]}]"
            elif inp.node().kind() == "aten::sub":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} - {res[b_name]})"
            elif inp.node().kind() == "aten::add":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} + {res[b_name]})"
            elif inp.node().kind() == "aten::floordiv":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"Floor[{res[a_name]} / {res[b_name]}]"
            elif inp.node().kind() == "aten::div":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} / {res[b_name]})"
            elif inp.node().kind() == "aten::Int":
                a = inp.node().input()
                return f"{res[a.debugName()]}"
            elif inp.node().kind() == "prim::If":
                cond = dfs(inp.node().input())
                true_clause = dfs(list(list(inp.node().blocks())[0].outputs())[0])
                false_clause = dfs(list(list(inp.node().blocks())[1].outputs())[0])
                return f"If[{cond}, {true_clause}, {false_clause}]"
            elif inp.node().kind() == "aten::remainder":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"Mod[{res[a_name]}, {res[b_name]}]"
            elif inp.node().kind() == "aten::eq":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} == {res[b_name]})"
            elif inp.node().kind() == "aten::mul":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} * {res[b_name]})"
            elif inp.node().kind() == "prim::TupleConstruct":
                ress = {}
                for next_inp in inp.node().inputs():
                    ress[next_inp.debugName()] = dfs(next_inp)
                return ress
            else:
                warnings.warn(f"{inp.node().kind()} not implemented")

    res = {}
    for inp in op_shape_graph.return_node().inputs():
        res[inp.debugName().replace(".", "_")] = dfs(inp)

    if len(res) == 1 and isinstance(list(res.values())[0], dict):
        res = list(res.values())[0]
    return res


def eval_backtrace_symbolic_outputs_latex(op_shape_graph):
    def dfs(inp):
        val_name = inp.debugName()
        if inp.node().kind() == "prim::Constant":
            return inp.node().i("value")
        elif inp.node().kind() == "prim::Param":
            return val_name
        else:
            res = {}
            for next_inp in inp.node().inputs():
                res[next_inp.debugName()] = dfs(next_inp)

            if inp.node().kind() == "aten::__getitem__":
                obj, item = list(inp.node().inputs())
                obj_name, item_name = obj.debugName(), item.debugName()
                return f"{obj_name}[{res[item_name]}]"
            elif inp.node().kind() == "aten::sub":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"{res[a_name]} - {res[b_name]}"
            elif inp.node().kind() == "aten::add":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"{res[a_name]} + {res[b_name]}"
            elif inp.node().kind() == "aten::floordiv":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return rf"\Floor*{{\begin{{matrix}}\dfrac{{ {res[a_name]} }}{{ {res[b_name]} }} \end{{matrix}} }}"
            elif inp.node().kind() == "aten::div":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"\frac {{ {res[a_name]} }}{{ {res[b_name]} }}"
            elif inp.node().kind() == "aten::Int":
                a = inp.node().input()
                return f"{res[a.debugName()]}"
            elif inp.node().kind() == "prim::If":
                cond = dfs(inp.node().input())
                true_clause = dfs(list(list(inp.node().blocks())[0].outputs())[0])
                false_clause = dfs(list(list(inp.node().blocks())[1].outputs())[0])
                return f"If[{cond}, {true_clause}, {false_clause}]"
            elif inp.node().kind() == "aten::remainder":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"Mod[{res[a_name]}, {res[b_name]}]"
            elif inp.node().kind() == "aten::eq":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} == {res[b_name]})"
            elif inp.node().kind() == "aten::mul":
                a, b = list(inp.node().inputs())
                a_name, b_name = a.debugName(), b.debugName()
                return f"({res[a_name]} \times {res[b_name]})"
            else:
                warnings.warn(f"{inp.node().kind()} not implemented")

    res = {}
    for inp in op_shape_graph.return_node().inputs():
        res[inp.debugName()] = dfs(inp)

    return res


def get_shape_compute_graph(frozen_model_graph, inp_shapes: List[List[int]]):
    torch._C._jit_pass_remove_mutation(frozen_model_graph)
    torch._C._jit_pass_propagate_shapes_on_graph(frozen_model_graph)
    torch._C._jit_pass_peephole(frozen_model_graph)
    torch._C._jit_pass_constant_propagation(frozen_model_graph)

    mod_inputs = list(frozen_model_graph.inputs())
    assert len(mod_inputs) == len(inp_shapes)
    for i, mod_input in enumerate(mod_inputs):
        if inp_shapes[i] is not None:
            mod_input.setType(mod_input.type().with_sizes(inp_shapes[i]))

    torch._C._jit_pass_propagate_shapes_on_graph(frozen_model_graph)

    label_constants(frozen_model_graph)
    remove_periods(frozen_model_graph)

    shape_compute_graph = (
        torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute(
            frozen_model_graph
        )
    )
    partial_eval_shape_graph = shape_compute_graph.partial_eval_shape_graph()
    partial_eval_shape_graph.makeMultiOutputIntoTuple()
    # remove branches
    for n in partial_eval_shape_graph.findAllNodes("prim::RaiseException"):
        n.destroy()
    torch._C._jit_pass_dce(partial_eval_shape_graph)

    remove_periods(partial_eval_shape_graph)

    tensor_to_sizes = {}
    for node in frozen_model_graph.nodes():
        for outp in node.outputs():
            if not isinstance(outp.type(), TensorType):
                continue

            try:
                sym_sizes = [
                    f"SS({o})" if o < 0 else o for o in outp.type().symbolic_sizes()
                ]
                tensor_to_sizes[outp.debugName()] = sym_sizes
            except:
                pass
        # print(torch._C._jit_shape_compute_graph_for_node(node))

    return shape_compute_graph, partial_eval_shape_graph, tensor_to_sizes


def get_op_shape_compute_graphs(shape_compute_graph):
    op_shape_graphs = sorted(
        list(shape_compute_graph.partial_eval_shape_graph().items()),
        key=lambda x: x[0].output().debugName(),
    )
    for _nn, gg in op_shape_graphs:
        for n in gg.findAllNodes("prim::RaiseException"):
            n.destroy()
        torch._C._jit_pass_dce(gg)

    return op_shape_graphs


def make_shape_stuff(model, ssa_to_param_map, inp_shapes: List[List[int]]):
    frozen_model = torch.jit.freeze(torch.jit.script(model.eval()))
    (
        shape_compute_graph,
        partial_shape_eval_graph,
        tensor_ssa_to_sizes,
    ) = get_shape_compute_graph(frozen_model.graph, inp_shapes)
    # op_shape_graphs = get_op_shape_compute_graphs(shape_compute_graph)
    print(f"{shape_compute_graph=}")
    debugname_sym_mapping = {
        "%" + n.debugName(): f"SS({v})"
        for n, v in shape_compute_graph.graph_output_to_symbolic_shape_dim().items()
    }

    sym_shape_to_index = {}
    sym_shape_to_val_debug_name = {}
    for index, output in enumerate(
        list(partial_shape_eval_graph.outputs())[0].node().inputs()
    ):
        d = debugname_sym_mapping["%" + output.debugName()]
        sym_shape_to_index[d] = index
        sym_shape_to_val_debug_name[f"{d}"] = f"%{output.debugName()}"

    print("sym_shape_to_val_debug_name", pformat(sym_shape_to_val_debug_name))

    input_syms = list(frozen_model.graph.inputs())[1].type().symbolic_sizes()

    sym_to_greeks = {f"SS({v})": IDX_TO_GREEKS[i] for i, v in enumerate(input_syms)}
    ssa_to_greeks = {}
    for i, (sym, ssa) in enumerate(
        sym_shape_to_val_debug_name.items(), start=len(input_syms)
    ):
        sym_to_greeks[sym] = IDX_TO_GREEKS[i]
        ssa_to_greeks[ssa] = IDX_TO_GREEKS[i]

    sym_ssa_to_shape_inputs = eval_backtrace_symbolic_outputs(partial_shape_eval_graph)
    sym_ssa_to_shape_inputs = {
        "%" + k: translate(v, ssa_to_param_map)
        for k, v in sym_ssa_to_shape_inputs.items()
    }

    return (
        partial_shape_eval_graph,
        sym_ssa_to_shape_inputs,
        sym_shape_to_val_debug_name,
        ssa_to_greeks,
        tensor_ssa_to_sizes,
        frozen_model.graph,
    )


def eval_with_mathematica(s, WEval, assumptions):
    if assumptions is None:
        assumptions = []
    fs = f"FullSimplify[{s}, Assumptions -> {{{', '.join(assumptions)}}}]"
    return WEval(wl.Expand(wlexpr(fs)))


def simplify_with_mathematica(sym_to_shape_inputs, ss_map):
    simplified = {}
    with WolframLanguageSession(
        # kernel="/opt/Wolfram/WolframEngine/12.3/Executables/WolframKernel",
        # kernel_loglevel=logging.DEBUG,
        initfile=(Path(__file__).parent / "initkernel.m").resolve(),
    ) as session:
        assumptions = [f"Mod[{rep}, 32] == 0, {rep} > 0" for rep in ss_map.values()]
        for shape_sym, shape_formula in sym_to_shape_inputs.items():
            if isinstance(shape_formula, str) and len(shape_formula) > 0:
                math_res = eval_with_mathematica(
                    shape_formula, session.evaluate, assumptions
                )
                print(
                    shape_sym,
                    f"shape_formula: {shape_formula}\n simplified: {math_res}",
                )
                # simplified[translate(shape_sym, ssa_to_greeks)] = math_res
                simplified[shape_sym] = math_res

    return simplified


def translate(text, conversion_dict):
    if not text or not isinstance(text, str):
        return text
    # preliminary transformation:
    # before = before or str.lower
    t = text
    for key, value in conversion_dict.items():
        # t = re.sub(rf"{key}\b", value, t)
        t = t.replace(key, value)
        # t = re.sub(rf"{key}\b", value, t)
    return t


def get_tensor_live_ranges(graph):
    tensor_last_use = defaultdict(lambda: -1)
    tensor_topo_idx = {}
    node_topo_idx = {}
    idx_nodes = list(reversed(list(enumerate(graph.nodes(), start=1))))
    node_topo_idx[graph.return_node()] = idx_nodes[0][0] + 1

    for idx, node in idx_nodes:
        node_topo_idx[node] = idx
        for outp in node.outputs():
            if not isinstance(outp.type(), TensorType):
                continue
            tensor_topo_idx[outp] = idx
            for use in outp.uses():
                tensor_last_use[outp] = max(
                    tensor_last_use[outp], node_topo_idx[use.user]
                )

    # for inp in graph.inputs():
    #     if not isinstance(inp.type(), TensorType):
    #         continue
    #     tensor_topo_idx[inp] = 0
    #     for use in inp.uses():
    #         tensor_last_use[inp] = max(tensor_last_use[inp], node_topo_idx[use.user])

    live_ranges = {}
    for val, idx in tensor_topo_idx.items():
        live_ranges[val.debugName()] = (idx, tensor_last_use[val])

    return live_ranges


def get_constraints(tensor_ssa_to_sizes, shape_sym_to_formula, live_ranges):
    tensor_ssa_to_sympy_expr = {}
    coeffs = set()
    for tensor, sizes in tensor_ssa_to_sizes.items():
        sympy_expr = np.prod(
            [sp.sympify(shape_sym_to_formula.get(s, s), rational=True) for s in sizes]
        )
        if sympy_expr.free_symbols:
            coeffs.add(tuple(sympy_expr.free_symbols))
        tensor_ssa_to_sympy_expr[tensor] = sympy_expr

    param_to_coeff = {}
    for i, coeff in enumerate(coeffs):
        thetai = sp.Symbol(f"θ_{i}")
        param_to_coeff[thetai] = coeff
        for tensor, sympy_expr in tensor_ssa_to_sympy_expr.items():
            if sympy_expr.free_symbols:
                tensor_ssa_to_sympy_expr[tensor] = sympy_expr.subs(
                    np.prod(list(sympy_expr.free_symbols)), thetai
                )

    ids = {
        id: tensor
        for id, tensor in zip(np.arange(len(live_ranges)), live_ranges.keys())
    }
    starts = np.array([live_ranges[tensor][0] for tensor in ids.values()])
    ends = np.array([live_ranges[tensor][1] for tensor in ids.values()])
    live_range_ncls = NCLS(
        starts=starts,
        ends=ends + 1,
        ids=np.array(list(ids.keys())),
    )
    edge_list = np.array(
        live_range_ncls.all_overlaps_both(
            starts=starts,
            ends=ends + 1,
            indexes=np.array(list(ids.keys())),
        )
    )
    edge_list = edge_list.T.astype(order="C", dtype=edge_list.dtype)

    return tensor_ssa_to_sympy_expr, ids, edge_list