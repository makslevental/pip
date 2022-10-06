import islpy as isl
from IPython.core.display import Math
from IPython.core.display_functions import display


def get_printer():
    printer = isl.Printer.to_str(isl.DEFAULT_CONTEXT)
    printer.set_output_format(isl.format.LATEX)
    return printer


def isl_set_get_latex(s):
    printer = get_printer()
    printer.print_set(s)
    retval = printer.get_str()
    return retval


def isl_map_get_latex(m):
    printer = get_printer()
    printer.print_map(m)
    retval = printer.get_str()
    return retval


def isl_union_map_get_latex(um):
    printer = get_printer()
    printer.print_union_map(um)
    retval = printer.get_str()
    return retval


def isl_union_set_get_latex(us):
    printer = get_printer()
    printer.print_union_set(us)
    retval = printer.get_str()
    return retval


def wrap_and_break_lines(string):
    if string.find("\\cup") != -1:
        string = string.replace("\\cup", "\\cup\\\\\\quad")
        string = string + "\\\\"
        string = "\\begin{array}{l}" + string + "\\end{array}"

    # string = r"\left[" + string + r"\right]"
    return string


def get_latex(latex_ip):
    if isinstance(latex_ip, isl.BasicMap):
        y = isl_map_get_latex(isl.Map(latex_ip))
    elif isinstance(latex_ip, isl.BasicSet):
        y = isl_set_get_latex(isl.Set(latex_ip))
    elif isinstance(latex_ip, isl.Set):
        y = isl_set_get_latex(latex_ip)
    elif isinstance(latex_ip, isl.Map):
        y = isl_map_get_latex(latex_ip)
    elif isinstance(latex_ip, isl.UnionMap):
        y = isl_union_map_get_latex(latex_ip)
    elif isinstance(latex_ip, isl.UnionSet):
        y = isl_union_set_get_latex(latex_ip)

    return wrap_and_break_lines(str(y))


def print_latex(latex_ip):
    print(get_latex(latex_ip))


def display_latex(latex_ip):
    display(Math(get_latex(latex_ip)))
