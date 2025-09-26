# metrics_calculator.py
import ast
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze

def calculate_metrics(code: str):
    metrics = {}
    tree = ast.parse(code)

    # --- Function-level metrics ---
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_code = ast.get_source_segment(code, node)

            # Cyclomatic Complexity
            cc = cc_visit(func_code)
            cyclo = cc[0].complexity if cc else 1

            # Nesting depth
            nesting = _max_depth(node)

            # Lines of code for function
            raw_metrics = analyze(func_code)
            sloc = raw_metrics.sloc

            metrics[node.name] = {
                "cyclomatic_complexity": cyclo,
                "nesting_depth": nesting,
                "SLOC": sloc
            }

    # --- Overall metrics ---
    raw = analyze(code)
    mi_score = mi_visit(code, True)

    metrics["_overall"] = {
        "maintainability_index": mi_score,
        "loc": raw.loc,
        "sloc": raw.sloc,
        "comments": raw.comments,
        "multi": raw.multi,
        "blank": raw.blank
    }

    return metrics


def _max_depth(node, current=0):
    """Helper: Calculate max nesting depth in AST."""
    if not list(ast.iter_child_nodes(node)):
        return current
    return max((_max_depth(child, current + 1) for child in ast.iter_child_nodes(node)), default=current)
