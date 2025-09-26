# semantic_summarizer.py
import ast

class SemanticSummarizer(ast.NodeVisitor):
    def __init__(self):
        self.init_statements = []
        self.loops = []
        self.updates = []
        self.branches = []
        self.returns = []

    def visit_Assign(self, node):
        try:
            target = ast.unparse(node.targets[0])
            value = ast.unparse(node.value)
            # if inside loop/branch → treat as update
            parent = getattr(node, "parent", None)
            if isinstance(parent, (ast.For, ast.While, ast.If)):
                self.updates.append(f"{target} = {value}")
            else:
                self.init_statements.append(f"{target} = {value}")
        except Exception:
            pass
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        try:
            target = ast.unparse(node.target)
            op = type(node.op).__name__
            value = ast.unparse(node.value)
            self.updates.append(f"{target} {op}= {value}")
        except Exception:
            pass
        self.generic_visit(node)

    def visit_For(self, node):
        try:
            target = ast.unparse(node.target)
            it = ast.unparse(node.iter)
            self.loops.append(f"for {target} in {it}")
        except Exception:
            pass
        self.generic_visit(node)

    def visit_While(self, node):
        try:
            cond = ast.unparse(node.test)
            self.loops.append(f"while {cond}")
        except Exception:
            pass
        self.generic_visit(node)

    def visit_If(self, node):
        try:
            cond = ast.unparse(node.test)
            self.branches.append(f"if {cond}")
        except Exception:
            pass
        self.generic_visit(node)

    def visit_Return(self, node):
        try:
            val = ast.unparse(node.value) if node.value else "None"
            self.returns.append(val)
        except Exception:
            pass
        self.generic_visit(node)

    def summarize(self, func_node: ast.FunctionDef):
        """Generate program-specific explanation for a function node."""
        self.visit(func_node)

        explanation = []

        # Purpose (roughly from name)
        fname = func_node.name.lower()
        if "fact" in fname:
            explanation.append("This function computes the factorial of n.")
        elif "fib" in fname:
            explanation.append("This function computes Fibonacci numbers.")
        elif "search" in fname:
            explanation.append("This function performs a search in a sequence.")
        elif "sort" in fname:
            explanation.append("This function sorts a sequence.")
        else:
            explanation.append("This function performs a computation.")

        # Process
        if self.init_statements:
            explanation.append("It initializes variables: " + ", ".join(self.init_statements) + ".")
        if self.loops:
            for loop in self.loops:
                explanation.append(f"It iterates with {loop}.")
        if self.updates:
            for upd in self.updates:
                explanation.append(f"It updates variables with `{upd}`.")
        if self.branches:
            for br in self.branches:
                explanation.append(f"It branches on condition `{br}`.")
        if self.returns:
            explanation.append(f"Finally, it returns {self.returns[-1]}.")

        return " ".join(explanation)

def set_parents(node, parent=None):
    """Recursively attach a .parent attribute to each AST node"""
    for child in ast.iter_child_nodes(node):
        child.parent = node
        set_parents(child, node)
