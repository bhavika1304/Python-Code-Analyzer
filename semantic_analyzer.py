# semantic_analyzer.py
import ast

class SemanticAnalyzer(ast.NodeVisitor):
    def __init__(self, func_node: ast.FunctionDef):
        self.func_node = func_node
        self.loops = []           # store loop types/ranges
        self.recursive = False    # does function call itself?
        self.calls = []           # external calls
        self.ops = []             # operations inside loops
        self.complexity = "O(1)"  # default complexity
        self.intent = []          # inferred purpose

        self.visit(func_node)
        self._infer_intent()
        self._estimate_complexity()

    # --- AST Visitors ---
    def visit_For(self, node):
        loop_expr = ast.unparse(node.iter) if hasattr(ast, "unparse") else "for-loop"
        self.loops.append(loop_expr)
        self._record_ops(node.body)
        self.generic_visit(node)

    def visit_While(self, node):
        self.loops.append("while")
        self._record_ops(node.body)
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            fname = node.func.id
            self.calls.append(fname)
            if fname == self.func_node.name:
                self.recursive = True
        self.generic_visit(node)

    # --- Helpers ---
    def _record_ops(self, body):
        """Check body of loop for +=, *= operations etc."""
        for stmt in body:
            if isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.op, ast.Add):
                    self.ops.append("+=")
                elif isinstance(stmt.op, ast.Mult):
                    self.ops.append("*=")

    def _infer_intent(self):
        # Simple intent heuristics
        src = ast.unparse(self.func_node) if hasattr(ast, "unparse") else ""
        fname = self.func_node.name.lower()

        if "*=" in self.ops or "factorial" in fname:
            self.intent.append("factorial-like calculation")
        elif "+=" in self.ops or "sum" in fname:
            self.intent.append("summation/accumulation")
        elif "sort" in fname:
            self.intent.append("sorting routine")
        elif "search" in fname:
            if self.recursive or "while" in self.loops:
                self.intent.append("binary search")
            else:
                self.intent.append("linear search")
        elif self.recursive and "n-1" in src:
            self.intent.append("recursive decrement algorithm")
        elif self.recursive and "/2" in src:
            self.intent.append("divide-and-conquer algorithm")

        if not self.intent:
            self.intent.append("general computation")

    def _estimate_complexity(self):
        depth = len(self.loops)

        # Loops
        if depth == 1:
            self.complexity = "O(n)"
        elif depth == 2:
            self.complexity = "O(n^2)"
        elif depth > 2:
            self.complexity = f"O(n^{depth})"

        # Recursion overrides
        if self.recursive:
            src = ast.unparse(self.func_node) if hasattr(ast, "unparse") else ""
            if "n-1" in src:
                self.complexity = "O(n)"
            elif "n//2" in src or "n/2" in src:
                self.complexity = "O(log n)"
            elif any(op in src for op in ["split", "merge"]):
                self.complexity = "O(n log n)"

    # --- Public output ---
    def explanation(self):
        fname = self.func_node.name
        parts = [f"Function `{fname}()` analysis:"]

        if self.intent:
            parts.append(f"Purpose: {', '.join(self.intent)}.")

        if self.loops:
            parts.append(f"It contains {len(self.loops)} loop(s): {', '.join(self.loops)}.")
        if self.recursive:
            parts.append("It uses recursion.")
        if self.calls:
            parts.append(f"It calls other functions: {', '.join(self.calls)}.")

        parts.append(f"Estimated time complexity: **{self.complexity}**.")
        return " ".join(parts)


def analyze_function_semantics(func_node: ast.FunctionDef) -> str:
    analyzer = SemanticAnalyzer(func_node)
    return analyzer.explanation()
