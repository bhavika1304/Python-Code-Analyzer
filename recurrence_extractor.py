# recurrence_extractor.py
import ast

class RecurrenceExtractor(ast.NodeVisitor):
    def __init__(self, func_name):
        self.func_name = func_name
        self.recursive_calls = []
        self.assignments = {}   # track variable assignments
        self.recurrence = None
        self.complexity = "O(1)"  # default
        self.debug_explanation = ""  # NEW: explain why classification was chosen

    def visit_Assign(self, node):
        # Track simple assignments like mid = (l+r)//2
        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            self.assignments[name] = node.value
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check if the function is calling itself
        if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
            self.recursive_calls.append(node.args)
        self.generic_visit(node)

    def _is_halving_expr(self, expr):
        """Detect if an expression represents halving, e.g. (l+r)//2"""
        return (
            isinstance(expr, ast.BinOp)
            and isinstance(expr.op, ast.FloorDiv)
            and isinstance(expr.right, ast.Constant)
            and expr.right.value == 2
        )

    def analyze(self, func_node: ast.FunctionDef):
        self.visit(func_node)

        if not self.recursive_calls:
            self.recurrence = "Not recursive"
            self.complexity = "O(1)"
            self.debug_explanation = "No recursive calls detected."
            return self

        factorial_like = False
        fibonacci_like = False
        divide_and_conquer = False
        binary_search_like = False

        for args in self.recursive_calls:
            for a in args:
                # factorial: T(n-1)
                if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Sub):
                    if isinstance(a.right, ast.Constant) and a.right.value == 1:
                        # Check if it's mid-1 (binary search) or n-1 (factorial)
                        if isinstance(a.left, ast.Name) and a.left.id == "mid":
                            mid_expr = self.assignments.get("mid")
                            if mid_expr and self._is_halving_expr(mid_expr):
                                binary_search_like = True
                                self.debug_explanation = "Detected mid=(l+r)//2 and recursive call with mid±1 → halves input size."
                        else:
                            factorial_like = True
                            self.debug_explanation = "Detected recursion with n-1 argument."

                # fibonacci: T(n-2)
                if isinstance(a, ast.BinOp) and isinstance(a.op, ast.Sub):
                    if isinstance(a.right, ast.Constant) and a.right.value == 2:
                        fibonacci_like = True
                        self.debug_explanation = "Detected recursion with n-1 and n-2 arguments."

                # divide & conquer: T(n/2)
                if isinstance(a, ast.BinOp) and isinstance(a.op, (ast.FloorDiv, ast.Div)):
                    if isinstance(a.right, ast.Constant) and a.right.value == 2:
                        divide_and_conquer = True
                        self.debug_explanation = "Detected recursion with n/2 argument."

        # Decide classification
        if factorial_like and not fibonacci_like:
            self.recurrence = "T(n) = T(n-1) + O(1)"
            self.complexity = "O(n)"
        elif factorial_like and fibonacci_like:
            self.recurrence = "T(n) = T(n-1) + T(n-2) + O(1)"
            self.complexity = "O(2^n)"
        elif binary_search_like:
            self.recurrence = "T(n) = T(n/2) + O(1)"
            self.complexity = "O(log n)"
        elif divide_and_conquer:
            if len(self.recursive_calls) > 1:
                self.recurrence = "T(n) = 2T(n/2) + O(n)"
                self.complexity = "O(n log n)"
            else:
                self.recurrence = "T(n) = T(n/2) + O(1)"
                self.complexity = "O(log n)"
        elif len(self.recursive_calls) == 2:
            # heuristic for merge sort
            self.recurrence = "T(n) = 2T(n/2) + O(n)"
            self.complexity = "O(n log n)"
            self.debug_explanation = "Two recursive calls detected (L and R slices)."
        else:
            self.recurrence = "T(n) = f(T(smaller n))"
            self.complexity = "Unknown"
            self.debug_explanation = "Could not classify recurrence pattern."

        return self
