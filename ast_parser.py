import ast
from typing import List

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.function_metrics = {}

    def visit_FunctionDef(self, node):
        """Called when a function is defined"""
        self.current_function = node.name
        self.current_nesting = 0
        self.max_nesting = 0
        self.cyclomatic_complexity = 1  # start at 1 (base path)

        # Visit inside the function
        self.generic_visit(node)

        # Save results for this function
        self.function_metrics[self.current_function] = {
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "max_nesting_depth": self.max_nesting
        }

    def visit_If(self, node):
        self.cyclomatic_complexity += 1
        self._increase_nesting(node)

    def visit_For(self, node):
        self.cyclomatic_complexity += 1
        self._increase_nesting(node)

    def visit_While(self, node):
        self.cyclomatic_complexity += 1
        self._increase_nesting(node)

    def visit_Try(self, node):
        self.cyclomatic_complexity += 1
        self._increase_nesting(node)

    def visit_With(self, node):
        # with-statements also introduce blocks
        self._increase_nesting(node)

    def _increase_nesting(self, node):
        """Helper to track nesting depth"""
        self.current_nesting += 1
        self.max_nesting = max(self.max_nesting, self.current_nesting)
        self.generic_visit(node)
        self.current_nesting -= 1


def analyze_code(source_code: str):
    """Parse Python source code and return metrics dictionary"""
    tree = ast.parse(source_code)
    analyzer = CodeAnalyzer()
    analyzer.visit(tree)
    return analyzer.function_metrics
