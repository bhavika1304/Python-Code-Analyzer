# dataflow_analyzer.py
import ast
from typing import List, Optional, Tuple, Any

# Simple helper to get readable source of a node (ast.unparse if available)
def _src(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return str(node)

class DataFlowAnalyzer(ast.NodeVisitor):
    """
    Enhanced dataflow analyzer that:
      - records assignments, loops, updates
      - detects recursion
      - heuristically infers time and space complexity using symbolic loop-bound analysis
    Heuristics cover:
      - for-range loops -> usually O(n)
      - while loops that shrink range multiplicatively (e.g., binary search) -> O(log n)
      - loops with additive updates -> O(n)
      - multiplicative growth/shrink (i *= 2 or i //= 2) -> O(log n) when used as loop bound
      - nested loops -> multiplicative composition (e.g., O(n^2))
      - structure allocations -> space = O(n)
      - recursion -> O(n) stack by default; if recursive argument is halved, we can hint O(log n)
    Notes:
      - This is static inference with best-effort heuristics; some constructs require symbolic solvers.
    """

    def __init__(self):
        # recorded information
        self.symbols = {}       # variable -> symbolic value (string)
        self.loop_info: List[str] = []   # brief descriptions of loops
        self.updates: List[str] = []     # variable updates recorded as strings
        self.recursive = False            # detect recursion
        self.structs: List[str] = []      # detect lists/dicts/sets created
        self.func_name: Optional[str] = None
        self.func_args: List[str] = []    # parameter names of current function

        # internal: store loop "objects" for more detailed analysis
        self._loops: List[dict] = []

    def analyze(self, func_node: ast.FunctionDef):
        """Analyze a function AST node and return analysis dict."""
        self.func_name = func_node.name
        self.func_args = [a.arg for a in getattr(func_node.args, "args", [])]
        self.visit(func_node)

        time_complexity = self._estimate_time()
        space_complexity = self._estimate_space()

        return {
            "symbols": self.symbols,
            "loops": self.loop_info,
            "updates": self.updates,
            "detailed_loops": self._loops,
            "complexity": time_complexity,
            "space_complexity": space_complexity,
        }

    # ---------- Visitors ----------
    def visit_Assign(self, node: ast.Assign):
        # Record simple assigns and detect container creation
        # Only handle simple single-target assignments for readability
        if len(node.targets) >= 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                value_src = _src(node.value)
                self.symbols[name] = value_src

                # detect container/list/dict/set creation or comprehensions
                if isinstance(node.value, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp)):
                    self.structs.append(type(node.value).__name__)

        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if isinstance(node.target, ast.Name):
            name = node.target.id
            op = type(node.op).__name__
            value = _src(node.value)
            self.updates.append(f"{name} {op}= {value}")
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        """Gather info for for-loops. We try to infer iteration kind for range-based loops."""
        loop_desc = ""
        loop_obj = {"type": "for", "desc": None, "kind": None, "target": None, "iter": None, "body_updates": []}

        # target name
        if isinstance(node.target, ast.Name):
            loop_obj["target"] = node.target.id

        # detect range(...) form
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
            args = [_src(a) for a in node.iter.args]
            loop_desc = f"for {loop_obj.get('target', '?')} in range({', '.join(args)})"
            loop_obj.update({"desc": loop_desc, "iter": "range", "kind": "unknown"})
            # heuristics for step/stop
            try:
                # if single argument -> range(n) or range(stop)
                if len(node.iter.args) == 1:
                    loop_obj["kind"] = "linear"  # O(n)
                elif len(node.iter.args) >= 2:
                    # range(start, stop, step)
                    step = node.iter.args[2] if len(node.iter.args) >= 3 else None
                    if step is None:
                        loop_obj["kind"] = "linear"
                    else:
                        # if step is a BinOp with multiplication by 2? still linear unless step depends on target
                        loop_obj["kind"] = "linear"
            except Exception:
                loop_obj["kind"] = "linear"
        else:
            # other iterables (lists, generator) — assume linear
            loop_desc = f"for {loop_obj.get('target', '?')} in {_src(node.iter)}"
            loop_obj.update({"desc": loop_desc, "iter": _src(node.iter), "kind": "linear"})

        self.loop_info.append(loop_desc or "for-loop")
        # inspect body to capture updates inside this loop
        body_updates = self._collect_updates_in_body(node.body)
        loop_obj["body_updates"] = body_updates
        # check if body creates containers
        loop_obj["allocates"] = any(self._creates_container_in_stmt(s) for s in node.body)
        self._loops.append(loop_obj)

        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        """Collect a while loop and perform deeper analysis on its condition and body."""
        cond_src = _src(node.test)
        loop_desc = f"while {cond_src}"
        self.loop_info.append(loop_desc)

        # analyze body updates
        body_updates = self._collect_updates_in_body(node.body)

        loop_obj = {
            "type": "while",
            "desc": loop_desc,
            "condition": cond_src,
            "body_updates": body_updates,
            "allocates": any(self._creates_container_in_stmt(s) for s in node.body),
            "inferred_kind": None,   # will be 'linear', 'log', 'constant', etc.
            "reason": None,
        }

        # attempt to infer complexity kind for this while loop
        kind, reason = self._infer_while_kind(node.test, node.body, body_updates)
        loop_obj["inferred_kind"] = kind
        loop_obj["reason"] = reason

        self._loops.append(loop_obj)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # detect recursion if the function calls itself directly by name
        if isinstance(node.func, ast.Name) and node.func.id == self.func_name:
            self.recursive = True

        # detect container-creating helper calls like list(), dict(), set(), or comprehension alternatives
        if isinstance(node.func, ast.Name) and node.func.id in ("list", "dict", "set"):
            self.structs.append(node.func.id)

        self.generic_visit(node)

    # ---------- Helpers for loop analysis ----------

    def _collect_updates_in_body(self, stmts: List[ast.stmt]) -> List[str]:
        """Collect assign/augassign statements in loop body and return readable strings."""
        updates = []
        for s in stmts:
            if isinstance(s, ast.Assign):
                if len(s.targets) >= 1 and isinstance(s.targets[0], ast.Name):
                    updates.append(f"{_src(s.targets[0])} = {_src(s.value)}")
            elif isinstance(s, ast.AugAssign):
                if isinstance(s.target, ast.Name):
                    updates.append(f"{_src(s.target)} {type(s.op).__name__}= {_src(s.value)}")
            # also check nested statements (ifs) for assignments
            if hasattr(s, "body") and isinstance(s.body, list):
                updates.extend(self._collect_updates_in_body(s.body))
            if hasattr(s, "orelse") and isinstance(s.orelse, list):
                updates.extend(self._collect_updates_in_body(s.orelse))
        return updates

    def _creates_container_in_stmt(self, stmt: ast.stmt) -> bool:
        """Return True if statement creates a container (list/dict/set) directly."""
        if isinstance(stmt, ast.Assign):
            v = stmt.value
            if isinstance(v, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.SetComp, ast.DictComp)):
                return True
            if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("list", "dict", "set"):
                return True
        # check nested blocks
        if hasattr(stmt, "body") and isinstance(stmt.body, list):
            for s in stmt.body:
                if self._creates_container_in_stmt(s):
                    return True
        return False

    def _is_var_halved_expr(self, expr: ast.AST, varname: str) -> bool:
        """
        Detect expressions that represent roughly 'var // 2' or '(l+r)//2' etc.
        We conservatively check for FloorDiv by constant >= 2 or RShift operations.
        """
        # var // 2 OR (something)//2
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.FloorDiv):
            right = expr.right
            if isinstance(right, ast.Constant) and isinstance(right.value, int) and right.value >= 2:
                return True
        # bitshift right: var >> 1
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.RShift):
            return True
        return False

    def _find_assignments_to_names(self, stmts: List[ast.stmt], names: List[str]) -> List[Tuple[str, str]]:
        """
        Return list of (target_name, rhs_src) assignments found in the stmts (searching nested bodies).
        """
        found = []
        for s in stmts:
            if isinstance(s, ast.Assign) and len(s.targets) >= 1 and isinstance(s.targets[0], ast.Name):
                t = s.targets[0].id
                if t in names:
                    found.append((t, _src(s.value)))
            if hasattr(s, "body") and isinstance(s.body, list):
                found.extend(self._find_assignments_to_names(s.body, names))
            if hasattr(s, "orelse") and isinstance(s.orelse, list):
                found.extend(self._find_assignments_to_names(s.orelse, names))
        return found

    def _infer_while_kind(self, test: ast.AST, body: List[ast.stmt], body_updates: List[str]) -> Tuple[str, Optional[str]]:
        """
        Heuristics to infer a while-loop iteration complexity class:
          - If body contains assignments that halve the gap between two bounds (mid pattern),
            and the condition is based on two bounds, infer 'log'.
          - If body contains multiplicative shrink/growth on the loop variable used in condition,
            infer 'log'.
          - If updates subtract/add a constant to the bound used in condition, infer 'linear'.
          - Otherwise fall back to 'unknown' (conservative -> linear).
        Returns (kind, reason)
        """
        # 1) If test is a Compare between two names (e.g., l <= r or i < n)
        if isinstance(test, ast.Compare):
            # left and right could be names or attributes
            left = test.left
            comparators = test.comparators
            if len(comparators) == 1 and isinstance(left, ast.Name) and isinstance(comparators[0], ast.Name):
                left_name = left.id
                right_name = comparators[0].id
                # check for mid computation and bound updates (binary search style)
                # look for mid = (l + r) // 2 or mid assignment; then l = mid + 1 or r = mid - 1
                # find any assignment to a 'mid' variable
                mid_assigns = [a for a in body_updates if ("mid =" in a or "mid=" in a)]
                if mid_assigns:
                    # find assignments to left/right bound names in body
                    bound_assigns = self._find_assignments_to_names(body, [left_name, right_name])
                    # look for left = mid + C or right = mid - C patterns
                    if any(("mid" in rhs and ("+" in rhs or "-" in rhs)) for (_, rhs) in bound_assigns):
                        return "O(log n)", "bounds halved each iteration (binary-search style)"
                # else check multiplicative shrink/growth on left/right in body
                mul_shrink = False
                for (t, rhs) in self._find_assignments_to_names(body, [left_name, right_name]):
                    # rhs like t // 2 or t >> 1 or t = t // 2
                    try:
                        parsed = ast.parse(rhs, mode="eval")
                        if self._is_var_halved_expr(parsed.body, t):
                            mul_shrink = True
                    except Exception:
                        # fallback string check
                        if "//2" in rhs or ">>" in rhs:
                            mul_shrink = True
                if mul_shrink:
                    return "O(log n)", "multiplicative shrink of bound in loop body"

                # additive updates (e.g., left = left + 1) -> linear
                add_updates = self._find_assignments_to_names(body, [left_name, right_name])
                if any(("+" in rhs and "mid" not in rhs) or ("-" in rhs and "mid" not in rhs) for (_, rhs) in add_updates):
                    return "O(n)", "additive updates to bound"

        # 2) If body contains multiplicative updates to the loop variable (i *= 2 or i = i * 2)
        for upd in body_updates:
            # examples: "i = i * 2", "i *= 2", "i = i // 2"
            if ("* 2" in upd) or ("*= 2" in upd) or ("// 2" in upd) or (">> 1" in upd) or ("//2" in upd):
                # multiplicative step suggests log behaviour if variable controls the loop
                return "O(log n)", "multiplicative update in loop body"

        # 3) If none of above, but we see any assignment in body that changes a loop-controlling variable by a constant -> linear
        if any(("+= " in u) or ("-= " in u) or ("+ " in u and "=" in u) or ("- " in u and "=" in u) for u in body_updates):
            return "O(n)", "constant-step updates in loop body"

        # Conservative fallback: unknown -> treat as linear
        return "O(n)", "default conservative assumption"

    # ---------- Complexity aggregation ----------
    def _estimate_time(self) -> str:
        """
        Aggregate per-loop kinds into a single function-level time complexity.
        Very simple composition:
          - multiply nested kinds (e.g., nested linear -> O(n^2))
          - if any loop is 'log' and another is 'linear' nested -> O(n log n)
          - if recursion exists, indicate 'recursive (heuristic)' and prefer recurrence extractor elsewhere
        """
        if self.recursive:
            # we don't attempt to fully solve recurrences here, leave it as recursive hint
            return "O(n) (recursive heuristic)"

        # if no loops -> O(1)
        if not self._loops:
            return "O(1)"

        # collect kinds in top-level loops only (we do not detect nesting depth robustly here,
        # but if multiple loops appear we conservatively multiply)
        kinds = []
        for loop in self._loops:
            kind = loop.get("inferred_kind") or loop.get("kind") or "unknown"
            kinds.append(kind)

        # map kinds to base symbols
        # treat unknown as linear (conservative)
        mapped = []
        for k in kinds:
            if k in ("O(1)", "constant", "constant-time"):
                mapped.append(("const", 0))
            elif k in ("O(log n)", "log"):
                mapped.append(("log", 1))
            elif k in ("O(n)", "linear", "unknown"):
                mapped.append(("n", 1))
            elif k == "linear":
                mapped.append(("n", 1))
            else:
                # if it's already formatted like "O(...)" just try to preserve
                if isinstance(k, str) and k.startswith("O("):
                    return k
                mapped.append(("n", 1))

        # simple multiplicative composition: multiply counts
        # count how many 'n' and how many 'log' factors
        n_count = sum(1 for t, _ in mapped if t == "n")
        log_count = sum(1 for t, _ in mapped if t == "log")

        if n_count == 0 and log_count == 0:
            return "O(1)"
        # compose into human-friendly complexity string
        parts = []
        if n_count > 0:
            parts.append("n" if n_count == 1 else f"n^{n_count}")
        if log_count > 0:
            parts.append("log n" if log_count == 1 else f"(log n)^{log_count}")
        return "O(" + " * ".join(parts) + ")"

    def _estimate_space(self) -> str:
        """
        Simple heuristics:
          - recursion -> O(n) (stack)
          - if any container creation detected -> O(n)
          - if loops allocate containers or comprehensions -> O(n)
          - else constant
        """
        if self.recursive:
            return "O(n)"  # recursion stack cost

        if self.structs:
            return "O(n)"

        # if any loop allocates containers conservatively treat as O(n)
        for loop in self._loops:
            if loop.get("allocates"):
                return "O(n)"

        return "O(1)"