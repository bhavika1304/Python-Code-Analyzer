# flowchart_generator.py
import ast

class FlowchartBuilder:
    def __init__(self):
        self.counter = 0
        self.lines = []

    def _new_id(self):
        self.counter += 1
        return f"n{self.counter}"

    def build_for_function(self, func_node: ast.FunctionDef, name="flow") -> str:
        self.lines = [f'digraph "{name}" {{', "  node [shape=box];"]
        start_id = "start"
        self.lines.append('  start [label="START", shape=ellipse, style=filled, fillcolor=lightgray];')
        prev = start_id

        for stmt in func_node.body:
            prev = self._handle_stmt(stmt, prev)

        self.lines.append(f"  {prev} -> end;")
        self.lines.append('  end [label="END", shape=ellipse, style=filled, fillcolor=lightgray];')
        self.lines.append("}")
        return "\n".join(self.lines)

    def _handle_stmt(self, stmt, prev):
        if isinstance(stmt, ast.If):
            cond_id = self._new_id()
            self.lines.append(f'  {cond_id} [label="Decision: {ast.unparse(stmt.test)}?", shape=diamond, style=filled, fillcolor=lightblue];')
            self.lines.append(f"  {prev} -> {cond_id};")

            # then branch
            then_prev = cond_id
            for b in stmt.body:
                then_prev = self._handle_stmt(b, then_prev)
            join_id = self._new_id()
            self.lines.append(f'  {then_prev} -> {join_id};')

            # else branch
            if stmt.orelse:
                else_prev = cond_id
                for b in stmt.orelse:
                    else_prev = self._handle_stmt(b, else_prev)
                self.lines.append(f'  {else_prev} -> {join_id};')
            else:
                self.lines.append(f'  {cond_id} -> {join_id} [label="False"];')

            return join_id

        elif isinstance(stmt, (ast.For, ast.While)):
            loop_id = self._new_id()
            text = ast.unparse(stmt).split(":")[0]
            self.lines.append(f'  {loop_id} [label="Loop: {text}", shape=parallelogram, style=filled, fillcolor=lightyellow];')
            self.lines.append(f"  {prev} -> {loop_id};")

            body_prev = loop_id
            for b in stmt.body:
                body_prev = self._handle_stmt(b, body_prev)
            self.lines.append(f"  {body_prev} -> {loop_id};")  # loop back

            exit_id = self._new_id()
            self.lines.append(f'  {loop_id} -> {exit_id} [label="exit"];')
            return exit_id

        elif isinstance(stmt, ast.Return):
            ret_id = self._new_id()
            self.lines.append(f'  {ret_id} [label="Return {ast.unparse(stmt.value)}", shape=ellipse, style=filled, fillcolor=lightgreen];')
            self.lines.append(f"  {prev} -> {ret_id};")
            return ret_id

        else:
            stmt_id = self._new_id()
            label = ast.unparse(stmt).replace('"', '\\"').splitlines()[0][:50]
            self.lines.append(f'  {stmt_id} [label="{label}", shape=box];')
            self.lines.append(f"  {prev} -> {stmt_id};")
            return stmt_id


def simple_dot_for_function(func_node: ast.FunctionDef, name="flow") -> str:
    builder = FlowchartBuilder()
    return builder.build_for_function(func_node, name)
