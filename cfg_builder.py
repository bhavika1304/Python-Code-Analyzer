# cfg_builder.py
import ast
import networkx as nx

class CFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.graph = nx.DiGraph()
        self.counter = 0
        self.current = None

    def _new_node(self, label):
        """Create unique node id with label"""
        self.counter += 1
        node_id = f"n{self.counter}"
        self.graph.add_node(node_id, label=label)
        return node_id

    def build(self, func_node: ast.FunctionDef):
        """Entry point: build CFG for a function"""
        start = self._new_node("START")
        self.current = start
        for stmt in func_node.body:
            self.current = self._handle_stmt(stmt, self.current)
        end = self._new_node("END")
        self.graph.add_edge(self.current, end)
        return self.graph

    def _handle_stmt(self, stmt, prev):
        """Handle one statement and return last node id"""

        if isinstance(stmt, ast.If):
            cond_id = self._new_node(f"If {ast.unparse(stmt.test)}")
            self.graph.add_edge(prev, cond_id)

            # then branch
            then_prev = cond_id
            for b in stmt.body:
                then_prev = self._handle_stmt(b, then_prev)

            # else branch
            else_prev = cond_id
            for b in stmt.orelse:
                else_prev = self._handle_stmt(b, else_prev)

            # join node
            join_id = self._new_node("JOIN")
            self.graph.add_edge(then_prev, join_id)
            self.graph.add_edge(else_prev, join_id)
            return join_id

        elif isinstance(stmt, (ast.For, ast.While)):
            loop_id = self._new_node(f"Loop {ast.unparse(stmt)}")
            self.graph.add_edge(prev, loop_id)

            body_prev = loop_id
            for b in stmt.body:
                body_prev = self._handle_stmt(b, body_prev)
            self.graph.add_edge(body_prev, loop_id)  # back edge

            exit_id = self._new_node("LoopExit")
            self.graph.add_edge(loop_id, exit_id)   # exit edge
            return exit_id

        elif isinstance(stmt, ast.Return):
            ret_id = self._new_node(f"Return {ast.unparse(stmt.value)}")
            self.graph.add_edge(prev, ret_id)
            return ret_id

        else:
            stmt_id = self._new_node(ast.unparse(stmt))
            self.graph.add_edge(prev, stmt_id)
            return stmt_id
