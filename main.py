# main.py
import ast
import json
import streamlit as st
import pandas as pd
from typing import Dict, Any, Set

from semantic_summarizer import SemanticSummarizer, set_parents
from semantic_analyzer import SemanticAnalyzer
from recurrence_extractor import RecurrenceExtractor
from cfg_builder import CFGBuilder
from dataflow_analyzer import DataFlowAnalyzer


st.set_page_config(page_title="Code Analyzer + Flowchart", layout="wide")
st.title("🐍 Python Code Analyzer")

# ---------- Top instructions ----------
st.markdown(
    """
**How to use**

1. Upload a Python file OR paste your code in the sidebar.
2. Click **Process Code** to analyze.
3. Explore results in the tabs: **Metrics**, **Explanations**, **Flowcharts**, **Raw JSON**.
4. You can refer to the **Metrics Guide** tab to understand the metrics analysed and how they affect your program.
"""
)

# ---------- Try imports ----------
has_metrics_calc = False
calc_exc = None
try:
    from metrics_calculator import calculate_metrics
    has_metrics_calc = True
except Exception as e:
    calc_exc = e
    has_metrics_calc = False

try:
    from utils import explain_metrics
except Exception as e:
    st.error("Cannot import utils.explain_metrics: " + str(e))
    raise

try:
    from flowchart_generator import simple_dot_for_function
    has_flowchart_module = True
except Exception:
    has_flowchart_module = False

# ---------- Sidebar: input & process button ----------
st.sidebar.header("Input")
uploaded_file = st.sidebar.file_uploader("Upload a Python file (.py)", type=["py"])
code_area = st.sidebar.text_area("Or paste Python code here", height=300)
process_button = st.sidebar.button("▶ Process Code")

# store code in session so user can edit without losing it
if "last_code" not in st.session_state:
    st.session_state["last_code"] = ""

if uploaded_file is not None:
    try:
        raw = uploaded_file.read()
        # sometimes it's bytes, sometimes string-like
        code_text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
    except Exception:
        uploaded_file.seek(0)
        code_text = uploaded_file.read().decode("utf-8") if isinstance(uploaded_file.read(), bytes) else uploaded_file.read()
    st.session_state["last_code"] = code_text
elif code_area and code_area.strip():
    st.session_state["last_code"] = code_area

code = st.session_state.get("last_code", "").strip()

# helper: count SLOC in range
def _count_sloc_from_segment(start_lineno, end_lineno, source_lines):
    start = start_lineno - 1
    end = end_lineno if end_lineno is not None else len(source_lines)
    seg = source_lines[start:end]
    return sum(1 for ln in seg if ln.strip() and not ln.strip().startswith("#"))

# fallback flowchart generator (DOT)
def _fallback_simple_dot_for_function(func_node: ast.FunctionDef, name: str = "flow"):
    parts = []
    parts.append(f'digraph "{name}"' + " {")
    parts.append("  node [shape=box];")
    parts.append('  start [label="START"];')
    prev = "start"
    idx = 0
    for stmt in func_node.body:
        try:
            label = ast.unparse(stmt).splitlines()[0][:70]
        except Exception:
            label = stmt.__class__.__name__
        label = label.replace('"', '\\"').replace("\n", "\\l")
        parts.append(f'  n{idx} [label="{label}"];')
        parts.append(f'  {prev} -> n{idx};')
        if isinstance(stmt, ast.If):
            parts.append(f'  n{idx} -> n{idx}_then [label="True"];')
            parts.append(f'  n{idx} -> n{idx}_else [label="False"];')
            parts.append(f'  n{idx}_then [label="THEN"];')
            parts.append(f'  n{idx}_else [label="ELSE"];')
            parts.append(f'  n{idx}_then -> n{idx}_join;')
            parts.append(f'  n{idx}_else -> n{idx}_join;')
            parts.append(f'  n{idx}_join [label="JOIN"];')
            prev = f'n{idx}_join'
        else:
            prev = f'n{idx}'
        idx += 1
    if idx == 0:
        parts.append('  start -> end;')
    else:
        parts.append(f'  {prev} -> end;')
    parts.append('  end [label="END"];')
    parts.append("}")
    return "\n".join(parts)

# AST-based behavior summarizer for a single function node
def summarize_function_behavior(func_node: ast.FunctionDef) -> str:
    features = []
    calls: Set[str] = set()
    has_return = False
    loops = 0
    ifs = 0
    comprehensions = 0
    is_recursive = False

    for node in ast.walk(func_node):
        if isinstance(node, (ast.For, ast.While)):
            loops += 1
        if isinstance(node, ast.If):
            ifs += 1
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            comprehensions += 1
        if isinstance(node, ast.Return):
            has_return = True
        if isinstance(node, ast.Call):
            # get simple function name where possible
            try:
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
            except Exception:
                pass
    # recursion check
    for c in calls:
        if c == func_node.name:
            is_recursive = True
            break

    if loops:
        features.append(f"contains {loops} loop{'s' if loops>1 else ''}")
    if ifs:
        features.append(f"uses {ifs} conditional block{'s' if ifs>1 else ''}")
    if comprehensions:
        features.append(f"uses {comprehensions} comprehension{'s' if comprehensions>1 else ''}")
    if has_return:
        features.append("returns values")
    if calls:
        example_calls = ", ".join(sorted(list(calls))[:5])
        features.append(f"calls other functions ({example_calls})")
    if is_recursive:
        features.append("is recursive (calls itself)")

    if not features:
        return "No notable control structures detected (likely simple linear code)."
    return "This function " + "; ".join(features) + "."

# Data containers used after user clicks Process Code
functions_metrics: Dict[str, Dict[str, Any]] = {}
overall_metrics: Dict[str, Any] = {}
raw_metrics_obj = None

# Remember if user has clicked process at least once
if process_button:
    st.session_state["run_analysis"] = True

if "run_analysis" not in st.session_state:
    st.info("Press **Process Code** in the sidebar to run analysis on the provided code.")
    st.stop()

if not code:
    st.error("No code provided. Paste code in the sidebar or upload a .py file.")
    st.stop()

source_lines = code.splitlines()

with st.spinner("Analyzing code..."):
    # Primary attempt: use calculate_metrics (radon-based)
    used_radon = False
    if has_metrics_calc:
        try:
            metrics = calculate_metrics(code)
            overall_metrics = metrics.get("_overall", {})
            functions_metrics = {k: v for k, v in metrics.items() if k != "_overall"}
            # normalize: ensure keys exist and naming consistent
            for fname, v in functions_metrics.items():
                # radon 'nesting' key name might be 'nesting_depth' or 'nesting'
                if "nesting" in v and "nesting_depth" not in v:
                    v["nesting_depth"] = v.pop("nesting")
            used_radon = True
        except Exception as e:
            st.warning("Radon-based metrics failed; falling back to AST analyzer. Error: " + str(e))
            used_radon = False

    # Fallback: use ast_parser.analyze_code (visitor)
    if not used_radon:
        try:
            from ast_parser import analyze_code  # your CodeAnalyzer visitor
            base = analyze_code(code)
            # base maps func_name -> {"cyclomatic_complexity":..., "max_nesting_depth":...}
            functions_metrics = {}
            for fname, data in base.items():
                functions_metrics[fname] = {
                    "cyclomatic_complexity": data.get("cyclomatic_complexity", 1),
                    "nesting_depth": data.get("max_nesting_depth", data.get("nesting_depth", 0)),
                    "SLOC": None,
                }
            overall_metrics = {}
            # compute SLOC using AST positions
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name in functions_metrics:
                    start = node.lineno
                    end = getattr(node, "end_lineno", None)
                    functions_metrics[node.name]["SLOC"] = _count_sloc_from_segment(start, end, source_lines)
        except Exception as e:
            st.error("Failed to analyze code with both radon and AST analyzer: " + str(e))
            st.stop()

# Prepare DataFrame
if functions_metrics:
    df = pd.DataFrame.from_dict(functions_metrics, orient="index")
    # normalize column names ordering
    col_order = []
    for col in ["cyclomatic_complexity", "nesting_depth", "SLOC", "start_line", "end_line"]:
        if col in df.columns:
            col_order.append(col)
    other_cols = [c for c in df.columns if c not in col_order]
    if col_order:
        df = df[col_order + other_cols]
else:
    df = pd.DataFrame()

# ✅ Save results after df is ready
st.session_state["functions_metrics"] = functions_metrics
st.session_state["overall_metrics"] = overall_metrics
st.session_state["df"] = df

functions_metrics = st.session_state.get("functions_metrics", {})
overall_metrics = st.session_state.get("overall_metrics", {})
df = st.session_state.get("df", pd.DataFrame())

# ---------- Tabs ----------
tab_metrics, tab_explain, tab_flow, tab_raw, tab_guide = st.tabs(
    ["📊 Metrics", "📝 Explanations", "🔗 Flowcharts", "📦 Raw JSON", "📖 Metrics Guide"]
)

# ----- Metrics Tab -----
with tab_metrics:
    st.header("Function metrics")
    if df.empty:
        st.info("No functions detected in code.")
    else:
        # style: highlight cyclomatic complexity
        def _highlight_cc(val):
            try:
                if val > 10:
                    return "background-color: #ff9999"  # red-ish
                if val > 5:
                    return "background-color: #ffe599"  # amber
            except Exception:
                pass
            return ""

        styled = df.style.map(lambda v: _highlight_cc(v) if isinstance(v, (int, float)) else "", subset=["cyclomatic_complexity"]) \
                        .format(precision=0, na_rep="-")
        st.dataframe(styled, width="stretch")

    # Overall metrics
    if overall_metrics:
        st.subheader("Overall metrics")
        try:
            overall_display = {
                "Maintainability Index": overall_metrics.get("maintainability_index"),
                "LOC": overall_metrics.get("loc"),
                "SLOC": overall_metrics.get("sloc"),
                "Comments": overall_metrics.get("comments"),
                "Multi-line Comments": overall_metrics.get("multi"),
                "Blank": overall_metrics.get("blank"),
            }
            st.json(overall_display)
        except Exception:
            st.write(overall_metrics)

# ----- Explanations Tab -----
with tab_explain:
    st.header("Natural-language explanations")
    if not functions_metrics:
        st.info("No function explanations available.")
    else:
        prepared = {fname: {
            "cyclomatic_complexity": v.get("cyclomatic_complexity"),
            "nesting_depth": v.get("nesting_depth"),
            "SLOC": v.get("SLOC")
        } for fname, v in functions_metrics.items()}

        try:
            metric_texts = explain_metrics(prepared)
        except Exception:
            metric_texts = {k: "" for k in prepared}

        tree = ast.parse(code)
        set_parents(tree)
        func_nodes = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

        for fname in sorted(prepared.keys()):
            st.markdown(f"### {fname}()")

            # Metrics
            if metric_texts.get(fname):
                st.markdown(f"**Metrics summary:** {metric_texts[fname]}")

            if fname in func_nodes:
                fn_node = func_nodes[fname]

                # Behavior
                st.markdown(f"**Behavior summary:** {summarize_function_behavior(fn_node)}")

                # Semantic Summarizer
                summarizer = SemanticSummarizer()
                sem_summary = summarizer.summarize(fn_node)
                st.markdown(f"**Semantic summary:** {sem_summary}")

                # Semantic Analyzer (intent + complexity)
                try:
                    analyzer = SemanticAnalyzer(fn_node)
                    if analyzer.intent:
                        st.markdown("**Deeper semantic intent:** " + "; ".join(analyzer.intent))

                    # Recurrence check
                    rec_complexity = None
                    try:
                        rec = RecurrenceExtractor(fname)
                        rec.visit(fn_node)
                        if getattr(rec, "recurrence", None):
                            rec_complexity = rec.complexity
                            st.markdown(f"**Recurrence Relation:** {rec.recurrence} → **{rec.complexity}**")
                    except Exception:
                        pass

                    # Choose semantic complexity (fallback if no recurrence)
                    complexity_est = rec_complexity if rec_complexity else analyzer.complexity
                    st.markdown(f"**Estimated Time Complexity (semantic):** {complexity_est}")

                except Exception as e:
                    st.warning(f"SemanticAnalyzer failed for {fname}: {e}")

                # --- Dataflow Analysis for Time + Space ---
                try:
                    df_result = DataFlowAnalyzer().analyze(fn_node)
                    if "complexity" in df_result and df_result["complexity"]:
                        st.markdown(f"**Estimated Time Complexity (dataflow):** {df_result['complexity']}")
                    if "space_complexity" in df_result and df_result["space_complexity"]:
                        st.markdown(f"**Estimated Space Complexity:** {df_result['space_complexity']}")
                    # ✅ Show per-loop reasoning if available
                    if "detailed_loops" in df_result and df_result["detailed_loops"]:
                        with st.expander("Show dataflow loop analysis details"):
                            for loop in df_result["detailed_loops"]:
                                st.write(
                                    f"- **{loop.get('desc', 'loop')}** → {loop.get('inferred_kind') or loop.get('kind')}")
                                if loop.get("reason"):
                                     st.caption(f"Reason: {loop['reason']}")
                except Exception as e:
                    st.warning(f"DataFlowAnalyzer failed for {fname}: {e}")

                # Always show disclaimer
                st.caption("⚠️ Complexity estimates are approximate and may not fully capture all edge cases or optimizations.")

            # Source preview
            with st.expander("Show function source"):
                node = func_nodes.get(fname)
                if node:
                    seg_start, seg_end = node.lineno - 1, getattr(node, "end_lineno", None)
                    src = "\n".join(source_lines[seg_start:seg_end])
                    st.code(src, language="python")

# ----- Flowcharts Tab -----
with tab_flow:
    st.header("Function flowcharts")
    tree = ast.parse(code)
    func_nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not func_nodes:
        st.info("No functions found for flowchart generation.")
    else:
        # display up to 3 columns
        ncols = min(3, max(1, len(func_nodes)))
        cols = st.columns(ncols)
        # Collect DOT sources
        dot_sources = {}
        for idx, fn in enumerate(func_nodes):
            col = cols[idx % ncols]
            with col:
                st.markdown(f"**{fn.name}**")
                if has_flowchart_module:
                    try:
                        dot_src = simple_dot_for_function(fn, name=fn.name)
                    except Exception as e:
                        st.warning(f"flowchart_generator failed for {fn.name}: {e}")
                        dot_src = _fallback_simple_dot_for_function(fn, name=fn.name)
                else:
                    dot_src = _fallback_simple_dot_for_function(fn, name=fn.name)
                st.graphviz_chart(dot_src)
                dot_sources[fn.name] = dot_src
        # --- Download all flowcharts as ZIP ---
        if dot_sources:
            import tempfile, zipfile, os
            from graphviz import Source
            import io

            # Generate ZIP only once per run, cache in session
            if "flowcharts_zip" not in st.session_state:
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "flowcharts.zip")

                    with zipfile.ZipFile(zip_path, "w") as zipf:
                        for fname, dot_src in dot_sources.items():
                            try:
                                graph = Source(dot_src)
                                out_path = os.path.join(tmpdir, f"{fname}.png")
                                graph.render(out_path, format="png", cleanup=True)
                                zipf.write(out_path + ".png", arcname=f"{fname}.png")
                            except Exception:
                                dot_path = os.path.join(tmpdir, f"{fname}.dot")
                                with open(dot_path, "w") as f:
                                    f.write(dot_src)
                                zipf.write(dot_path, arcname=f"{fname}.dot")

                    with open(zip_path, "rb") as f:
                        st.session_state["flowcharts_zip"] = f.read()

            # Download button uses cached zip
            st.download_button(
                "⬇️ Download All Flowcharts (ZIP)",
                st.session_state["flowcharts_zip"],
                file_name="flowcharts.zip",
                mime="application/zip",
                key="download_all_flowcharts"
            )

# ----- Raw JSON Tab -----
with tab_raw:
    st.header("Raw metrics JSON")
    full_report = {"functions": functions_metrics, "overall": overall_metrics}
    st.json(full_report)

    st.markdown("**Downloads**")
    try:
        # Save once in session
        if "metrics_csv" not in st.session_state and not df.empty:
            st.session_state["metrics_csv"] = df.to_csv().encode("utf-8")
        if "metrics_json" not in st.session_state:
            st.session_state["metrics_json"] = json.dumps(full_report, indent=2)

        if "metrics_csv" in st.session_state:
            st.download_button(
                "Download metrics CSV",
                st.session_state["metrics_csv"],
                file_name="function_metrics.csv",
                mime="text/csv",
                key="metrics_csv_dl"
            )

        if "metrics_json" in st.session_state:
            st.download_button(
                "Download full JSON",
                st.session_state["metrics_json"],
                file_name="metrics_full.json",
                mime="application/json",
                key="metrics_json_dl"
            )
    except Exception as e:
        st.error("Failed to prepare downloads: " + str(e))

# ----- Metrics Guide Tab -----
with tab_guide:
    st.header("Metrics Guide for Beginners")

    st.markdown("""
    ### Cyclomatic Complexity (CC)
    - **What it is:** A measure of how many independent execution paths exist through your function.
    - **How it's calculated:** Every `if`, `for`, `while`, `try/except`, or logical operator (`and`, `or`) adds to complexity.
    - **Why it matters:** 
      - CC ≤ 5 → Easy to understand and test.  
      - CC 6–10 → Moderate, needs careful testing.  
      - CC > 10 → Complex, may be hard to maintain or prone to bugs.  
    - **Tip:** High CC functions may benefit from splitting into smaller helpers.

    ### Nesting Depth
    - **What it is:** The deepest level of nested blocks (loops, conditionals, try/except).
    - **Why it matters:** Deeply nested code is harder to follow and debug.
      - Depth ≤ 2 → Good.  
      - Depth 3–4 → Manageable.  
      - Depth > 4 → Refactor recommended.  
    - **Tip:** Flattening logic with early returns or helper functions improves readability.

    ### Source Lines of Code (SLOC)
    - **What it is:** Count of actual executable code lines (ignores comments and blanks).
    - **Why it matters:** Longer functions are often harder to maintain and test.  
      - Short (≤ 20 lines) → Easy to scan.  
      - Medium (21–50 lines) → Acceptable.  
      - Long (> 50 lines) → Consider refactoring.  

    ### Maintainability Index (MI)
    - **What it is:** A composite score (0–100) based on SLOC, Cyclomatic Complexity, and Halstead Volume (an information-theory metric).
    - **Why it matters:** Indicates how easy the code is to maintain.  
      - MI ≥ 85 → Highly maintainable.  
      - MI 65–85 → Moderate maintainability.  
      - MI < 65 → Hard to maintain.  

    ### LOC (Lines of Code), Comments, Blanks
    - **LOC:** Total lines (including comments and blanks).
    - **Comments:** Count of comment lines (more comments = better documentation).
    - **Blanks:** Empty lines for readability and logical separation.

    ---

    ### Semantic Summaries
    - **What it is:** An interpretation of what your code *does*, based on its structure.
    - **Why it matters:** Helps you (or others) understand the *intent* of a function, not just its metrics.  
    - **Tip:** Semantic summaries highlight key patterns (searching, sorting, updating) to make functions easier to describe and review.

    ### Recurrence Relations
    - **What it is:** A mathematical equation describing recursive calls (e.g., `T(n) = 2T(n/2) + O(n)`).
    - **Why it matters:** Helps predict runtime growth for recursive functions.
    - **Tip:** Mastering common recurrences (binary search, mergesort, quicksort) helps reason about algorithm efficiency.

    ### Deeper Semantic Intent
    - **What it is:** A higher-level description of the algorithm or purpose your function implements (e.g., *binary search*, *sorting*, *factorial*).  
    - **Why it matters:** Moves beyond syntax to highlight the *real intent* of the code.  
    - **Tip:** Helps when comparing code against known algorithm patterns.

    ### Estimated Time Complexity (semantic)
    - **What it is:** An algorithmic complexity estimate inferred from semantic analysis and recurrence relations.  
    - **Why it matters:** Captures high-level algorithm classes (O(n), O(log n), O(n log n)).  
    - **Tip:** Often accurate for recursive divide-and-conquer algorithms.

    ### Estimated Time Complexity (dataflow)
    - **What it is:** A complexity estimate derived by analyzing loops, updates, and variable dependencies.  
    - **Why it matters:** Detects cost of iterative patterns (linear scans, nested loops).  
    - **Tip:** Dataflow-based estimates are precise for loops but may miss divide-and-conquer patterns.

    ### Estimated Space Complexity
    - **What it is:** An estimate of additional memory used by recursion depth or data structures.  
    - **Why it matters:** Helps predict memory usage alongside runtime.  
      - O(1): Constant space (no recursion, no large structures).  
      - O(n): Uses recursion or builds linear-size structures (lists, sets).  
      - O(n²): Builds quadratic-size structures (matrices).  
    - **Tip:** Always check both time and space to understand tradeoffs.
    ---

    ⚠️ **Reminder:**  
    These metrics are **approximations**. Real performance depends on many factors (data distribution, language/runtime optimizations, hardware). Use them as *guidelines*, not absolute truths.

    💡 **Tip for beginners:**  
    A function with high metrics isn’t always “bad” — but it’s a sign to check readability, test coverage, and maintainability. Refactoring (smaller functions, fewer nested blocks) often improves code health.
    """)

# ---------- End ----------
st.markdown("---")
st.markdown("Built with ❤️ — Coding made easy for you.")
