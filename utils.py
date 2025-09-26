# utils.py

def explain_metrics(metrics: dict) -> dict:
    """
    Generate human-readable explanations for each function's metrics.

    Args:
        metrics (dict): Dictionary of metrics per function:
            {
              "foo": {"cyclomatic_complexity": 5, "nesting_depth": 2, "SLOC": 15},
              ...
            }

    Returns:
        dict: {function_name: explanation_string}
    """
    explanations = {}

    for func, data in metrics.items():
        cc = data.get("cyclomatic_complexity", "?")
        nesting = data.get("nesting_depth", "?")
        sloc = data.get("SLOC", "?")

        # Heuristic interpretation
        if cc <= 5:
            cc_text = "low (easy to understand)"
        elif cc <= 10:
            cc_text = "moderate (some branching)"
        else:
            cc_text = "high (complex, consider refactoring)"

        if nesting <= 2:
            nesting_text = "shallow nesting"
        elif nesting <= 4:
            nesting_text = "moderate nesting"
        else:
            nesting_text = "deep nesting (harder to follow)"

        explanation = (
            f"Function `{func}()` has cyclomatic complexity **{cc}** ({cc_text}), "
            f"maximum nesting depth **{nesting}** ({nesting_text}), and about **{sloc}** lines of code. "
        )

        # Optional: add quick readability tip
        if cc > 12 or nesting > 8 or sloc > 20:
            explanation += "⚠️ This function may be hard to maintain."

        explanations[func] = explanation

    return explanations