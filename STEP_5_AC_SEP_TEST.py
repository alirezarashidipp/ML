# test_ac_split.py
import sys
import os
import difflib
import pytest

# Add your project path so imports work
gcp_path = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL CODE"
if gcp_path not in sys.path:
    sys.path.append(gcp_path)

# Import the function under test
from STEP_23_AC_SEPERATION import split_ac

# --- Coverage scenarios for AC header detection/split ---
@pytest.mark.parametrize("raw, expected_before, expected_after", [
    # 1) No AC at all
    ("This is a simple text.", "This is a simple text.", ""),

    # 2) 'ac' in the middle (standalone)
    ("this is a text that does have ac and it is something.",
     "this is a text that does have ",
     "ac and it is something."),

    # 3) 'acceptance criteria' phrase
    ("please see acceptance criteria: the steps are here",
     "please see ",
     "acceptance criteria: the steps are here"),

    # 4) 'ac' at start with colon
    ("ac: must follow these rules",
     "",
     "ac: must follow these rules"),

    # 5) multiple matches -> split at first match
    ("intro... acceptance criteria first... later ac: second",
     "intro... ",
     "acceptance criteria first... later ac: second"),

    # 6) uppercase AC with dash
    ("AC - items start here",
     "",
     "AC - items start here"),

    # 7) mixed case and extra spaces
    ("  Please consider   Acceptance   Criteria  - list below",
     "  Please consider   ",
     "Acceptance   Criteria  - list below"),

    # 8) newlines before AC
    ("intro line\nanother line\nacceptance criteria\n- do x",
     "intro line\nanother line\n",
     "acceptance criteria\n- do x"),

    # 9) AC at very end
    ("text then AC", "text then ", "AC"),

    # 10) ensure 'ac' inside words does not trigger (depends on your regex!)
    ("this package is updated", "this package is updated", ""),

    # 11) colon variant
    ("Acceptance Criteria: do A; do B", "", "Acceptance Criteria: do A; do B"),

    # 12) em-dash variant
    ("Acceptance Criteria—do this", "", "Acceptance Criteria—do this"),

    # 13) en-dash variant
    ("Acceptance Criteria–do this", "", "Acceptance Criteria–do this"),

    # 14) with parentheses near header
    ("Something (Acceptance Criteria) follows next", "Something ", "(Acceptance Criteria) follows next"),

    # 15) bracketed AC line
    ("[AC] steps here", "[", "AC] steps here"),  # will depend on your current pattern behavior

    # 16) lowercase phrase with punctuation
    ("we need acceptance criteria, then proceed", "we need ", "acceptance criteria, then proceed"),

    # 17) only 'ac' token with trailing spaces
    ("   ac    \n- do this", "   ", "ac    \n- do this"),

    # 18) AC preceded by punctuation
    ("note: acceptance criteria -> do x", "note: ", "acceptance criteria -> do x"),

    # 19) line equals 'ac'
    ("line1\nac\n- step", "line1\n", "ac\n- step"),

    # 20) nothing but whitespace
    ("   \n\t", "   \n\t", ""),
])
def test_split_ac(raw, expected_before, expected_after):
    before, after = split_ac(raw)

    if before != expected_before:
        diff_b = "\n".join(difflib.unified_diff(
            expected_before.splitlines(),
            before.splitlines(),
            fromfile="expected_before",
            tofile="actual_before",
            lineterm=""
        ))
        print("\n--- BEFORE MISMATCH ---")
        print("EXPECTED:\n", expected_before)
        print("ACTUAL:\n", before)
        print("DIFF:\n", diff_b)

    if after != expected_after:
        diff_a = "\n".join(difflib.unified_diff(
            expected_after.splitlines(),
            after.splitlines(),
            fromfile="expected_after",
            tofile="actual_after",
            lineterm=""
        ))
        print("\n--- AFTER MISMATCH ---")
        print("EXPECTED:\n", expected_after)
        print("ACTUAL:\n", after)
        print("DIFF:\n", diff_a)

    assert before == expected_before
    assert after == expected_after


if __name__ == "__main__":
    # Allow running this file directly: python test_ac_split.py
    raise SystemExit(pytest.main([os.path.abspath(__file__)]))
