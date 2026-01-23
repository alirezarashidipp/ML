import sys
import os
import pytest
import difflib

# --------------------------------------------------
# Project path (adjust if needed)
# --------------------------------------------------
project_path = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL CODE"
if project_path not in sys.path:
    sys.path.append(project_path)

from STEP_200_JIRA_NLI_WHO_WHAT_WHY import detect_who_what_why


# --------------------------------------------------
# Coverage scenarios
# --------------------------------------------------
@pytest.mark.parametrize("raw, expected_flags", [

    # 1) WHO only
    (
        "As a user, I access the dashboard",
        {"WHO": True, "WHAT": False, "WHY": False}
    ),

    # 2) WHAT only
    (
        "The system should generate a report",
        {"WHO": False, "WHAT": True, "WHY": False}
    ),

    # 3) WHY only
    (
        "So that users can work faster and reduce errors",
        {"WHO": False, "WHAT": False, "WHY": True}
    ),

    # 4) WHO + WHAT
    (
        "As an admin, I want to manage users",
        {"WHO": True, "WHAT": True, "WHY": False}
    ),

    # 5) WHAT + WHY
    (
        "The system should validate inputs in order to reduce risk",
        {"WHO": False, "WHAT": True, "WHY": True}
    ),

    # 6) WHO + WHAT + WHY (classic user story)
    (
        "As a developer, I want to deploy the code so that releases are faster",
        {"WHO": True, "WHAT": True, "WHY": True}
    ),

    # 7) Empty / invalid input
    (
        "   ",
        {"WHO": False, "WHAT": False, "WHY": False}
    ),
])
def test_detect_who_what_why_flags(raw, expected_flags):

    result = detect_who_what_why(raw)
    actual_flags = result["flags"]

    # --- detailed diff on mismatch (like your example) ---
    for key in expected_flags:
        if actual_flags[key] != expected_flags[key]:
            diff = "\n".join(difflib.unified_diff(
                [str(expected_flags[key])],
                [str(actual_flags[key])],
                fromfile=f"expected[{key}]",
                tofile=f"actual[{key}]",
                lineterm=""
            ))
            print(f"\n--- MISMATCH on {key} ---")
            print("EXPECTED:", expected_flags[key])
            print("ACTUAL:", actual_flags[key])
            print("DIFF:\n", diff)

    assert actual_flags == expected_flags


# --------------------------------------------------
# Optional: sanity check on score ranges
# --------------------------------------------------
def test_scores_are_valid_probabilities():
    text = "As a user, I want to reset my password so that I can regain access"

    result = detect_who_what_why(text)
    scores = result["scores"]

    for dim, score in scores.items():
        assert 0.0 <= score <= 1.0, f"{dim} score out of range: {score}"


# --------------------------------------------------
# CLI entry for local execution
# --------------------------------------------------
if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__)]))
