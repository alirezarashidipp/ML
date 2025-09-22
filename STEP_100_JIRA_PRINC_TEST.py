import sys
import os
import pytest
import difflib

# مسیر پروژه‌ات
gcp_path = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL CODE"
if gcp_path not in sys.path:
    sys.path.append(gcp_path)

from STEP_100_JIRA_PRINC import flag_story_quality

# ---------------- Coverage scenarios ----------------
@pytest.mark.parametrize("raw, expected", [
    # 1) Text with role
    ("As a user I want to login", {
        "has_acceptance_criteria": 0,
        "has_role_defined": 1,
        "has_goal_defined": 1,
        "has_reason_defined": 0,
    }),
    # 2) Text with acceptance criteria
    ("Acceptance Criteria: user must login", {
        "has_acceptance_criteria": 1,
        "has_role_defined": 0,
        "has_goal_defined": 0,
        "has_reason_defined": 0,
    }),
    # 3) Regex variant (AC:)
    ("ac: must be done", {
        "has_acceptance_criteria": 1,
        "has_role_defined": 0,
        "has_goal_defined": 0,
        "has_reason_defined": 0,
    }),
    # 4) Goal phrase
    ("I would like to reset my password", {
        "has_acceptance_criteria": 0,
        "has_role_defined": 0,
        "has_goal_defined": 1,
        "has_reason_defined": 0,
    }),
    # 5) Reason phrase
    ("So that I can track my expenses", {
        "has_acceptance_criteria": 0,
        "has_role_defined": 0,
        "has_goal_defined": 0,
        "has_reason_defined": 1,
    }),
    # 6) Mixed (role + goal + reason)
    ("As an admin I want to manage users so that they have correct access", {
        "has_acceptance_criteria": 0,
        "has_role_defined": 1,
        "has_goal_defined": 1,
        "has_reason_defined": 1,
    }),
    # 7) Empty text
    ("   ", {
        "has_acceptance_criteria": 0,
        "has_role_defined": 0,
        "has_goal_defined": 0,
        "has_reason_defined": 0,
    }),
])
def test_flag_story_quality(raw, expected):
    actual = flag_story_quality(raw)

    # Compare dicts key by key
    for key in expected:
        if actual[key] != expected[key]:
            diff = "\n".join(difflib.unified_diff(
                [str(expected[key])],
                [str(actual[key])],
                fromfile=f"expected[{key}]",
                tofile=f"actual[{key}]",
                lineterm=""
            ))
            print(f"\n--- MISMATCH on {key} ---")
            print("EXPECTED:", expected[key])
            print("ACTUAL:", actual[key])
            print("DIFF:\n", diff)

    assert actual == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__)]))
