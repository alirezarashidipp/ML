# ============================================================
# TEST_STEP_101_EMBEDDING.py
# - Simple pytest for STEP_101_JIRA_PRINC.py
# - Checks ONLY: role / what / why flags
#
# How to run:
#   python TEST_STEP_101_EMBEDDING.py
# or:
#   pytest -q TEST_STEP_101_EMBEDDING.py
# ============================================================

import sys
import os
import pytest
import difflib

# مسیر پروژه‌ات (همان فولدری که STEP_101_JIRA_PRINC.py داخلش است)
gcp_path = r"C:\Users\45315874\Desktop\EXTERNAL WORKS\JRE\Code\FINAL CODE"
if gcp_path not in sys.path:
    sys.path.append(gcp_path)

# ✅ Import from your main file
# NOTE: this expects STEP_101_JIRA_PRINC.py to expose:
#   - embedding_upgrade_flags(text, flags)
#   - TARGET_FLAGS
from STEP_101_JIRA_PRINC import embedding_upgrade_flags, TARGET_FLAGS


def run_step101_on_text(text: str):
    """
    We don't care about previous tagging.
    Start all 3 flags as 0 and let STEP_101 embedding fallback decide.
    """
    flags0 = {k: 0 for k in TARGET_FLAGS}  # role/what/why all false initially
    return embedding_upgrade_flags(text, flags0)


# ---------------- Tricky scenarios ----------------
@pytest.mark.parametrize("raw, expected", [
    # 1) Standard story (explicit Role + What)
    ("As a user, I want to login.", {
        "has_role_defined": 1,
        "has_goal_defined": 1,
        "has_reason_defined": 0,
    }),

    # 2) Role implied, no "As a ..." (should still often be caught by embedding)
    ("Admin should be able to reset user passwords.", {
        "has_role_defined": 1,
        "has_goal_defined": 1,
        "has_reason_defined": 0,
    }),

    # 3) What only, technical imperative
    ("Create a new table for the database.", {
        "has_role_defined": 0,
        "has_goal_defined": 1,
        "has_reason_defined": 0,
    }),

    # 4) Why without "so that" (implicit reason style)
    ("In order to reduce incidents, enforce MFA on login.", {
        "has_role_defined": 0,
        "has_goal_defined": 1,
        "has_reason_defined": 1,
    }),

    # 5) All three (one-line)
    ("As an administrator, I need to manage access in order to meet compliance.", {
        "has_role_defined": 1,
        "has_goal_defined": 1,
        "has_reason_defined": 1,
    }),

    # 6) Multi-sentence / chunking
    ("As a customer, I want to update my address. So that I can receive shipments correctly.", {
        "has_role_defined": 1,
        "has_goal_defined": 1,
        "has_reason_defined": 1,
    }),

    # 7) Pure noise / should stay 0
    ("Please investigate. Thanks.", {
        "has_role_defined": 0,
        "has_goal_defined": 0,
        "has_reason_defined": 0,
    }),

    # 8) Empty
    ("   ", {
        "has_role_defined": 0,
        "has_goal_defined": 0,
        "has_reason_defined": 0,
    }),
])
def test_step101_embedding_only(raw, expected):
    actual_full = run_step101_on_text(raw)

    # keep only 3 keys (role/what/why)
    actual = {k: int(actual_full.get(k, 0)) for k in expected.keys()}

    # nicer diffs
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
            print("TEXT:", raw)
            print("EXPECTED:", expected[key])
            print("ACTUAL:", actual[key])
            print("DIFF:\n", diff)

    assert actual == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__)]))
