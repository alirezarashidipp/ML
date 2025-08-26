# test_cleaner.py
import pytest
from cleaner import clean_text

@pytest.mark.parametrize("raw,expected", [
    ("Non-AIM users", "non aim users."),
    ("summary:\nas an Ops user, i want to check.", "summary, as an ops user, i want to check."),
    (
        "as a customer\ni want to do this and that so i have new thing\n\nacceptence criterai:\nadding more column\nhaving new data",
        "as a customer i want to do this and that so i have new thing. acceptence criterai: adding more column having new data."
    ),
    ("*Acceptance Criteria:*\ngiven data works well.", "acceptance criteria: given data works well.")
])
def test_cleaning_cases(raw, expected):
    cleaned = clean_text(raw)
    assert cleaned == expected
