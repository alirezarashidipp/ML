# test_ac_split.py
import pytest
from STEP_23_AC_SEPERATION import split_ac

@pytest.mark.parametrize("raw, expected_before, expected_after", [
    ("This is a simple text.", "This is a simple text.", ""),
    ("this is a text that does have ac and it is something.",
     "this is a text that does have ",
     "ac and it is something."),
    ("please see acceptance criteria: the steps are here",
     "please see ",
     "acceptance criteria: the steps are here"),
    ("ac: must follow these rules",
     "",
     "ac: must follow these rules"),
    ("intro... acceptance criteria first... later ac: second",
     "intro... ",
     "acceptance criteria first... later ac: second"),
])
def test_split_ac(raw, expected_before, expected_after):
    before, after = split_ac(raw)
    assert before == expected_before
    assert after == expected_after
