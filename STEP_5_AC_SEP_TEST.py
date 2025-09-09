
import pytest
from STEP_23_AC_SEPERATION import split_ac   

@pytest.mark.parametrize("raw, expected_before, expected_after", [
    # 1. plain text without AC
    ("This is a simple text.", "This is a simple text.", ""),
    # 2. with 'ac' in the middle
    ("this is a text that does have ac and it is something.",
     "this is a text that does have ",
     "ac and it is something."),
    # 3. with 'acceptance criteria'
    ("please see acceptance criteria: the steps are here",
     "please see ",
     "acceptance criteria: the steps are here"),
    # 4. 'ac' at start
    ("ac: must follow these rules",
     "",
     "ac: must follow these rules"),
    # 5. multiple mentions (should split at first)
    ("intro... acceptance criteria first... later ac: second",
     "intro... ",
     "acceptance criteria first... later ac: second"),
])
def test_split_ac(raw, expected_before, expected_after):
    before, after = split_ac(raw)
    assert before == expected_before
    assert after == expected_after
