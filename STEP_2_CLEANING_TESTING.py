# test_cleaner_all.py
import pytest
from cleaner import clean_text

# -------------------------
# Main 20 coverage scenarios
# -------------------------
@pytest.mark.parametrize("raw,expected", [
    # 1) color tags should be removed
    ("{color:#FF0000}something{color}", "something."),
    # 2) internal hyphen between word chars -> space
    ("Non-AIM users", "non aim users."),
    # 3) short-line comma rule (label + next line joined with comma)
    ("summary:\nas an Ops user, i want to check.", "summary, as an ops user, i want to check."),
    # 4) acceptance criteria header with markdown
    ("*Acceptance Criteria:*\ngiven data works well.", "acceptance criteria: given data works well."),
    # 5) inline acceptance criteria (ac1:)
    ("ac1: add new column to table", "acceptance criteria: add new column to table."),
    # 6) acceptance criteria header followed by multiple lines
    ("ac2:\nfirst point\nsecond point\n\nnext", "acceptance criteria: first point second point. next."),
    # 7) Atlassian code block should be removed
    ("before {code}print('x'){code} after", "before after."),
    # 8) Atlassian image should be removed
    ("see !image.png! please", "see please."),
    # 9) URLs should be removed
    ("check https://example.com now", "check now."),
    # 10) emails should be removed
    ("email me at test.user+prod@example.co.uk", "email me at."),
    # 11) Jira IDs should be removed
    ("Fix in PROJ-1234 asap", "fix in asap."),
    # 12) wiki heading (h2.) should be removed and joined with comma
    ("h2. Title\nNext line here", "title, next line here."),
    # 13) markdown heading (#) should be removed and joined with comma
    ("# Heading\nBody continues", "heading, body continues."),
    # 14) bullet points should be flattened
    ("* first item\n* second item", "first item, second item."),
    # 15) realistic table: drop header and flatten each row
    (
        "|| Name || Role || Age ||\n"
        "| Alice | Developer | 30 |\n"
        "| Bob   | Tester    | 28 |\n"
        "| Carol | Manager   | 35 |",
        "alice, developer, 30. bob, tester, 28. carol, manager, 35."
    ),
    # 16) versions/decimals should be preserved, extra punctuation dropped
    ("updated to v1.2.3 and 2.0 today!!!", "updated to v1.2.3 and 2.0 today."),
    # 17) plain numbers should be dropped
    ("we have 123 bugs and 45 tasks", "we have bugs and tasks."),
    # 18) abbreviations should be preserved
    ("works fine, i.e. acceptable e.g. ok etc.", "works fine, i.e. acceptable e.g. ok etc."),
    # 19) links [text|url] and [text] should be simplified to text
    ("See [Click here|http://x.y] and also [More]", "see click here and also more."),
    # 20) styles and contractions should be cleaned
    # Note: contractions.fix converts "Don't" -> "do not"
    ("{{CODE}} _text_ -del- +ins+ Don't do it", "code text del ins do not do it."),
])
def test_cleaning_coverage(raw, expected):
    cleaned = clean_text(raw)
    assert cleaned == expected


# -----------------------------------
# Extra 12 edge-case scenarios (Z1-Z12)
# -----------------------------------
@pytest.mark.parametrize("raw,expected", [
    # Z1) zero-width and nbsp normalization
    ("A\u200B\u200D\u2060B\u00A0C", "a b c."),
    # Z2) nested HTML tags and entities
    ("<b><i>Bold&nbsp;Ital</i></b> <span>Text</span>", "bold ital text."),
    # Z3) noformat block should be stripped like code
    ("keep {noformat}\nraw *stars*\n{noformat} end", "keep end."),
    # Z4) fenced code block (multiline) removed
    ("before ```\ncode line 1\ncode line 2\n```\nafter", "before after."),
    # Z5) acceptance-criteria phrase header (common misspelling)
    # If you haven't implemented the AC phrase-header patch,
    # this will likely fail; marked xfail below.
    ("Acceptence Criterai:\nDo X\nDo Y", "acceptence criterai: do x do y."),
    # Z6) en/em dashes and chained hyphens inside words
    ("pre–prod pre—prod A-B-C re-enter", "pre prod pre prod a b c re enter."),
    # Z7) short-line comma with exact 4 words + trailing colon
    ("note this now:\nrun the check", "note this now, run the check."),
    # Z8) short-line rule when next line is empty (should not join)
    ("title:\n\nbody continues", "title. body continues."),
    # Z9) long line without punctuation (should get period if >= min_len)
    ("this is a quite long line that should end with a period automatically",
     "this is a quite long line that should end with a period automatically."),
    # Z10) IP address and time; numbers should be dropped but sentence still valid
    ("connect to 10.0.0.1 at 10:30", "connect to at."),
    # Z11) table with empty cells and extra pipes
    ("|| A || B || C ||\n| 1 | | val |\n| | z | |\n| k | m | n |",
     "val. z. k, m, n."),
    # Z12) links with spaces around pipe and simple [text]
    ("See [Click here | https://x.y] & also [More]", "see click here also more."),
])
def test_cleaning_edge_cases(raw, expected):
    cleaned = clean_text(raw)
    assert cleaned == expected


# Mark Z5 as xfail until AC phrase header support is implemented.
@pytest.mark.xfail(reason="AC phrase header misspelling not yet consolidated without patch")
def test_Z5_phrase_header_acceptance_criteria():
    raw = "Acceptence Criterai:\nDo X\nDo Y"
    expected = "acceptence criterai: do x do y."
    cleaned = clean_text(raw)
    assert cleaned == expected
