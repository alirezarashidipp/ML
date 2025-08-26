# test_cleaner_mvp.py
# Minimal tests for the final cleaner
# - Assumes you have: from cleaner_final import clean_text, clean_df
# - Run: python -m unittest test_cleaner_mvp.py -v

import unittest
import pandas as pd

# Import your functions here
from cleaner_final import clean_text, clean_df

class TestCleanerMVP(unittest.TestCase):

    def test_tables_flatten_with_commas(self):
        raw = "|| Role || Action ||\n| user | login |\n| admin | delete |"
        out = clean_text(raw)
        # Expect: header dropped, cells joined by ", ", row ends with "."
        self.assertIn("user, login.", out)
        self.assertIn("admin, delete.", out)

    def test_bullets_to_sentences(self):
        raw = "* first item\n- second item\n# third item\n1. fourth item"
        out = clean_text(raw)
        # Each item should become a sentence and end with a period
        for s in ["first item.", "second item.", "third item.", "fourth item."]:
            self.assertIn(s, out)

    def test_acceptance_criteria_inline_and_block(self):
        raw = "ac1: user can login\n\nac2:\npassword must be strong\nand have symbols"
        out = clean_text(raw)
        self.assertIn("acceptance criteria: user can login.", out)
        self.assertIn("acceptance criteria: password must be strong and have symbols.", out)

    def test_links_emails_urls_removed_keep_text(self):
        raw = "see [the docs|https://example.com] and write to a@b.com, visit https://x.y"
        out = clean_text(raw)
        # Link text kept, URLs/emails removed
        self.assertIn("see the docs", out)
        self.assertNotIn("http", out)
        self.assertNotIn("@", out)

    def test_code_blocks_removed(self):
        raw = "{code}\nSELECT * FROM X;\n{code}\ntext after code"
        out = clean_text(raw)
        self.assertIn("text after code", out)
        self.assertNotIn("select * from x", out)

    def test_versions_kept_numbers_dropped(self):
        raw = "release v1.2.3 with 42 bugs fixed and version 2.0 confirmed"
        out = clean_text(raw)
        # versions kept
        self.assertIn("v1.2.3", out)
        self.assertIn("2.0", out)
        # plain numbers dropped (42 should be removed or isolated away)
        self.assertNotIn("42", out)

    def test_headings_quotes_markdown_headings(self):
        raw = "h2. Title here\n# Another Title\nbq. quoted text\nnormal line"
        out = clean_text(raw)
        # headings/quote markers removed but text preserved or merged
        self.assertIn("title here", out)
        self.assertIn("another title", out)
        self.assertIn("quoted text", out)
        self.assertIn("normal line", out)

    def test_sentence_segmentation_and_final_punctuation(self):
        raw = "as a business I want X\nso that I get Y"
        out = clean_text(raw)
        # Should become two sentences, each ending with a period
        self.assertIn("as a business i want x.", out)
        self.assertIn("so that i get y.", out)
        self.assertTrue(out.endswith("."))

    def test_idempotency(self):
        raw = "h1. Title\n* item one\n* item two\n||H||\n|a|b|"
        once = clean_text(raw)
        twice = clean_text(once)
        self.assertEqual(once, twice)

class TestCleanerDF(unittest.TestCase):

    def test_clean_df_adds_output_column(self):
        df = pd.DataFrame({
            "ISSUE_DESC_STR": [
                "h1. Title\nac1: user can login",
                "| a | b |\n| c | d |"
            ],
            "Other": [1, 2]
        })
        out = clean_df(df, text_col="ISSUE_DESC_STR", out_col="ISSUE_DESC_STR_CLEANED")
        self.assertIn("ISSUE_DESC_STR_CLEANED", out.columns)
        self.assertEqual(len(out), 2)
        # sanity on outputs
        self.assertIn("acceptance criteria: user can login.", out.loc[0, "ISSUE_DESC_STR_CLEANED"])
        self.assertIn("a, b.", out.loc[1, "ISSUE_DESC_STR_CLEANED"])
        self.assertIn("c, d.", out.loc[1, "ISSUE_DESC_STR_CLEANED"])

if __name__ == "__main__":
    unittest.main()
