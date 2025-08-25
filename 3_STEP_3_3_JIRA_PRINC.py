import re
import spacy
from spacy.matcher import Matcher
from typing import Dict

# Feature keys
_FEATURE_KEYS = [
    'has_acceptance_criteria',
    'has_role_defined',
    'has_goal_defined',
    'has_reason_defined',
]

# Load spaCy model and initialize matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)


pattern_role = [
    {"LOWER": "as"},
    {"LOWER": {"IN": ["a", "an", "the"]}},
    {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}
     ]


matcher.add("has_role_defined", [pattern_role])

# 2. Goal: various phrases
goal_phrases = [
"i want to", "i would like to", "i need to", "i can", 
"i am able to", "wish to", "we need to",


"we want to", "we would like to", "we need to", "we can", 
"we are able to", "wish to",

"i'm able to", 
"we're able to", 
"i'd like to", 
"we'd like to",

"im able to", 
"were able to", 
"id like to", 
"wed like to",]


patterns_goal = []
for phrase in goal_phrases:
    tokens = phrase.split()
    patterns_goal.append([{"LOWER": tok} for tok in tokens])
matcher.add("has_goal_defined", patterns_goal)

# 3. Reason: "so that", "so i can", "in order to"
reason_phrases = ["so that", "so i can", "in order to"]
patterns_reason = []
for phrase in reason_phrases:
    tokens = phrase.split()
    patterns_reason.append([{"LOWER": tok} for tok in tokens])
matcher.add("has_reason_defined", patterns_reason)



# 4. Acceptance: phrase-based and regex fallback
acceptance_phrases = ["acceptance criteria", "AC", "acceptance criterion", "acc. crit."]
patterns_acceptance = []
for phrase in acceptance_phrases:
    tokens = phrase.split()
    patterns_acceptance.append([{"LOWER": tok.strip(".") if tok.endswith('.') else tok} for tok in tokens])
matcher.add("has_acceptance_criteria", patterns_acceptance)
# regex fallback for variants like ac:, a/c:
_ACCEPTANCE_REGEX = re.compile(r"\b(?:ac|a/c)[:\-]?\b", re.I)


def flag_story_quality(description: str) -> Dict[str, int]:
    # Ensure string input
    if not isinstance(description, str):
        return {k: 0 for k in _FEATURE_KEYS}

    flags = {k: 0 for k in _FEATURE_KEYS}





    if _ACCEPTANCE_REGEX.search(description):
        flags['has_acceptance_criteria'] = 1

    # Normalize newlines and whitespace
    raw_text = description.replace("\r\n", "\n").replace("\r", "\n")
    raw_text = re.sub(r"[\{\}\[\]\*\—\-\-_/\\+#]", " ", raw_text)
    raw_text = re.sub(r"\s{2,}", " ", raw_text).strip()
    
    # Run spaCy matcher
    doc = nlp(raw_text)
    matches = matcher(doc)
    for match_id, start, end in matches:
        feature = nlp.vocab.strings[match_id]
        flags[feature] = 1

    # Fallback for acceptance regex
    if not flags['has_acceptance_criteria'] and _ACCEPTANCE_REGEX.search(raw_text):
        flags['has_acceptance_criteria'] = 1

    return flags

