# ============================================================
# scorer2.py
# ============================================================
#
# WHAT THIS DOES (read this first):
#
#   For EVERY word in every input sentence — including small
#   words like "I", "in", "the", "before", "along" — this
#   script does the following steps:
#
#   STEP 1  Tokenise the sentence (split into individual words)
#           and POS-tag each word (is it a noun? verb? pronoun?)
#
#   STEP 2  Look up WordNet for EVERY word.
#           WordNet is a large English dictionary organised by
#           meaning. Each entry has:
#             - synonyms  (words with the same meaning)
#             - hypernyms (broader/more general words)
#             - hyponyms  (specific/more narrow words) <-- we want these
#           For "bank" (financial sense):
#             hyponyms = ["savings bank", "commercial bank",
#                         "credit union", "central bank", ...]
#           These "specific words" are what we score next.
#
#   STEP 3  For ambiguous content words (nouns, verbs, adj, adv
#           with 2+ senses), run ConSeC FIRST to pick the correct
#           WordNet sense for THIS sentence's context.
#           Example: "bank" in "I deposited money in the bank"
#           -> ConSeC picks sense: "financial institution"
#           Then we pull hyponyms of THAT specific sense only.
#
#   STEP 4  Score each specific word (hyponym) with ConSeC.
#           We ask: "how well does this specific word fit at
#           the position of the original word in the sentence?"
#           ConSeC returns a probability 0.0-1.0.
#           Higher score = better fit in this context.
#
#   STEP 5  Sort specific words by score (highest first).
#           Attach the top-scored specific word to the original:
#             "bank"  +  "savings bank" (score 0.91)
#             ->  key becomes  "bank_savingsBank"
#
#   STEP 6  Save output in BOTH formats:
#           JSON:  { "bank_savingsBank": ["savings bank (0.91)", ...] }
#           CSV:   one row per word, all columns
#
# IMPORTANT NOTE ON FUNCTION WORDS ("in", "the", "before"):
#   These words are NOT nouns/verbs. WordNet has very limited
#   entries for them. We handle them with a FALLBACK TABLE that
#   maps each common function word to a descriptive label.
#   "the"    -> "definite article (1.00)"
#   "in"     -> looks up WordNet ADV synsets for inside/within
#   "before" -> looks up WordNet ADV synsets for previously/earlier
#   If WordNet has real hyponyms for them, we score those too.
#   If not, the fallback label is shown with score 1.00.
#
# USAGE:
#   python scorer2.py                    reads sentences.txt
#   python scorer2.py myfile.txt         custom input
#
# OUTPUT:
#   scorer_output.json     the exact format you requested
#   scorer_output.csv      one row per word
# ============================================================

import sys
import csv
import json
import os

from nltk.corpus import wordnet as wn

# Import from your existing wsd_pipeline.py -- never modified
from wsd_pipeline import (
    tokenize_and_tag,
    detect_ambiguous_words,
    disambiguate_word,
    SPACY_TO_WN_POS,
)


# ============================================================
# FALLBACK TABLE
# ============================================================
# Words that have no WordNet hyponyms (or no WordNet entry at
# all) get a descriptive label instead. Score is always 1.00
# because there is no ambiguity for structural words.
#
# Add more words here if you find any being skipped.

FALLBACK_LABELS = {
    "i":      [("first person singular", 1.00)],
    "me":     [("first person object form", 1.00)],
    "my":     [("first person possessive", 1.00)],
    "we":     [("first person plural", 1.00)],
    "our":    [("first person plural possessive", 1.00)],
    "you":    [("second person pronoun", 1.00)],
    "your":   [("second person possessive", 1.00)],
    "he":     [("third person masculine singular", 1.00)],
    "she":    [("third person feminine singular", 1.00)],
    "they":   [("third person plural", 1.00)],
    "it":     [("third person neutral", 1.00)],
    "the":    [("definite article", 1.00)],
    "a":      [("indefinite article", 1.00)],
    "an":     [("indefinite article", 1.00)],
    "and":    [("coordinating conjunction", 1.00)],
    "or":     [("disjunctive conjunction", 1.00)],
    "but":    [("contrastive conjunction", 1.00)],
    "so":     [("resultative conjunction", 1.00)],
    "yet":    [("contrastive adverb", 1.00)],
    "nor":    [("negative conjunction", 1.00)],
    "of":     [("relational preposition", 1.00)],
    "to":     [("directional preposition or infinitive marker", 1.00)],
    "for":    [("purposive preposition", 1.00)],
    "on":     [("locative preposition", 1.00)],
    "at":     [("locative preposition", 1.00)],
    "by":     [("agentive preposition", 1.00)],
    "with":   [("comitative preposition", 1.00)],
    "from":   [("ablative preposition", 1.00)],
    "as":     [("comparative conjunction", 1.00)],
    "into":   [("directional preposition", 1.00)],
    "onto":   [("directional preposition", 1.00)],
    "upon":   [("locative preposition", 1.00)],
    "is":     [("third person singular present of be", 1.00)],
    "are":    [("plural present of be", 1.00)],
    "was":    [("past singular of be", 1.00)],
    "were":   [("past plural of be", 1.00)],
    "be":     [("infinitive of be", 1.00)],
    "been":   [("past participle of be", 1.00)],
    "have":   [("auxiliary verb have", 1.00)],
    "has":    [("third person singular of have", 1.00)],
    "had":    [("past tense of have", 1.00)],
    "do":     [("auxiliary verb do", 1.00)],
    "does":   [("third person singular of do", 1.00)],
    "did":    [("past tense of do", 1.00)],
    "will":   [("future auxiliary", 1.00)],
    "would":  [("conditional auxiliary", 1.00)],
    "could":  [("past or conditional of can", 1.00)],
    "should":  [("deontic modal auxiliary", 1.00)],
    "may":    [("epistemic modal auxiliary", 1.00)],
    "might":  [("weak epistemic modal", 1.00)],
    "shall":  [("future obligation auxiliary", 1.00)],
    "must":   [("necessity modal", 1.00)],
    "not":    [("negation particle", 1.00)],
    "no":     [("negative determiner", 1.00)],
    "this":   [("proximal demonstrative", 1.00)],
    "that":   [("distal demonstrative", 1.00)],
    "these":  [("proximal plural demonstrative", 1.00)],
    "those":  [("distal plural demonstrative", 1.00)],
}


# ============================================================
# STEP 1 — GET SPECIFIC WORDS (HYPONYMS) FROM WORDNET
# ============================================================
#
# WHY HYPONYMS (specific words) instead of synonyms?
#
# You asked for "specific words" for each word.
# In WordNet, "specific words" are called HYPONYMS.
#
# Hyponyms go from general -> specific:
#   "bank" (financial) -> hyponyms = ["savings bank", "credit union",
#                                      "commercial bank", "central bank"]
#   "walk"             -> hyponyms = ["shuffle", "march", "stride",
#                                      "stroll", "saunter", "trudge"]
#   "river"            -> hyponyms = ["Amazon", "Nile", "tributary",
#                                      "headwaters", "creek"]
#
# These are EXACTLY the "specific words" you described in your example.
#
# HOW WE FIND THE RIGHT SYNSET:
#   If ConSeC already picked a sense (e.g. bank.n.01 = financial),
#   we use that synset directly.
#   Otherwise we use the first (most common) synset.

def get_specific_words(lemma, wn_pos, chosen_synset_name=None):
    """
    Gets hyponyms (specific words) for a word from WordNet.

    Parameters
    ----------
    lemma              : str  e.g. "bank", "walk", "river"
    wn_pos             : WordNet POS constant e.g. wn.NOUN
    chosen_synset_name : str  e.g. "bank.n.01"  (from ConSeC result)
                         If given, we use exactly this synset.
                         If None, we use the first (most common) synset.

    Returns
    -------
    list of str  -- the hyponym lemma names (deduplicated)
    str          -- the synset name that was used
    """
    if wn_pos is None:
        return [], None

    # Try to find synsets for this lemma+POS
    synsets = wn.synsets(lemma, pos=wn_pos)
    if not synsets:
        synsets = wn.synsets(lemma)   # try all POS as fallback
    if not synsets:
        return [], None

    # Pick the correct synset
    target_synset = synsets[0]   # default: most common sense

    if chosen_synset_name:
        for ss in synsets:
            if ss.name() == chosen_synset_name:
                target_synset = ss
                break

    # Collect all hyponyms (specific words) from this synset
    # wn uses .hyponyms() to get one level down in the hierarchy
    specifics = []
    seen = set()

    for hypo_synset in target_synset.hyponyms():
        for lemma_obj in hypo_synset.lemmas():
            name = lemma_obj.name().replace("_", " ")
            name_lower = name.lower()
            if name_lower not in seen and name_lower != lemma.lower():
                specifics.append(name)
                seen.add(name_lower)

    return specifics, target_synset.name()


# ============================================================
# STEP 2 — SCORE EACH SPECIFIC WORD WITH CONSEC
# ============================================================
#
# WHY SCORE SPECIFIC WORDS?
#   Not all hyponyms fit equally well in a given sentence.
#   "I deposited money in the bank"
#   "bank" hyponyms: ["savings bank", "commercial bank", "piggy bank"]
#   "savings bank" fits much better here than "piggy bank".
#
# HOW SCORING WORKS:
#   We ask ConSeC: "at position P in this sentence, which of
#   these specific words' definitions fits best?"
#   ConSeC returns a probability score for each candidate.
#   We return them sorted highest score first.
#
# WHAT IF A SPECIFIC WORD IS NOT IN WORDNET ITSELF?
#   Some hyponym names (e.g. "Amazon" as a river) may not have
#   WordNet synsets of their own. In that case we use the
#   hyponym synset's definition as the candidate text directly.

def score_specific_words_with_consec(specifics, sentence, position,
                                     wn_pos, tagged):
    """
    Scores each specific word using ConSeC in the sentence context.

    Parameters
    ----------
    specifics  : list of str  e.g. ["savings bank", "credit union", ...]
    sentence   : str          full original sentence
    position   : int          0-indexed token position of the original word
    wn_pos     : WordNet POS constant
    tagged     : list of dicts from tokenize_and_tag()

    Returns
    -------
    list of (score, specific_word)  sorted descending by score
    """
    if not specifics or wn_pos is None:
        return []

    tokens = [t["word"] for t in tagged]

    # Map wn_pos back to a spaCy string (needed by disambiguate_word)
    wn_to_spacy = {
        wn.NOUN: "NOUN",
        wn.VERB: "VERB",
        wn.ADJ:  "ADJ",
        wn.ADV:  "ADV",
    }
    spacy_pos_str = wn_to_spacy.get(wn_pos, "NOUN")

    scored = []

    for sp_word in specifics:
        sp_lemma = sp_word.replace(" ", "_")

        # Find synsets for this specific word to get its definition
        sp_synsets = wn.synsets(sp_lemma, pos=wn_pos)
        if not sp_synsets:
            sp_synsets = wn.synsets(sp_word, pos=wn_pos)
        if not sp_synsets:
            # No synset found -- skip this specific word
            continue

        num_senses = len(sp_synsets)

        # Build the word_info dict that disambiguate_word() expects
        word_info = {
            "word":       sp_word,
            "lemma":      sp_lemma,
            "pos":        spacy_pos_str,
            "wn_pos":     wn_pos,
            "position":   position,
            "num_senses": num_senses,
        }

        try:
            results = disambiguate_word(sentence, word_info, tokens)
            if results:
                score = round(float(results[0][0]), 4)
                scored.append((score, sp_word))
        except Exception:
            # If ConSeC fails for this word, skip it silently
            pass

    # Sort highest score first
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ============================================================
# STEP 3 — CAMEL-CASE HELPER FOR THE KEY NAME
# ============================================================
#
# The output key format is:  originalWord_bestSpecificWord
#
# Examples:
#   "bank"  + "savings bank"       -> "bank_savingsBank"
#   "walk"  + "march"              -> "walk_march"
#   "river" + "Amazon River"       -> "river_amazonRiver"
#
# We camelCase the specific word so the key has no spaces.

def to_camel(text):
    """
    Converts a phrase to camelCase.
    "savings bank" -> "savingsBank"
    "financial institution" -> "financialInstitution"
    """
    parts = text.strip().split()
    if not parts:
        return text
    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:])


def make_key(original_word, best_specific):
    """
    Builds the output key: word_bestSpecific
    If no specific word found, returns just the original word.
    """
    if not best_specific:
        return original_word
    return f"{original_word}_{to_camel(best_specific)}"


# ============================================================
# STEP 4 — PROCESS ONE WORD FULLY
# ============================================================

def process_one_word(token_info, sentence, tagged,
                     chosen_sense=None, chosen_synset=None):
    """
    Full pipeline for one token.

    Parameters
    ----------
    token_info      : dict from tokenize_and_tag() -- has word, pos, idx, lemma
    sentence        : str  full sentence
    tagged          : list of all token dicts (for ConSeC context)
    chosen_sense    : str  definition ConSeC picked (if already run)
    chosen_synset   : str  synset name e.g. "bank.n.01" (if already run)

    Returns
    -------
    dict with keys:
      original_word   : the token string
      label           : word_bestSpecific (the output key)
      best_specific   : top-scored specific word
      best_score      : score of the top specific word
      specifics_scored: list of {"specific": str, "score": float}
    """
    word      = token_info["word"]
    spacy_pos = token_info["pos"]
    position  = token_info["idx"]
    lemma     = token_info.get("lemma", word.lower())

    # Skip punctuation
    if spacy_pos in ("PUNCT", "SPACE", "SYM", "X"):
        return None

    # Map spaCy POS to WordNet POS
    wn_pos = SPACY_TO_WN_POS.get(spacy_pos)

    # ── Try WordNet hyponyms first ─────────────────────────
    specifics_raw, used_synset = get_specific_words(
        lemma, wn_pos, chosen_synset
    )

    if specifics_raw:
        # Score each specific word with ConSeC
        scored = score_specific_words_with_consec(
            specifics_raw, sentence, position, wn_pos, tagged
        )

        if scored:
            best_score    = scored[0][0]
            best_specific = scored[0][1]
        else:
            # Scoring returned nothing (all words failed WordNet lookup)
            # Fall back to unscored list with 0.0 scores
            scored        = [(0.0, s) for s in specifics_raw[:10]]
            best_score    = 0.0
            best_specific = specifics_raw[0] if specifics_raw else None

        label = make_key(word, best_specific)

        return {
            "original_word":    word,
            "label":            label,
            "best_specific":    best_specific,
            "best_score":       best_score,
            "used_synset":      used_synset or "",
            "chosen_sense":     chosen_sense or "",
            "specifics_scored": [
                {"specific": sp, "score": sc}
                for sc, sp in scored
            ],
        }

    # ── Fallback for function words / words not in WordNet ──
    # Check the fallback table (exact word match, lowercase)
    fallback = FALLBACK_LABELS.get(word.lower())

    if fallback:
        label_text, label_score = fallback[0]
        return {
            "original_word":    word,
            "label":            word,   # no specific to attach for these words
            "best_specific":    label_text,
            "best_score":       label_score,
            "used_synset":      "",
            "chosen_sense":     label_text,
            "specifics_scored": [
                {"specific": lbl, "score": sc}
                for lbl, sc in fallback
            ],
        }

    # ── Word has no WordNet entry AND no fallback ──────────
    # Return it as-is so the output is complete for every word.
    return {
        "original_word":    word,
        "label":            word,
        "best_specific":    None,
        "best_score":       None,
        "used_synset":      "",
        "chosen_sense":     "",
        "specifics_scored": [],
    }


# ============================================================
# STEP 5 — PROCESS A FULL SENTENCE
# ============================================================
#
# KEY DESIGN DECISION:
#   We run run_wsd_pipeline logic first on ambiguous content
#   words so ConSeC picks the correct sense BEFORE we fetch
#   hyponyms. This ensures:
#     "bank" in "I deposited money in the bank"
#     -> ConSeC says: bank.n.01 (financial institution)
#     -> hyponyms of bank.n.01: savings bank, credit union...
#     NOT hyponyms of bank.n.09 (river bank)!
#
#   Without this step you would get WRONG specific words.

def process_sentence(sentence_id, sentence):
    """
    Processes every token in the sentence.

    Returns a dict:
      sentence_id : int
      sentence    : str
      words       : list of per-word result dicts
    """
    print(f"\n[{sentence_id}] Processing: {sentence}")

    doc, tagged = tokenize_and_tag(sentence)
    tokens = [t["word"] for t in tagged]

    # ── Run ConSeC on ambiguous words first ─────────────────
    # This gives us the correct synset for each ambiguous word
    # so we can fetch the RIGHT hyponyms later.

    # detect_ambiguous_words returns only ambiguous content words
    ambiguous = detect_ambiguous_words(tagged)

    # Build a lookup: token position -> (chosen_sense, synset_name)
    consec_results = {}   # {position: {"sense": str, "synset": str}}

    for word_info in ambiguous:
        pos_idx = word_info["position"]
        try:
            results = disambiguate_word(sentence, word_info, tokens)
            if results:
                best_score, best_def = results[0]

                # Find the matching synset name
                lemma  = word_info["lemma"]
                wn_pos = word_info["wn_pos"]
                synset_name = None

                if wn_pos:
                    for ss in wn.synsets(lemma, pos=wn_pos):
                        if ss.definition() == best_def:
                            synset_name = ss.name()
                            break

                consec_results[pos_idx] = {
                    "sense":   best_def,
                    "synset":  synset_name,
                    "score":   round(float(best_score), 4),
                }
                print(f"   ConSeC: '{word_info['word']}' -> {best_def[:55]}  "
                      f"[{best_score:.4f}]")

        except Exception as e:
            print(f"   Warning: ConSeC failed for '{word_info['word']}': {e}")

    # ── Now process every token ──────────────────────────────
    word_results = []

    for token_info in tagged:
        word    = token_info["word"]
        pos_idx = token_info["idx"]

        # Skip punctuation
        if token_info["pos"] in ("PUNCT", "SPACE", "SYM", "X"):
            continue

        # Get ConSeC result for this token if available
        cr             = consec_results.get(pos_idx, {})
        chosen_sense   = cr.get("sense")
        chosen_synset  = cr.get("synset")

        print(f"   '{word}' ({token_info['pos']})", end=" ", flush=True)

        result = process_one_word(
            token_info, sentence, tagged,
            chosen_sense=chosen_sense,
            chosen_synset=chosen_synset,
        )

        if result is None:
            print("-> skipped (punctuation)")
            continue

        # Short preview for terminal
        preview = ", ".join(
            f"{s['specific']}({s['score']:.2f})"
            for s in result["specifics_scored"][:3]
        )
        print(f"-> {result['label']}  [{preview}]")

        word_results.append(result)

    return {
        "sentence_id": sentence_id,
        "sentence":    sentence,
        "words":       word_results,
    }


# ============================================================
# FORMAT OUTPUT  (the exact format you asked for)
# ============================================================

def format_output_dict(sentence_result):
    """
    Converts internal result into the requested JSON format:

    {
      "I":                    ["first person singular (1.00)"],
      "deposited_put":        ["put (0.93)", "placed (0.89)", ...],
      "money_cash":           ["cash (0.96)", "currency (0.91)", ...],
      "in_inside":            ["inside (0.95)", "within (0.90)", ...],
      "the":                  ["definite article (1.00)"],
      "bank_savingsBank":     ["savings bank (0.91)", "credit union (0.80)", ...],
      "before_previously":    ["previously (0.88)", "earlier (0.72)", ...],
      "walking_march":        ["march (0.85)", "stride (0.79)", ...],
      "along_beside":         ["beside (0.82)", "alongside (0.71)", ...],
      "river_tributary":      ["tributary (0.88)", "creek (0.70)", ...],
    }

    KEY   = label (word_bestSpecific or just word for function words)
    VALUE = list of "specific (score)" strings, sorted high to low
    """
    output = {}
    for w in sentence_result["words"]:
        label  = w["label"]
        values = [
            f"{s['specific']} ({s['score']:.2f})"
            for s in w["specifics_scored"]
        ]
        if not values:
            values = ["(no specific words found)"]
        output[label] = values
    return output


# ============================================================
# SAVE JSON
# ============================================================

def save_json(all_results, output_file="scorer_output.json"):
    """
    Saves two things in one JSON file:
      1. "output" -- the clean word_specific: [scored list] format
      2. "full"   -- all internal data for debugging
    """
    clean = {}
    for r in all_results:
        clean[r["sentence"]] = format_output_dict(r)

    payload = {
        "output": clean,
        "full":   all_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nJSON saved -> {output_file}")

    # Also save the clean output separately for easy reading
    clean_file = output_file.replace(".json", "_clean.json")
    with open(clean_file, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)
    print(f"Clean JSON saved -> {clean_file}")


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(all_results, output_file="scorer_output.csv"):
    """
    One row per word per sentence.

    Columns:
      sentence_id     : sentence number
      sentence        : full original sentence
      original_word   : the token  e.g. "bank"
      label           : word_bestSpecific  e.g. "bank_savingsBank"
      best_specific   : top specific word  e.g. "savings bank"
      best_score      : score of top specific  e.g. 0.91
      chosen_sense    : the WordNet definition ConSeC chose
      used_synset     : synset name  e.g. "bank.n.01"
      all_specifics   : pipe-separated scored list
                        e.g. "savings bank(0.91) | credit union(0.80)"
    """
    headers = [
        "sentence_id",
        "sentence",
        "original_word",
        "label",
        "best_specific",
        "best_score",
        "chosen_sense",
        "used_synset",
        "all_specifics",
    ]

    rows = []
    for result in all_results:
        for w in result["words"]:
            all_sp_str = " | ".join(
                f"{s['specific']}({s['score']:.4f})"
                for s in w["specifics_scored"]
            )
            rows.append({
                "sentence_id":   result["sentence_id"],
                "sentence":      result["sentence"],
                "original_word": w["original_word"],
                "label":         w["label"],
                "best_specific": w["best_specific"] or "",
                "best_score":    w["best_score"] if w["best_score"] is not None else "",
                "chosen_sense":  w["chosen_sense"],
                "used_synset":   w["used_synset"],
                "all_specifics": all_sp_str,
            })

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV saved  -> {output_file}  ({len(rows)} rows)")


# ============================================================
# LOAD SENTENCES
# ============================================================

def load_sentences(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: '{filepath}'")
        print("Create sentences.txt with one sentence per line.")
        sys.exit(1)

    sentences = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)

    print(f"Loaded {len(sentences)} sentences from '{filepath}'")
    return sentences


# ============================================================
# PRINT PRETTY  (shows the exact output format in terminal)
# ============================================================

def print_pretty(sentence_result):
    output = format_output_dict(sentence_result)
    print(f"\n{'='*60}")
    print(f"Sentence: {sentence_result['sentence']}")
    print(f"{'='*60}")
    print("{")
    items = list(output.items())
    for i, (key, vals) in enumerate(items):
        comma = "," if i < len(items) - 1 else ""
        vals_str = json.dumps(vals, ensure_ascii=False)
        print(f'  "{key}": {vals_str}{comma}')
    print("}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    input_file = sys.argv[1] if len(sys.argv) > 1 else "sentences.txt"
    sentences  = load_sentences(input_file)

    print(f"\nScoring specific words for {len(sentences)} sentence(s)...")
    print("=" * 60)

    all_results = []

    for idx, sentence in enumerate(sentences, start=1):
        try:
            result = process_sentence(idx, sentence)
            all_results.append(result)
            print_pretty(result)          # show formatted output immediately

        except Exception as e:
            import traceback
            print(f"\nError on sentence {idx}: {e}")
            traceback.print_exc()
            all_results.append({
                "sentence_id": idx,
                "sentence":    sentence,
                "words":       [],
                "error":       str(e),
            })

    print("\n" + "=" * 60)
    print("Saving ...")
    save_json(all_results)
    save_csv(all_results)

    total_words = sum(len(r.get("words", [])) for r in all_results)
    with_specifics = sum(
        1 for r in all_results
        for w in r.get("words", [])
        if w.get("best_specific")
    )

    print(f"\nSummary:")
    print(f"  Sentences processed       : {len(all_results)}")
    print(f"  Total words processed     : {total_words}")
    print(f"  Words with specific found : {with_specifics}")
    print(f"\nDone!")
