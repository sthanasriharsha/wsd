# ============================================================
# AUTOMATIC WSD PIPELINE
# - No target word needed from user
# - Auto-detects ambiguous words using POS tags
# - Uses ConSeC for disambiguation
# - Shows word relationships and related words
# ============================================================

import torch
import hydra
import spacy
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
import nltk

from src.consec_dataset import ConsecSample, ConsecDefinition
from src.disambiguation_corpora import DisambiguationInstance
from src.pl_modules import ConsecPLModule
from src.scripts.model.predict import predict

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


# ============================================================
# STEP 0 — Load models (runs once at startup)
# ============================================================

# Load spaCy for POS tagging and word relationships
# spaCy reads a sentence and labels every word:
#   noun, verb, adjective, etc.
# Run this first: python -m spacy download en_core_web_sm
print("🔄 Loading spaCy...")
nlp = spacy.load("en_core_web_sm")

# Load ConSeC for the actual disambiguation
CHECKPOINT = "experiments/released-ckpts/consec_semcor_normal_best.ckpt"
print("🔄 Loading ConSeC model... (takes ~1 min)")
module = ConsecPLModule.load_from_checkpoint(CHECKPOINT)

# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
module.to(device)
module.freeze()
module.sense_extractor.evaluation_mode = True

tokenizer = hydra.utils.instantiate(module.hparams.tokenizer.consec_tokenizer)
print(f"✅ All models loaded! Running on: {device}\n")


# ============================================================
# STEP 1 — Tokenize + POS tag the sentence
# ============================================================

def tokenize_and_tag(sentence):
    """
    Uses spaCy to split the sentence into words and label each one.
    Returns a list of (word, POS_tag, dependency_relation) tuples.

    Example output for "The fisherman sat on the river bank":
      [("The",        "DET",   "det"),
       ("fisherman",  "NOUN",  "nsubj"),
       ("sat",        "VERB",  "ROOT"),
       ("on",         "ADP",   "prep"),
       ("the",        "DET",   "det"),
       ("river",      "NOUN",  "compound"),
       ("bank",       "NOUN",  "pobj")]

    POS = Part of Speech (what kind of word it is)
    dep = dependency (what role it plays in the sentence)
    """
    doc = nlp(sentence)
    tagged = []
    for token in doc:
        tagged.append({
            "word":  token.text,
            "pos":   token.pos_,   # NOUN, VERB, ADJ, ADV...
            "tag":   token.tag_,   # more specific: NN, VBZ, JJ...
            "dep":   token.dep_,   # nsubj, dobj, pobj...
            "lemma": token.lemma_, # base form: "sitting" → "sit"
            "idx":   token.i       # position in sentence
        })
    return doc, tagged


# ============================================================
# STEP 2 — Analyze word relationships
# ============================================================

def analyze_relationships(doc):
    """
    Uses spaCy's dependency tree to find how words relate to each other.

    In a sentence, words are not independent — they connect to each other:
      "river bank" → "river" MODIFIES "bank" (compound noun)
      "fisherman sat" → "fisherman" is the SUBJECT of "sat"
      "sat on bank" → "bank" is the OBJECT of the preposition "on"

    This function returns those relationships as readable pairs.
    """
    relationships = []
    for token in doc:
        if token.dep_ != "ROOT":  # ROOT = the main verb, skip it
            relationships.append({
                "word":     token.text,
                "relation": token.dep_,
                "governs":  token.head.text  # the word it connects to
            })
    return relationships


# ============================================================
# STEP 3 — Detect ambiguous words automatically
# ============================================================

# Map spaCy POS tags → WordNet POS tags
# WordNet uses single letters: n=noun, v=verb, a=adjective, r=adverb
SPACY_TO_WN_POS = {
    "NOUN": wn.NOUN,
    "VERB": wn.VERB,
    "ADJ":  wn.ADJ,
    "ADV":  wn.ADV,
}

def is_ambiguous(word, wn_pos):
    """
    A word is considered ambiguous if WordNet has MORE THAN ONE sense for it.
    Example:
      "bank" (noun) → 10 senses → ambiguous ✓
      "fisherman" (noun) → 1 sense → not ambiguous ✗
      "the" (DET) → not in WordNet → skip ✗
    """
    synsets = wn.synsets(word, pos=wn_pos)
    return len(synsets) > 1  # more than 1 meaning = ambiguous

def detect_ambiguous_words(tagged):
    """
    Goes through every word in the sentence.
    Keeps only: nouns, verbs, adjectives, adverbs
    that have more than one WordNet sense.
    Skips stopwords like "the", "on", "a".
    """
    ambiguous = []
    for token_info in tagged:
        pos = token_info["pos"]

        # Only check content words (not "the", "a", "on", etc.)
        if pos not in SPACY_TO_WN_POS:
            continue

        word  = token_info["lemma"]  # use base form for WordNet lookup
        wn_pos = SPACY_TO_WN_POS[pos]

        if is_ambiguous(word, wn_pos):
            num_senses = len(wn.synsets(word, pos=wn_pos))
            ambiguous.append({
                "word":       token_info["word"],   # original word
                "lemma":      word,                  # base form
                "pos":        pos,
                "wn_pos":     wn_pos,
                "position":   token_info["idx"],
                "num_senses": num_senses
            })

    return ambiguous


# ============================================================
# STEP 4 — Disambiguate using ConSeC
# ============================================================

def get_candidates(lemma, wn_pos):
    """
    Gets all WordNet definitions for a word as ConsecDefinition objects.
    These are the "answer choices" the model will score.
    """
    candidates = []
    for synset in wn.synsets(lemma, pos=wn_pos):
        candidates.append(
            ConsecDefinition(synset.definition(), lemma)
        )
    return candidates

def disambiguate_word(sentence, word_info, all_tokens):
    """
    Runs ConSeC on one ambiguous word.
    Passes the full sentence as context so the model
    understands the surrounding meaning.
    """
    tokens    = all_tokens
    position  = word_info["position"]
    lemma     = word_info["lemma"]
    wn_pos    = word_info["wn_pos"]

    candidates = get_candidates(lemma, wn_pos)
    if not candidates:
        return None

    # Build the ConSeC input sample
    # This wraps the sentence + candidates into the format ConSeC expects
    sample = ConsecSample(
        sample_id="auto-query",
        position=position,
        disambiguation_context=[
            DisambiguationInstance("d0", "s0", f"i{i}", t, None, None, None)
            for i, t in enumerate(tokens)
        ],
        candidate_definitions=candidates,
        gold_definitions=None,
        context_definitions=[],         # no manual context — sentence handles it
        in_context_sample_id2position={"auto-query": position},
        disambiguation_instance=None,
        kwargs={},
    )

    # Run the model
    _, probs = next(
        predict(module, tokenizer, [sample],
                text_encoding_strategy="simple-with-linker")
    )

    # Sort by score (highest first)
    idxs = torch.tensor(probs).argsort(descending=True)
    results = [(probs[i], candidates[i].text) for i in idxs.tolist()]

    return results


# ============================================================
# STEP 5 — Get related words from the chosen WordNet sense
# ============================================================

def get_related_words(lemma, wn_pos, chosen_definition):
    """
    Once we know the correct sense, finds related words using WordNet:
    - Synonyms  : words meaning the same thing
    - Hypernyms : broader category ("river bank" → "slope", "land")
    - Hyponyms  : more specific kinds (if any)

    This answers your requirement: "suggest related words based on context"
    — because the related words come from the CORRECT sense only,
    not from unrelated meanings.
    """
    # Find the synset whose definition matches what ConSeC picked
    for synset in wn.synsets(lemma, pos=wn_pos):
        if synset.definition() == chosen_definition:

            # Synonyms = other words in the same synset
            synonyms = [l.name().replace("_", " ")
                        for l in synset.lemmas()
                        if l.name().lower() != lemma.lower()]

            # Hypernyms = the broader concept above this word
            hypernyms = [h.lemmas()[0].name().replace("_", " ")
                         for h in synset.hypernyms()]

            # Hyponyms = more specific versions of this concept
            hyponyms  = [h.lemmas()[0].name().replace("_", " ")
                         for h in synset.hyponyms()[:4]]  # limit to 4

            return {
                "definition": synset.definition(),
                "synonyms":   synonyms,
                "hypernyms":  hypernyms,
                "hyponyms":   hyponyms,
                "examples":   synset.examples()
            }
    return None


# ============================================================
# MAIN — Tie everything together
# ============================================================

def analyze_sentence(sentence):
    """
    Full automatic pipeline:
    1. Tokenize + POS tag
    2. Find word relationships
    3. Detect ambiguous words
    4. Disambiguate each one with ConSeC
    5. Show correct sense + related words
    """

    print(f"\n{'='*60}")
    print(f"Sentence: {sentence}")
    print(f"{'='*60}")

    # --- Step 1: Tokenize and tag ---
    doc, tagged = tokenize_and_tag(sentence)
    tokens = [t["word"] for t in tagged]

    print("\n📝 Tokens and POS tags:")
    for t in tagged:
        print(f"   {t['word']:<15} {t['pos']:<8} ({t['dep']})")

    # --- Step 2: Word relationships ---
    relationships = analyze_relationships(doc)
    print("\n🔗 Word relationships:")
    for r in relationships:
        print(f"   '{r['word']}' --[{r['relation']}]--> '{r['governs']}'")

    # --- Step 3: Find ambiguous words ---
    ambiguous_words = detect_ambiguous_words(tagged)

    if not ambiguous_words:
        print("\n✅ No ambiguous words found in this sentence.")
        return

    print(f"\n⚠️  Ambiguous words detected ({len(ambiguous_words)} found):")
    for a in ambiguous_words:
        print(f"   '{a['word']}' — {a['num_senses']} possible senses in WordNet")

    # --- Step 4 + 5: Disambiguate each word ---
    print(f"\n🧠 Running ConSeC disambiguation...\n")

    for word_info in ambiguous_words:
        word = word_info["word"]
        print(f"  ── {word.upper()} ──")

        results = disambiguate_word(sentence, word_info, tokens)

        if not results:
            print(f"   ❌ Could not disambiguate '{word}'\n")
            continue

        best_score, best_def = results[0]

        print(f"   ✅ Chosen sense : {best_def}")
        print(f"   Confidence     : {best_score:.4f}")

        # Show top 3 alternatives
        print(f"   All senses considered:")
        for score, defn in results[:4]:
            marker = "👉" if defn == best_def else "  "
            print(f"      {marker} {score:.4f}  {defn}")

        # Related words from the correct sense
        related = get_related_words(
            word_info["lemma"], word_info["wn_pos"], best_def
        )
        if related:
            if related["synonyms"]:
                print(f"   Synonyms  : {', '.join(related['synonyms'])}")
            if related["hypernyms"]:
                print(f"   Broader   : {', '.join(related['hypernyms'])}")
            if related["hyponyms"]:
                print(f"   Specific  : {', '.join(related['hyponyms'])}")
            if related["examples"]:
                print(f"   Example   : \"{related['examples'][0]}\"")

        print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    # First install spaCy model if not done yet:
    # python -m spacy download en_core_web_sm

    print("🧠 Automatic Word Sense Disambiguation System")
    print("No target word needed — system detects ambiguity automatically\n")

    while True:
        sentence = input("Enter a sentence (or 'exit'): ").strip()
        if sentence.lower() == "exit":
            print("👋 Bye!")
            break
        if sentence:
            analyze_sentence(sentence)