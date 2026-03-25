import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import requests
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Imports pour Sumy
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# --- Configuration & Auto-installation ---
STORAGE_DIR = Path("./storage")
STORAGE_DIR.mkdir(exist_ok=True)


def ensure_models():
    """Vérifie et télécharge les modèles manquants."""
    if not spacy.util.is_package("en_core_web_sm"):
        print("Téléchargement spaCy (12MB)...", file=sys.stderr)
        os.system("python -m spacy download en_core_web_sm")

    import nltk

    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            print(f"Téléchargement NLTK : {resource}...", file=sys.stderr)
            nltk.download(resource, quiet=True)


ensure_models()
NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])

# Liste des livres demandée par le sujet [cite: 62]
BOOKS = {
    11: "Alice's Adventures in Wonderland",
    12: "Through the Looking-Glass",
    16: "Peter Pan",
    55: "The Wonderful Wizard of Oz",
    113: "The Secret Garden",
    120: "Treasure Island",
    236: "The Jungle Book",
    108: "The Return of Sherlock Holmes",
    834: "The Memoirs of Sherlock Holmes",
    863: "The Mysterious Affair at Styles",
    1661: "The Adventures of Sherlock Holmes",
    61262: "Poirot Investigates",
    69087: "The Murder of Roger Ackroyd",
    70114: "The Big Four",
    35: "The Time Machine",
    36: "The War of the Worlds",
    84: "Frankenstein",
    159: "The Island of Doctor Moreau",
    164: "Twenty Thousand Leagues under the Sea",
    345: "Dracula",
    68283: "The Call of Cthulhu",
}

# --- Fonctions de traitement ---


def clean_gutenberg_text(text):
    start_pattern = r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    end_pattern = r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    parts = re.split(start_pattern, text, flags=re.IGNORECASE)
    text = parts[-1]
    text = re.split(end_pattern, text, flags=re.IGNORECASE)[0]
    return text.strip()


def get_book_text(book_id):
    local_path = STORAGE_DIR / f"{book_id}.txt"
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")

    urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
    ]
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                text = clean_gutenberg_text(resp.text)
                local_path.write_text(text, encoding="utf-8")
                return text
        except:
            continue
    return None


def get_lexdiv(text):
    """Analyse lexicale selon la nomenclature Epitech [cite: 21-35]."""
    doc = NLP(text[:100000])
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    if not tokens:
        return {}
    counts = Counter(tokens)
    tok, typ = len(tokens), len(counts)
    return {
        "tok": tok,
        "typ": typ,
        "hap": len([w for w, c in counts.items() if c == 1]),
        "ttr": round(typ / tok, 4),
        "mwl": round(sum(len(w) for w in tokens) / tok, 2),
        "mwf": round(tok / typ, 2),
    }


def get_topics(text):
    """Extraction des 10 mots clés par section."""
    # Découpage par chapitre (cas classique Gutenberg)
    sections = re.split(r"\bCHAPTER [IVXLCDM\d]+\b", text, flags=re.IGNORECASE)
    sections = [s.strip() for s in sections if len(s.strip()) > 500]

    if not sections:  # Fallback si pas de chapitres détectés
        words = text.split()
        sections = [" ".join(words[i : i + 5000]) for i in range(0, len(words), 5000)]

    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(sections)
    feature_names = vectorizer.get_feature_names_out()

    result = {}
    for i, row in enumerate(tfidf_matrix):
        top_indices = row.toarray().flatten().argsort()[-10:][::-1]
        result[i + 1] = [feature_names[idx] for idx in top_indices]
    return result


def get_entities(text):
    """Extraction personnages et lieux[cite: 46]."""
    doc = NLP(text[:150000])
    chars = Counter([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    locs = Counter([ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]])
    return {
        "characters": [name for name, count in chars.most_common(15)],
        "locations": [name for name, count in locs.most_common(10)],
    }


def get_summary(text):
    """Résumé via Sumy LSA[cite: 54]."""
    parser = PlaintextParser.from_string(" ".join(text.split()), Tokenizer("english"))
    summarizer = LsaSummarizer()
    return " ".join([str(s) for s in summarizer(parser.document, 5)])


def get_similar(book_id):
    """Recommandation (Top 5)[cite: 58]."""
    ids = list(BOOKS.keys())
    texts = [get_book_text(bid)[:30000] or "" for bid in ids]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    idx = ids.index(book_id)
    sims = cosine_similarity(matrix[idx], matrix).flatten()
    related = sims.argsort()[::-1]
    return [BOOKS[ids[i]] for i in related if ids[i] != book_id][:5]


def generate_card(book_id):
    """Génère la fiche complète demandée [cite: 69-82]."""
    text = get_book_text(book_id)
    if not text:
        return {"error": "Inconnu"}
    return {
        "info": {"id": str(book_id), "title": BOOKS.get(book_id)},
        "lexdiv": get_lexdiv(text),
        "topics": get_topics(text),
        "entities": get_entities(text),
        "summary": get_summary(text),
        "similar": get_similar(book_id),
    }


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    for opt in ["lexdiv", "topics", "entities", "summarize", "similar", "card"]:
        group.add_argument(f"--{opt}", type=int)

    args = parser.parse_args()
    action = [k for k, v in vars(args).items() if v is not None][0]
    bid = getattr(args, action)

    txt = get_book_text(bid)
    if not txt:
        sys.exit(1)

    mapping = {
        "lexdiv": lambda t: get_lexdiv(t),
        "topics": lambda t: get_topics(t),
        "entities": lambda t: get_entities(t),
        "summarize": lambda t: get_summary(t),
        "similar": lambda _: get_similar(bid),
        "card": lambda _: generate_card(bid),
    }

    result = mapping[action](txt)
    print(json.dumps(result, indent=2) if isinstance(result, (dict, list)) else result)
