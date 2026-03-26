import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import spacy
from requests.exceptions import RequestException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Imports pour Sumy
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

# --- Configuration & Auto-installation ---
STORAGE_DIR = Path("./storage")
CACHE_DIR = Path("./cache")
STORAGE_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)


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


def fetch_and_save_text(urls, local_path):
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()

            text = clean_gutenberg_text(resp.text)

            local_path.write_text(text, encoding="utf-8")

            return text

        except RequestException as e:
            print(f"Erreur lors de l'accès à {url} : {e}")
            continue

    return None


def get_book_text(book_id):
    """Récupère le texte d'un livre depuis le cache ou le télécharge."""
    storage_file = STORAGE_DIR / f"{book_id}.txt"
    if storage_file.exists():
        return storage_file.read_text()

    # URL de base pour les livres Gutenberg
    base_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    text = fetch_and_save_text([base_url], storage_file)
    return text


def get_lexdiv(text, book_id, use_cache=True):
    """Analyse lexicale selon la nomenclature Epitech [cite: 21-35]."""
    cache_file = CACHE_DIR / f"{book_id}_lexdiv.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    doc = NLP(text[:100000])
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    if not tokens:
        return {}
    counts = Counter(tokens)
    tok, typ = len(tokens), len(counts)
    result = {
        "tok": tok,
        "typ": typ,
        "hap": len([w for w, c in counts.items() if c == 1]),
        "ttr": round(typ / tok, 4),
        "mwl": round(sum(len(w) for w in tokens) / tok, 2),
        "mwf": round(tok / typ, 2),
    }
    cache_file.write_text(json.dumps(result))
    return result


def get_topics(text, book_id, use_cache=True):
    """Extraction des 10 mots clés par section."""
    cache_file = CACHE_DIR / f"{book_id}_topics.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

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
    # Convert sparse matrix to dense array
    if hasattr(tfidf_matrix, "toarray"):
        dense_matrix = tfidf_matrix.toarray()  # type: ignore
    else:
        dense_matrix = np.array(tfidf_matrix)

    for i in range(len(sections)):
        row_array = dense_matrix[i]
        top_indices = np.argsort(row_array)[-10:][::-1]
        result[i + 1] = [feature_names[idx] for idx in top_indices]

    cache_file.write_text(json.dumps(result))
    return result


def get_entities(text, book_id, use_cache=True):
    """Extraction personnages et lieux[cite: 46]."""
    cache_file = CACHE_DIR / f"{book_id}_entities.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    doc = NLP(text[:150000])
    chars = Counter([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    locs = Counter([ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]])
    result = {
        "characters": [name for name, count in chars.most_common(15)],
        "locations": [name for name, count in locs.most_common(10)],
    }
    cache_file.write_text(json.dumps(result))
    return result


def get_summary(text, book_id, use_cache=True):
    """Résumé via Sumy LSA[cite: 54]."""
    cache_file = CACHE_DIR / f"{book_id}_summary.json"
    if use_cache and cache_file.exists():
        return cache_file.read_text()

    parser = PlaintextParser.from_string(" ".join(text.split()), Tokenizer("english"))
    summarizer = LsaSummarizer()
    result = " ".join([str(s) for s in summarizer(parser.document, 5)])
    cache_file.write_text(result)
    return result


def get_similar(book_id, use_cache=True):
    """Recommandation (Top 5)[cite: 58]."""
    cache_file = CACHE_DIR / f"{book_id}_similar.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text())

    ids = list(BOOKS.keys())
    if book_id not in ids:
        return []
    texts = [get_book_text(bid) or "" for bid in ids]
    texts = [text[:30000] if text else "" for text in texts]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    idx = ids.index(book_id)
    sims = cosine_similarity(matrix[idx : idx + 1], matrix).flatten()
    related = sims.argsort()[::-1]
    result = [BOOKS[ids[i]] for i in related if ids[i] != book_id][:5]
    cache_file.write_text(json.dumps(result))
    return result


def generate_card(book_id):
    """Génère la fiche complète demandée [cite: 69-82]."""
    text = get_book_text(book_id)
    if not text:
        return {"error": "Inconnu"}
    return {
        "info": {"id": str(book_id), "title": BOOKS.get(book_id)},
        "lexdiv": get_lexdiv(text, book_id),
        "topics": get_topics(text, book_id),
        "entities": get_entities(text, book_id),
        "summary": get_summary(text, book_id),
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
        "lexdiv": lambda t: get_lexdiv(t, bid),
        "topics": lambda t: get_topics(t, bid),
        "entities": lambda t: get_entities(t, bid),
        "summarize": lambda t: get_summary(t, bid),
        "similar": lambda _: get_similar(bid),
        "card": lambda _: generate_card(bid),
    }

    result = mapping[action](txt)
    print(json.dumps(result, indent=2) if isinstance(result, (dict, list)) else result)
