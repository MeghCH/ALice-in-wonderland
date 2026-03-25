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
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer

# --- Configuration & Auto-installation des modèles ---
STORAGE_DIR = Path("./storage")
STORAGE_DIR.mkdir(exist_ok=True)


def ensure_models():
    """Vérifie et télécharge les modèles manquants au runtime."""
    # Check spaCy
    if not spacy.util.is_package("en_core_web_sm"):
        print("Téléchargement du modèle spaCy (12MB)...", file=sys.stderr)
        os.system("python -m spacy download en_core_web_sm")

    # Check NLTK (requis par Sumy)
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


ensure_models()
NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])

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

    text = re.split(start_pattern, text, flags=re.IGNORECASE)[-1]
    text = re.split(end_pattern, text, flags=re.IGNORECASE)[0]
    return text.strip()


def get_book_text(book_id):
    local_path = STORAGE_DIR / f"{book_id}.txt"
    if local_path.exists():
        return local_path.read_text(encoding="utf-8")

    urls = [
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt",
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                text = clean_gutenberg_text(resp.text)
                local_path.write_text(text, encoding="utf-8")
                return text
        except requests.RequestException:
            continue
    return None


def get_lexdiv(text):
    # Analyse sur un échantillon pour rester ultra-léger
    doc = NLP(text[:100000])
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    if not tokens:
        return {}

    counts = Counter(tokens)
    return {
        "tokens": len(tokens),
        "unique": len(counts),
        "ttr": round(len(counts) / len(tokens), 4),
        "avg_word_len": round(sum(len(w) for w in tokens) / len(tokens), 2),
    }


def get_entities(text):
    doc = NLP(text[:150000])  # Limite pour la RAM
    chars = Counter([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
    locs = Counter([ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]])

    return {
        "characters": [name for name, count in chars.most_common(15)],
        "locations": [name for name, count in locs.most_common(10)],
    }


def get_summary(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    # On prend les 5 phrases les plus importantes
    summary = summarizer(parser.document, 5)
    return " ".join([str(s) for s in summary])


def get_similar(book_id):
    book_id = int(book_id)
    texts, ids = [], []
    for bid in BOOKS:
        t = get_book_text(bid)
        if t:
            texts.append(t[:30000])  # Petit échantillon pour TF-IDF
            ids.append(bid)

    if book_id not in ids:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    idx = ids.index(book_id)

    sims = cosine_similarity(matrix[idx], matrix).flatten()
    related = sims.argsort()[::-1]

    return [BOOKS[ids[i]] for i in related if ids[i] != book_id][:5]


def generate_card(book_id):
    text = get_book_text(book_id)
    if not text:
        return {"error": f"Livre {book_id} introuvable."}

    return {
        "info": {"id": book_id, "title": BOOKS.get(int(book_id), "Inconnu")},
        "lexical": get_lexdiv(text),
        "entities": get_entities(text),
        "summary": get_summary(text),
        "similar": get_similar(book_id),
    }


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Define actions
    parser.add_argument(
        "--summarize",
        type=int,
        dest="summarize_id",
        help="Summarize the book with the given ID",
    )
    parser.add_argument(
        "--lexical",
        type=int,
        dest="lexical_id",
        help="Perform lexical analysis on the book with the given ID",
    )
    parser.add_argument(
        "--entities",
        type=int,
        dest="entities_id",
        help="Extract entities from the book with the given ID",
    )
    parser.add_argument(
        "--similar",
        type=int,
        dest="similar_id",
        help="Find similar books to the book with the given ID",
    )
    parser.add_argument(
        "--card",
        type=int,
        dest="card_id",
        help="Generate a full analysis card for the book with the given ID",
    )

    args = parser.parse_args()

    # Determine which action to perform
    if args.summarize_id is not None:
        book_id = args.summarize_id
        action = "summarize"
    elif args.lexical_id is not None:
        book_id = args.lexical_id
        action = "lexical"
    elif args.entities_id is not None:
        book_id = args.entities_id
        action = "entities"
    elif args.similar_id is not None:
        book_id = args.similar_id
        action = "similar"
    elif args.card_id is not None:
        book_id = args.card_id
        action = "card"
    else:
        parser.print_help()
        sys.exit(1)

    text_data = get_book_text(book_id)
    if not text_data:
        print(f"Erreur : Impossible de récupérer le texte pour l'ID {book_id}")
        sys.exit(1)

    if action == "summarize":
        print(get_summary(text_data))
    elif action == "lexical":
        print(json.dumps(get_lexdiv(text_data), indent=2))
    elif action == "entities":
        print(json.dumps(get_entities(text_data), indent=2))
    elif action == "similar":
        print(get_similar(book_id))
    elif action == "card":
        print(json.dumps(generate_card(book_id), indent=2))
