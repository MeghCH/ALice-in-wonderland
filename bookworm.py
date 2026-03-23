import requests
import os
import argparse
import json
from collections import Counter
import re

STORAGE_DIR = "./storage"
os.makedirs(STORAGE_DIR, exist_ok=True)


def clean_text(text):
    start = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end = "*** END OF THE PROJECT GUTENBERG EBOOK"

    if start in text:
        text = text.split(start)[1]
    if end in text:
        text = text.split(end)[0]

    return text.strip()


def downclean(book_id):
    file_name = f"{book_id}.txt"
    # Si le fichier existe -> lire depuis le disque
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return f.read()
    lien = requests.get(f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt")
    if lien.status_code == 200:
        texte = clean_text(lien.text)
        with open(file_name, "w") as f:
            f.write(texte)
        return texte

    else:
        lien_bis = requests.get(
            f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"
        )
        if lien_bis.status_code == 200:
            texte = clean_text(lien_bis.text)
            with open(file_name, "w") as f:
                f.write(texte)
            return texte
        else:
            return None


# CLI
parser = argparse.ArgumentParser()
parser.add_argument("--lexdiv", type=int)
parser.add_argument("--topics", type=int)
parser.add_argument("--entities", type=int)
parser.add_argument("--summarize", type=int)
parser.add_argument("--similar", type=int)
parser.add_argument("--card", type=int)
args = parser.parse_args()


def get_storage_path(book_id, operation):
    return os.path.join(STORAGE_DIR, f"{book_id}_{operation}.json")


def load_or_compute(book_id, operation, compute_func):
    storage_path = get_storage_path(book_id, operation)
    if os.path.exists(storage_path):
        with open(storage_path, "r") as f:
            return json.load(f)
    result = compute_func(book_id)
    with open(storage_path, "w") as f:
        json.dump(result, f)
    return result


def lexdiv(book_id):
    def compute_lexdiv(book_id):
        import nltk

        nltk.download("punkt", quiet=True)

        text = downclean(book_id)
        if text is None:
            return None
        tokens = nltk.word_tokenize(text)
        tokens = [t for t in tokens if t.isalpha()]
        tokens = [t.lower() for t in tokens]
        tok = len(tokens)
        typ = len(set(tokens))
        counts = Counter(tokens)
        hap = 0
        for word in counts:
            if counts[word] == 1:
                hap += 1
        ttr = typ / tok
        mwl = sum(len(word) for word in tokens) / tok
        mwf = tok / typ

        del tokens, counts
        return {"tok": tok, "typ": typ, "hap": hap, "ttr": ttr, "mwl": mwl, "mwf": mwf}

    return load_or_compute(book_id, "lexdiv", compute_lexdiv)


def split_into_sections(text):
    import re

    sections = []
    current_section = []

    section_pattern = re.compile(
        r"^\s*"
        r"(?:CHAPTER|Chapter|CHAP\.|Chap\.|SCENE|Scene|SC\.|Sc\.|BOOK|Book|PART|Part|SECTION|Section|ACT|Act|Chapitre|Livre|Partie|Scène|Acte)"
        r"\s+[IVXLCDMivxlcdm0-9]+"  # Roman numerals or numbers
        r"|^[IVXLCDMivxlcdm]+\s*"  # Standalone Roman numerals
        r"|^\d+\s*"  # Standalone numbers
        r"|^First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth"
        r"\s+Part"
        r"\s*$"
    )

    for line in text.split("\n"):
        line = line.strip()
        if line and section_pattern.match(line):
            if current_section:
                sections.append(" ".join(current_section))
                current_section = []
        elif line:
            current_section.append(line)

    if current_section:
        sections.append(" ".join(current_section))

    return sections if sections else [text]


def topics(book_id):
    def compute_topics(book_id):
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)

        text = downclean(book_id)
        if text is None:
            return None

        sections = split_into_sections(text)
        result = {}

        for i, section in enumerate(sections):
            tokens = nltk.word_tokenize(section.lower())
            stop_words = set(nltk.corpus.stopwords.words("english"))
            tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

            if tokens:
                word_freq = Counter(tokens)
                top_words = [word for word, _ in word_freq.most_common(10)]
                result[i + 1] = top_words
                del word_freq

        return result

    return load_or_compute(book_id, "topics", compute_topics)


def similar(book_id):
    def compute_similar(book_id):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        texts = []
        id = []

        for t in BOOKS:
            text = downclean(t)
            if text:
                texts.append(text)
                id.append(t)

        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(texts)
        idx = id.index(book_id)
        scores = cosine_similarity(matrix[idx], matrix).flatten()
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        result = []
        for i in ranked:
            if id[i] != book_id:
                result.append(BOOKS[id[i]])
            if len(result) == 5:
                break

        del vectorizer, matrix, scores, ranked
        return result

    return load_or_compute(book_id, "similar", compute_similar)


def entities(book_id):
    def compute_entities(book_id):
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("maxent_ne_chunker", quiet=True)
        nltk.download("words", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)

        text = downclean(book_id)
        if text is None:
            return None

        try:
            sentences = nltk.sent_tokenize(text)
            characters = set()
            locations = set()

            for sentence in sentences:
                for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence))):
                    if hasattr(chunk, "label"):
                        if chunk.label() == "PERSON":
                            characters.update([c[0] for c in chunk.leaves()])
                        elif chunk.label() in ["GPE", "LOCATION", "FACILITY"]:
                            locations.update([c[0] for c in chunk.leaves()])

            return {
                "characters": list(characters),
                "locations": list(locations),
            }
        except:
            return {"characters": [], "locations": []}

    return load_or_compute(book_id, "entities", compute_entities)


def summarize(book_id):
    def compute_summarize(book_id):
        import nltk

        nltk.download("punkt", quiet=True)

        text = downclean(book_id)
        if text is None:
            return None

        try:
            sentences = nltk.sent_tokenize(text)

            word_freq = Counter()
            for sentence in sentences:
                words = nltk.word_tokenize(sentence.lower())
                words = [w for w in words if w.isalpha()]
                word_freq.update(words)

            important_sentences = []
            for sentence in sentences:
                score = sum(
                    word_freq[word.lower()]
                    for word in nltk.word_tokenize(sentence)
                    if word.isalpha()
                )
                important_sentences.append((score, sentence))

            important_sentences.sort(reverse=True, key=lambda x: x[0])
            summary = " ".join([s[1] for s in important_sentences[:5]])

            del word_freq, important_sentences, sentences
            return summary
        except:
            return "Summary unavailable"

    return load_or_compute(book_id, "summarize", compute_summarize)


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
    84: "Frankenstein; Or, The Modern Prometheus",
    159: "The Island of Doctor Moreau",
    164: "Twenty Thousand Leagues under the Sea",
    345: "Dracula",
    68283: "The Call of Cthulhu",
}


def card(book_id):
    response = requests.get(f"https://gutendex.com/books/{book_id}")
    if response.status_code != 200:
        return None
    data = response.json()

    authors = data["authors"][0]["name"] if data["authors"] else "Unknown"
    bookshelves = ", ".join(data["bookshelves"])

    return {
        "info": {"id": str(book_id), "authors": authors, "bookshelves": bookshelves},
        "lexdiv": lexdiv(book_id),
        "topics": topics(book_id),
        "entities": entities(book_id),
        "summary": summarize(book_id),
        "similar": similar(book_id),
    }


if args.lexdiv:
    print(lexdiv(args.lexdiv))

elif args.topics:
    print(topics(args.topics))

elif args.entities:
    print(entities(args.entities))

elif args.summarize:
    print(summarize(args.summarize))

elif args.similar:
    print(similar(args.similar))

elif args.card:
    print(card(args.card))
