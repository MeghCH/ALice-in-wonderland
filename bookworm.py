import requests
import os
import argparse
import nltk
from collections import Counter
import spacy
from gensim import corpora
from gensim.models import LdaModel
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def clean_text(text):
    start = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end = "*** END OF THE PROJECT GUTENBERG EBOOK"
    
    if start in text:
        text = text.split(start)[1]
    if end in text:
        text = text.split(end)[0]
    
    return text.strip()

# Téléchargement des livres + nettoyage du header & footer
def downclean(book_id):
    file_name = f"{book_id}.txt"
    # Si le fichier existe -> lire depuis le disque 
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return f.read()
    lien = requests.get (f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt")
    if lien.status_code == 200:
        texte = clean_text(lien.text)
        with open(file_name, "w") as f:
            f.write(texte)
        return texte
    # sinon -> télécharger -> nettoyer -> sauvegarder -> retourner
    else:
        lien_bis = requests.get (f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt")
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

def lexdiv(book_id):

    text = downclean(book_id)
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
    ttr = typ/tok
    mwl = sum(len(word) for word in tokens)/tok
    mwf = tok/typ
    return {
    "tok": tok,
    "typ": typ,
    "hap": hap,
    "ttr": ttr,
    "mwl": mwl,
    "mwf": mwf
}

def topics(book_id):
    text = downclean(book_id)
    tokens = nltk.word_tokenize(text.lower())
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    lda = LdaModel(corpus, num_topics=4, id2word=dictionary)

    result = {}
    for i in range(4):
        words = [word for word, score in lda.show_topic(i, topn=10)]
        result[i + 1] = words
    return result


def entities(book_id):
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2000000
    text = downclean(book_id)
    doc = nlp(text)
    characters = []
    locations = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            characters.append(ent.text)
        elif ent.label_ in ["GPE", "LOC"]:
            locations.append(ent.text)
    return {"characters": list(set(characters)), "locations": list(set(locations))}

def summarize(book_id):
    text = downclean(book_id)
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentences_count=5)
    return " ".join([str(sentence) for sentence in summary])

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
    68283: "The Call of Cthulhu"

}

def similar(book_id):
    texts = []
    id = []
    # Collecter les textes
    for t in BOOKS:
        text = downclean(t)
        if text:
            texts.append(text)
            id.append(t)

    # Vectoriser
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
    return result

def card(book_id):
    response = requests.get(f"https://gutendex.com/books/{book_id}")
    data = response.json()

    authors = data["authors"][0]["name"] if data["authors"] else "Unknown"
    bookshelves = ", ".join(data["bookshelves"])

    return {
        "info": {
            "id": str(book_id),
            "authors": authors,
            "bookshelves": bookshelves
        },
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

