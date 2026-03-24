## 📖 Bookworm - NLP Engine for Project Gutenberg

**Bookworm** is a lightweight Natural Language Processing (NLP) engine designed to transform the vast library of Project Gutenberg into concise "book cards".Created for the startup *Through the Looking-Glass*, this tool helps publishers and editors quickly grasp the essence of a book without reading it in its entirety.

---

## Features

The `bookworm.py` script performs several atomic NLP tasks to transform raw text into structured understanding.

**Lexical Diversity (`--lexdiv <ID>`)**: Evaluates the richness of an author's vocabulary by calculating metrics like TTR (Type-Token Ratio), Hapax legomena, and mean word length


**Topic Modeling (`--topics <ID>`)**: Uncovers the main themes of a book by extracting the top 10 words from each section


**Named Entity Recognition (`--entities <ID>`)**: Automatically identifies characters and locations mentioned in the text to ease cross-referencing


**Book Summarization (`--summarize <ID>`)**: Condenses the book into a short, descriptive summary of a few sentences

**Book Similarity (`--similar <ID>`)**: Identifies stylistic or thematic resemblances between books using vectorization and cosine similarity to recommend 5 similar titles.

**Book Card (`--card <ID>`)**: Compiles all the above information into a single, structured JSON-like metadata sheet.

---

##  Installation & Setup

### 1. Prerequisites
* **Python 3**
* **pip** (Python install package)

### 2. Clone the Repository
```bash
git clone <repository-url>
cd bookworm
```

### 3. Install Dependencies
This project is designed to run fast without consuming heavy resources. 
```bash
pip install -r requirements.txt
```

## Usage

Run the engine using the command line interface (CLI) with a specific Gutenberg Book ID

```bash
python bookworm.py [OPTION] [ID]
```

### Examples:
* **Generate a full book card for Alice's Adventures in Wonderland:**
    ```bash
    python bookworm.py --card 11
    ```
* **Analyze the lexical diversity of Sherlock Holmes:**
    ```bash
    python bookworm.py --lexdiv 1661
    ```
* **Find books similar to Dracula:**
    ```bash
    python bookworm.py --similar 345
    ```

---

##  Supported Book Collection
The engine is pre-configured to work with a specific collection, including
* **Children/Young Adult**: *Alice's Adventures in Wonderland* (11), *Peter Pan* (16), *The Jungle Book* (236).
* **Crime & Mystery**: *The Adventures of Sherlock Holmes* (1661), *Poirot Investigates* (61262).
* **Science-Fiction & Fantasy**: *The Time Machine* (35), *Frankenstein* (84), *The Call of Cthulhu* (68283).

---

## Technical Requirements
**Persistence**: Computationally expensive operations are saved to reuse results later
**Performance**: The tool avoids heavy models to ensure it runs seamlessly on any machine
**Error Handling**: The program handles errors gracefully with clear user messages.
