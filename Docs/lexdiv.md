 ``` mermaid
    flowchart TD
    A["Raw book text
    get_book_text(book_id)"] --> B["Sanitization
    re.split(Gutenberg markers)"]
    B --> C["SpaCy Tokenization
    NLP.tokenizer(text)"]
    C --> D["Alpha Filtering
    [t for t in doc if t.is_alpha]"]
    D --> E["Vocabulary Mapping
    Counter(tokens.lower())"]
    E --> F["Metric Calculation
    TTR, MWL, MWF formulas"]
    F --> G["JSON Statistics
    tok, typ, hap, ttr, mwl, mwf"]

    classDef input fill:#AECF7A,stroke:#8FBA6E,color:#fff
    classDef tokenize fill:#0d9488,stroke:#0f766e,color:#fff
    classDef transform fill:#7c3aed,stroke:#6d28d9,color:#fff
    classDef score fill:#e8593c,stroke:#c2410c,color:#fff
    classDef output fill:#d97706,stroke:#b45309,color:#fff

    class A,B input
    class C,D tokenize
    class E transform
    class F score
    class G output
``` 
### Lexical Diversity Pipeline (`--lexdiv`)

This pipeline converts a raw text stream into a set of mathematical indices that quantify linguistic richness.

* **Extraction & Sanitization**: The raw Project Gutenberg file is parsed to strip out the legal headers and footers, leaving only the author's original prose.
* **Normalization**: The text is converted to lowercase to ensure that words like "Alice" and "alice" are treated as the same token.
* **Selective Tokenization**: Using the `spacy` tokenizer, the text is broken into individual words. A filter is applied (`is_alpha`) to discard punctuation, symbols, and numbers, focusing strictly on vocabulary.
* **Frequency Mapping**: A `Counter` object aggregates the occurrences of each unique word (**Type**) against the total word count (**Token**).
* **Metric Calculation**: The engine calculates the complexity of the writing style using the following metrics:
    * **MWL** (Mean Word Length): Average number of characters per word.
    * **Hapax Legomena**: Identification of words appearing only once in the entire text.