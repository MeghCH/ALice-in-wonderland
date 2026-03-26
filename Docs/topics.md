```mermaid
flowchart TD
    A["Cleaned book text
    fetch_book_pipeline(book_id)"] --> B["Structural Splitting
    re.split('CHAPTER', text)"]
    B --> C["Section Selection
    First 10 valid sections"]
    C --> D["TF-IDF Vectorization
    TfidfVectorizer(stop_words)"]
    D --> E["Feature Ranking
    Sorting matrix coefficients"]
    E --> F["Label Mapping
    get_feature_names_out()"]
    F --> G["Top 10 words per chapter
    Final topics dictionary"]

    classDef input fill:#AECF7A,stroke:#8FBA6E,color:#fff
    classDef tokenize fill:#0d9488,stroke:#0f766e,color:#fff
    classDef transform fill:#7c3aed,stroke:#6d28d9,color:#fff
    classDef score fill:#e8593c,stroke:#c2410c,color:#fff
    classDef output fill:#d97706,stroke:#b45309,color:#fff

    class A,B input
    class C tokenize
    class D,E transform
    class F score
    class G output
```

---

### Topic Modeling Pipeline (`--topics`)
This pipeline identifies the thematic essence of different sections of the book using statistical weighting.

* **Structural Segmentation**: The text is split into chapters using Regular Expressions (matching "CHAPTER" patterns). To optimize performance, the first 10 significant sections are selected.
* **Noise Reduction**: "Stop words" (common functional words like *the, a, is, of*) are filtered out as they carry no semantic weight.
* **TF-IDF Vectorization**: The system calculates the **Term Frequency-Inverse Document Frequency** score. This highlights words that are frequent within a specific chapter but unique relative to the rest of the book.
* **Feature Ranking**: Within each section, terms are ranked by their TF-IDF weight in descending order.
* **Thematic Extraction**: The top 10 highest-scoring terms per chapter are extracted to represent the "Topics" of that section.
