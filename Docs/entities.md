```mermaid
flowchart TD
    A["Cleaned book text
    fetch_book_pipeline(book_id)"] --> B["Content Slicing
    text[:150000] for performance"]
    B --> C["SpaCy NER Model
    NLP(text) - en_core_web_sm"]
    C --> D["Label Filtering
    Filter PERSON, GPE, LOC"]
    D --> E["Frequency Counting
    Counter(entities)"]
    E --> F["Top Common Selection
    most_common(15) & (10)"]
    F --> G["Structured Entities
    characters & locations JSON"]

    classDef input fill:#AECF7A,stroke:#8FBA6E,color:#fff
    classDef tokenize fill:#0d9488,stroke:#0f766e,color:#fff
    classDef transform fill:#7c3aed,stroke:#6d28d9,color:#fff
    classDef score fill:#e8593c,stroke:#c2410c,color:#fff
    classDef output fill:#d97706,stroke:#b45309,color:#fff

    class A,B input
    class C tokenize
    class D transform
    class E,F score
    class G output
```

---

### Named Entity Recognition Pipeline (`--entities`)
This pipeline utilizes a pre-trained Deep Learning model to extract real-world concepts from the narrative.

* **Contextual Analysis**: The `en_core_web_sm` model performs a dependency parse to understand the grammatical context of every sentence.
* **NER Classification**: The model identifies "Named Entities" and assigns them categories based on linguistic patterns.
* **Label Filtering**:
    * **PERSON**: Extracted to identify characters.
    * **GPE / LOC**: Extracted to identify countries, cities, or geographic landmarks.
* **Frequency Aggregation**: Occurrences are counted to distinguish between main protagonists/settings and minor mentions.
* **Selection**: The pipeline returns the top 15 characters and top 10 locations, providing a structured overview of the book's world.
