# Pipeline of summarize

```mermaid
flowchart TD
    A["Raw book text
    get_book_text(book_id)"] --> B["Normalization
    join(text.split())"]
    B --> C["Sentence tokenization
    Sumy Tokenizer(english)"]
    C --> D["Term-sentence matrix
    TF-IDF weighting"]
    D --> E["SVD decomposition
    A = U × Σ × Vᵀ"]
    E --> F["Sentence scoring
    Using Vᵀ vectors"]
    F --> G["Top 5 sentence selection
    Final extractive summary"]
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

## Step descriptions

1. **Raw text** : Retrieved and cleaned from Project Gutenberg.
2. **Normalization** : Multiple spaces and line breaks are merged into single spaces.
3. **Tokenization** : The text is split into sentences using `Tokenizer("english")` and `PlaintextParser`.
4. **Term-sentence matrix** : Each sentence is converted into a TF-IDF vector.
5. **SVD decomposition** : The matrix is decomposed into `U × Σ × Vᵀ` to extract latent concepts.
6. **Scoring** : Each sentence receives a score based on its contribution to the main concepts.
7. **Selection** : The top 5 sentences form the extractive summary.
