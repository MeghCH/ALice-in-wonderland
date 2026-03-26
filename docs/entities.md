```mermaid
flowchart TD
    A[Cleaned text - 150k characters] --> B[SpaCy: NER analysis]
    B --> C{Filtering by label}
    C -- PERSON --> D[Character extraction]
    C -- GPE/LOC --> E [Place Extraction]
    D --> F[Counter: Occurrence Aggregation]
    E --> G[Counter: Occurrence Aggregation]
    F --> H[Selection: Top 15]
    G --> I[Selection: Top 10]
    H & I --> J[Result: JSON Entities]
```

