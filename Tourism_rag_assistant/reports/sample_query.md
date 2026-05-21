# Sample query

**Question**

Что посмотреть в Ярославле, если люблю старинные храмы и набережную?

**Expected RAG behavior**

1. Retrieve documents about Ярославль.
2. Prefer churches, monasteries, historic buildings and embankment-related landmarks.
3. Rerank retrieved candidates with ColBERT.
4. Answer only from context and mention that the recommendation is limited by retrieved sources.

**Notebook observation**

The final notebook check showed stable retrieval and focused answers. The author marked the result as successful: “Ну огонь же!”
