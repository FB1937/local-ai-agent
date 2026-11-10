# Optional multilingual semantic evidence retrieval

The evidence-passage selector can optionally use Ollama embeddings to find passages
whose meaning is similar to the user's question, even when the wording differs.

## Enablement

The feature is disabled by default. Install the multilingual embedding model once:

```powershell
ollama pull bge-m3
```

Then add these settings to the local `.env`:

```env
ENABLE_SEMANTIC_RETRIEVAL=1
OLLAMA_EMBEDDING_MODEL=bge-m3
OLLAMA_EMBEDDING_TIMEOUT=60
OLLAMA_EMBEDDING_MIN_SCORE=0.35
```

The existing `OLLAMA_URL` setting is reused. No Python package is added; the existing
`requests` dependency calls Ollama's `/api/embed` endpoint.

BGE-M3 is selected because its Ollama model card describes support for more than 100
languages, which makes it a more suitable candidate for Korean/English mixed sources
than an English-only embedding model. This is a model choice to evaluate, not a claim
that retrieval quality has already been validated.

## Retrieval flow

1. Split the extracted source body into sentence/line passages.
2. Send the current question and a bounded batch of up to 80 passages to Ollama.
3. Compute cosine similarity between the question vector and each passage vector.
4. Keep passages meeting the configured minimum similarity score.
5. Retain at most 4 passages and 3,200 characters per source.
6. If embedding is disabled, unavailable, invalid, or produces no passage above the
   threshold, fall back to the existing deterministic lexical selector.

The embedding request is bounded by a 2,000-character question limit, 1,200 characters
per passage, 80 candidate passages, and a configurable timeout capped at 120 seconds.

## Important limitations

- Cosine similarity is a retrieval signal, not a probability, truth score, or entailment test.
- A semantically similar passage may still fail to support the requested claim.
- The threshold is a starting heuristic and must be calibrated against labelled examples.
- At most the first 80 extracted passages are considered in this initial implementation.
- OCR, persistent embedding caches, vector databases, and cross-encoder reranking are
  intentionally out of scope.
- If the embedding model is missing, the application continues using lexical retrieval;
  install the model to use semantic retrieval.

## Ollama API

The implementation uses Ollama's batch `POST /api/embed` endpoint. See the official
[Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings)
and [BGE-M3 model page](https://ollama.com/library/bge-m3).
