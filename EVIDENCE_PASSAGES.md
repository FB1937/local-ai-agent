# Query-relevant evidence passages

The answer-generation stage no longer places up to 8,000 characters of every extracted
page directly into the prompt. It first selects a small set of sentences with lexical
overlap against the current question.

## Selection behavior

- Tokenizes the current question and extracted text with a lightweight regular expression.
- Removes a small set of generic Korean stopwords.
- Splits extracted text into sentence/line segments.
- Scores each segment by the fraction of query tokens found in that segment.
- Keeps at most 4 passages and 3,200 characters per source.
- Preserves PDF page markers when a page marker is adjacent to extracted text.
- Includes the source ID and URL in the prompt so citations continue to refer to the
  original evidence source.

When no lexical overlap is found, no body passage is supplied. The search-result title
and snippet remain available to the model.

## Limitations

- This is lexical matching, not semantic search or entailment verification.
- Korean morphology, synonyms, paraphrases, and technical abbreviations may cause
  relevant passages to be missed.
- A selected passage can share words with a question without supporting its answer.
- Passage scores are heuristic overlap ratios, not confidence probabilities.
- The original extracted text remains available to deterministic validators; this
  component only controls the bounded excerpts sent to the answer-generation prompt.

A future semantic reranker may improve recall, but should be evaluated against labelled
question/passage pairs before replacing or overriding this deterministic baseline.
