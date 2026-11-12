# Extracted content quality signals

The evidence pipeline attaches a deterministic `content_quality` object to each
attempted official HTML/PDF extraction.

## Signals

- `status`: `usable`, `thin`, `low_signal`, `encoding_warning`, `no_text`,
  or `not_assessed`.
- `score`: heuristic 0–1 signal combining length, token diversity, and alphanumeric
  character ratio. It is not a probability.
- `char_count` and `token_count`: approximate extracted content size.
- `unique_token_ratio`: distinct normalized tokens divided by token count.
- `alphanumeric_ratio`: alphanumeric characters divided by normalized text length.
- `repeated_line_ratio`: repeated non-empty lines divided by line count.
- `replacement_char_ratio`: Unicode replacement characters divided by raw text length.

The score is a diagnostic heuristic, not a calibrated confidence measure. It does not
verify claims, determine source trust, or prove that a document is relevant to the
question.

## How ranking uses it

- Extracted title and a bounded prefix of extracted body text contribute to lexical
  relevance scoring.
- A small ranking bonus is applied to `usable` content.
- A small ranking penalty is applied to `low_signal` content.
- Thin, empty, failed, or unassessed extraction does not automatically discard a
  search result; its title/snippet can still be useful evidence.

## Important limitations

- The heuristic is language-agnostic but not linguistically sophisticated.
- Tokenization is approximate, especially for Korean morphology and tables.
- A long document can still be irrelevant or factually wrong.
- A low score does not prove a document is unusable.
- A high score does not mean its claims are true.
- PDF image-only/scanned pages may be reported as `no_text`; OCR is not included.
