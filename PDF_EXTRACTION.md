# Official PDF text extraction

The SEARCH/HYBRID evidence pipeline can extract text from first-party official PDF URLs
alongside the existing HTML extraction path.

## Processing flow

1. Search results are classified using the existing official-source policy.
2. Only official or company-official URLs are eligible for fetching.
3. The downloader follows redirects manually, re-checks every destination, rejects
   private/local network addresses, and enforces the existing 1.5 MB response limit.
4. PDF candidates are identified by a `.pdf` URL path or an explicit PDF MIME type
   already present in the search result.
5. `pypdf` extracts text and document metadata. The extracted text is attached to the
   same evidence result as `extracted_text`, with page markers to preserve traceability.

## Limits

- At most 3 official documents are fetched per search execution (shared HTML/PDF budget).
- At most 40 PDF pages are parsed per document.
- At most 30,000 extracted characters are retained per document.
- A PDF download is limited to 1.5 MB.
- Encrypted PDFs and PDFs with no extractable text are reported as extraction failures
  or empty results; no password guessing is attempted.
- Scanned/image-only PDFs are not OCR-processed. OCR is a separate future capability.
- PDF extraction is text extraction, not a guarantee that tables, columns, footnotes,
  mathematical notation, or reading order are perfectly reconstructed.

## Installation

Install the declared project dependencies:

```powershell
..venv\Scripts\python.exe -m pip install -r requirements.txt
```

The PDF parser is declared as `pypdf` in `requirements.txt`.

## Observability

Each result records its document type, extraction engine, final URL, retrieved time,
extracted character count, extraction status, and (for PDFs) the parsed page count and
whether the 40-page limit was reached. Execution traces report `pypdf` separately from
`trafilatura`.

## Security notes

Search result URLs are untrusted. PDF fetching uses the same public-HTTP URL validation,
manual redirect handling, final-source classification, timeout, and byte-size limit as
HTML extraction. Do not add a browser-based PDF viewer or OCR service without preserving
these constraints.
