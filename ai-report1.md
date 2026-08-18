# TODO Implementation Report — `src/`

All 18 TODOs in `src/` implemented following the pseudocode already present in each comment block. Contracts verified against `tests/`. 13 network-free tests pass; the rest require downloadable models / Qdrant / OpenAI and were not exercised here.

## M1 — `m1_chunking.py`

| Function | What it does |
|----------|--------------|
| `chunk_semantic` | Splits into sentences (`re.split` on `.!?`/`\n\n`), embeds with `all-MiniLM-L6-v2`, groups adjacent sentences while cosine sim ≥ `threshold`, starts a new chunk below it. |
| `chunk_hierarchical` | Accumulates paragraphs into parents ≤ `parent_size`; each parent gets `parent_id="parent_N"`, then is split into children ≤ `child_size` via new `_split_by_size` helper. Children carry `parent_id` linking back. |
| `chunk_structure_aware` | `re.split` on markdown headers `#{1,3}`; emits one chunk per section as `header\n\ncontent` with `section` in metadata. |

Added helper `_split_by_size(text, size)` — word-preserving splitter used by hierarchical children.

**Tests:** hierarchical + structure (7) pass. Semantic + `compare_strategies` need `sentence_transformers` model download.

## M2 — `m2_search.py`

| Function | What it does |
|----------|--------------|
| `segment_vietnamese` | `underthesea.word_tokenize` → `replace("_", " ")` so BM25's space-split matches multi-word queries. Wrapped in try/except → raw text fallback. |
| `BM25Search.index` | Tokenizes each chunk via `segment_vietnamese`, builds `BM25Okapi`. |
| `BM25Search.search` | Scores query, returns top-k `SearchResult(method="bm25")`, filtering `score > 0`. |
| `DenseSearch.index` | `recreate_collection` (cosine, dim 1024), encodes texts, upserts `PointStruct`s with text in payload. |
| `DenseSearch.search` | Encodes query, `query_points` (qdrant ≥2.0 API), returns `method="dense"`. |
| `reciprocal_rank_fusion` | RRF `Σ 1/(k+rank+1)` keyed by text, sorted desc, top-k as `method="hybrid"`. |

**Tests:** RRF (2) pass. BM25 needs `rank_bm25`; Dense needs a running Qdrant.

## M3 — `m3_rerank.py`

| Function | What it does |
|----------|--------------|
| `CrossEncoderReranker._load_model` | Lazy-loads `sentence_transformers.CrossEncoder` (not FlagEmbedding — crashes on transformers ≥5.0). |
| `CrossEncoderReranker.rerank` | Predicts (query, doc) pairs, sorts desc, returns top-k `RerankResult` with rank. Handles scalar-score edge case. |
| `FlashrankReranker.rerank` | Optional lightweight path via `flashrank.Ranker`; maps results back by passage id, lazy-caches the ranker. |

**Tests:** all require the reranker model download.

## M4 — `m4_eval.py`

| Function | What it does |
|----------|--------------|
| `evaluate_ragas` | Builds HF `Dataset`, runs RAGAS 4 metrics, returns per-question `EvalResult`s + aggregate means. Whole thing in try/except → zeros on failure (no API key / wrong Python). |
| `failure_analysis` | Per result: avg of 4 metrics, `worst_metric = min`, maps to diagnosis + fix via diagnostic tree; sorts ascending, returns bottom-N. |

**Tests:** all 4 pass (RAGAS import failure path returns zeros cleanly).

## M5 — `m5_enrichment.py`

| Function | What it does |
|----------|--------------|
| `summarize_chunk` | `gpt-4o-mini` 2–3 sentence summary; extractive fallback (first 2 sentences) with no API key. |
| `generate_hypothesis_questions` | LLM generates N questions; extractive fallback turns long sentences into `?` questions. |
| `contextual_prepend` | LLM one-line locator prepended to text; fallback prepends `Trích từ {title}. `. Original text always preserved. |
| `extract_metadata` | LLM → JSON metadata (`response_format=json_object`); fallback default dict. |
| `_enrich_single_call` | Combined 1-call/chunk producing summary+questions+context+metadata as JSON. Cost path for production. |

**Tests:** all 12 pass on fallback paths (no API key needed).

## Notes / deliberate choices

- Added `response_format={"type":"json_object"}` on the two metadata/JSON LLM calls so `json.loads` doesn't choke on prose — not in the original pseudocode but required for the parse to be reliable.
- `segment_vietnamese` and `FlashrankReranker` guard imports with try/except so a missing optional dep degrades gracefully instead of crashing the whole module import.
- RAGAS row metric coercion uses `or 0.0` to handle `NaN`/`None` cells.
- Everything else is verbatim from the recipe comments — no added abstractions.
