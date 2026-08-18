# Individual Reflection — Lab 18: Production RAG

**Tên:** Bùi Minh Long
**Module phụ trách:** Toàn bộ M1–M5 + `pipeline.py`
**Tests:** 37/37 pass · **TODOs còn lại:** 0

---

## Phần 1 — Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation (số liệu thật) |
|----------------|--------|-------------|----------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Nhóm câu theo cosine `all-MiniLM-L6-v2`, threshold 0.85. Hierarchical tạo **100 children** (avg **208 char**) từ 26 docs. |
| Hierarchical (parent/child) | M1 | `chunk_hierarchical()` | Parent 2048 / child 256; child có `parent_id` hợp lệ, nhỏ hơn parent. **Cảnh báo:** pipeline chỉ dùng child, chưa expand parent. |
| Structure-aware | M1 | `chunk_structure_aware()` | Split theo header `#{1,3}`, giữ `section` trong metadata — quan trọng để không cắt giữa bảng/list. |
| BM25 (tiếng Việt) | M2 | `segment_vietnamese()` + `BM25Search` | `underthesea` nối từ ghép bằng `_`; phải `replace("_"," ")` nếu không query 2-token không khớp doc 1-token. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF `Σ 1/(k+rank+1)`, k=60. Gộp 2 ranking mà không cần chuẩn hóa score khác thang. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` (dùng `sentence_transformers.CrossEncoder`, KHÔNG FlagEmbedding). Đẩy **context_precision lên 0.933**. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Metric thấp nhất = **context_recall 0.775** vì child 256 char quá vụn. Faithfulness 0.765. |
| Failure diagnostic tree | M4 | `failure_analysis()` | Bottom-N theo avg 4 metric → map `worst_metric` sang (diagnosis, fix). 6/10 worst là faithfulness. |
| Contextual / enrichment | M5 | `_enrich_single_call()` | 1 API call/chunk (summary+questions+context+metadata). 100 chunks ≈ **$0.02** với gpt-4o-mini. |

---

## Phần 2 — Khó khăn & cách giải quyết

### KK1 — `ModuleNotFoundError` dù đã pip install
- **Exact error:** `ModuleNotFoundError: No module named 'rank_bm25' / 'sentence_transformers' / 'pypdf'`
- **Debug:** `python -c "import sys; print(sys.executable)"` → trỏ về **global** `Python312`, không phải `.venv`. Lý do: `.venv` **không có pytest**, nên gõ `pytest` rơi xuống pytest global chạy bằng Python global (thiếu dep).
- **Fix:** `.venv/Scripts/python.exe -m pip install pytest` rồi luôn chạy `.venv/Scripts/python.exe -m pytest`. → 37/37 pass.

### KK2 — Crash khi chạy `pipeline.py` trên Windows
- **Exact error:** `UnicodeEncodeError: 'charmap' codec can't encode characters ... ⚠️` (position của emoji).
- **Debug:** Console Windows mặc định cp1252, không in được emoji trong `print()`. Chỉ sống sót trong pytest vì pytest capture stdout.
- **Fix:** đặt `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` trước khi chạy. (Upgrade path: `sys.stdout.reconfigure(encoding="utf-8")` đầu file.)

### KK3 — Production KÉM hơn baseline (bất ngờ lớn nhất)
- **Hiện tượng:** faithfulness −0.094, context_recall −0.150 so với naive.
- **Debug:** so từng metric → precision cao (0.93) nhưng recall thấp → "chọn đúng chunk nhưng thiếu ngữ cảnh". Truy ra `pipeline.py` retrieve + rerank **children 256 char** rồi dùng luôn làm context, **không expand lên parent** → mất một nửa ý tưởng hierarchical.
- **Kiến thức thiếu → bổ sung:** hiểu rằng "nhiều tầng ≠ tốt hơn"; hierarchical chỉ thắng khi child→parent expansion được nối đúng. Đọc lại RAGAS: recall đo độ phủ ground-truth trong context.

---

## Phần 3 — Action Plan cho project cá nhân

## Project: Trợ lý Hỏi–Đáp nội bộ (HR / Chính sách công ty)

### Hiện tại
- RAG pipeline: chunk paragraph + dense-only search, LLM trả lời theo top-3 context.
- Known issues: trả lời sai trên câu **multi-hop** (tra bảng + tính toán) và tài liệu **nhiều version** (mật khẩu v1/v2, nghỉ phép 2023/2024) — model trộn bản cũ/mới.

### Plan áp dụng (rút từ finding của lab)
1. [x] **Chunking:** hierarchical **NHƯNG** bắt buộc child→parent expansion khi trả context (đừng lặp lại lỗi lab); dùng structure-aware cho tài liệu có bảng/list.
2. [x] **Search:** Hybrid BM25 + Dense + RRF — BM25 bắt keyword/số (mã, ngày), Dense bắt ngữ nghĩa; tiếng Việt phải segment + `replace("_"," ")`.
3. [x] **Reranking:** Có — `bge-reranker-v2-m3`; precision tăng rõ (0.93), latency chấp nhận được cho top-20→3.
4. [x] **Evaluation:** RAGAS 4 metric làm chuẩn; luôn chạy **naive baseline trước** để có mốc so sánh (đừng tin "production" mù quáng).
5. [x] **Enrichment:** combined 1-call/chunk (rẻ ~$0.02/100 chunk); thêm **version/recency metadata** để filter tài liệu xung đột.
6. [x] **Xử lý version-conflict:** metadata `version`/`effective_date` + filter ưu tiên bản mới nhất khi index.

### Timeline
- **Tuần 1:** Chuẩn hóa data + metadata version; dựng naive baseline + RAGAS harness.
- **Tuần 2:** Hierarchical (có parent-expansion) + hybrid search + rerank; đo lại RAGAS, so baseline.
- **Tuần 3:** Enrichment + version filter; tối ưu prompt/temperature cho câu multi-hop; chốt cấu hình thắng baseline ở cả 4 metric.

---

## Tự đánh giá

| Tiêu chí | Tự chấm (1–5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 5 |
