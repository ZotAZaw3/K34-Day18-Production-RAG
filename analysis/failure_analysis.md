# Failure Analysis — Lab 18: Production RAG

**Nhóm:** [điền tên]
**Thành viên:** [điền] · Implement toàn bộ M1–M5 + pipeline

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8583 | 0.7646 | **−0.094** |
| Answer Relevancy | 0.7220 | 0.7281 | +0.006 |
| Context Precision | 0.9250 | 0.9333 | +0.008 |
| Context Recall | 0.9250 | 0.7750 | **−0.150** |

> Naive = `chunk_basic` (paragraph ~500 char) + dense-only, top-3.
> Production = hierarchical children (256 char) + enrichment + BM25/Dense/RRF + cross-encoder rerank top-3.

### Finding chính (phản trực giác)

**Production KÉM HƠN baseline ở faithfulness (−0.094) và context_recall (−0.150).**
Precision cực cao (0.93) → reranker chọn đúng chunk, nhưng recall tụt → **chunk đúng nhưng KHÔNG đủ ngữ cảnh**.

Nguyên nhân gốc chung cho cả 5 failure:

1. **Hierarchical children dùng làm context cuối mà KHÔNG expand lên parent.**
   `pipeline.py` index + rerank + trả về *children 256 char*, không bao giờ dùng parent 2048. Ý tưởng hierarchical ("retrieve child → return parent") bị mất một nửa → context vụn, câu trả lời thiếu dữ kiện → faithfulness/recall giảm. Baseline chunk 500 char giữ nguyên đoạn nên recall cao hơn.
2. **Knowledge base có document xung đột version** (`mat_khau_v1` 90 ngày vs `v2` 120 ngày; `nghi_phep_nam_v2023` vs `v2024`). Chunk nhỏ retrieve cả bản cũ lẫn mới → LLM trộn/chọn sai → hallucinate.
3. **Enrichment (M5) prepend câu context do LLM sinh** vào `enriched_text` → thêm noise nhẹ, và metadata gốc bị `auto_metadata` ghi đè (mất `parent_id` link).

---

## Bottom-5 Failures

### #1 — score 0.396 · worst: **faithfulness**
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** v2.0 hiện hành = **120 ngày** (v1 90 ngày đã bị thay thế).
- **Got:** Trộn/chọn sai giữa 90 và 120 ngày.
- **Error Tree:** Output sai → Context đúng? **KHÔNG** (retrieve cả chunk v1 lẫn v2) → Query OK? Có → **fix ở retrieval**.
- **Root cause:** Document xung đột version + chunk nhỏ, không có version/recency filter → context mâu thuẫn.
- **Suggested fix:** Metadata filter ưu tiên `version=latest`, hoặc dedup bản cũ khi index; giữ câu "v2.0 thay thế v1" trong cùng chunk.

### #2 — score 0.500 · worst: **answer_relevancy**
- **Question:** Nhân viên Senior 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương khoảng nào?
- **Expected:** 15 + 3 = **18 ngày**; lương Senior (P3–P4) **20–35 triệu**.
- **Got:** Chỉ trả lời một vế (ngày phép *hoặc* lương), không đủ 2 ý.
- **Error Tree:** Output thiếu → Context đủ 2 chủ đề? **KHÔNG** (chỉ 1 trong 2 bảng được retrieve) → Query OK → **fix ở recall + prompt**.
- **Root cause:** Câu hỏi multi-part cần 2 nguồn (nghỉ phép + bảng lương); children 256 char chỉ kéo 1 chủ đề.
- **Suggested fix:** Parent-expansion để lấy trọn 2 section; prompt yêu cầu trả lời đủ mọi sub-question.

### #3 — score 0.500 · worst: **faithfulness**
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior max 20.000.000 × **85% = 17.000.000 VNĐ**.
- **Got:** Con số bịa / sai phép tính.
- **Error Tree:** Output sai → Context đủ? **KHÔNG** (quy tắc 85% và bảng lương ở 2 chunk khác nhau) → Query OK → **fix ở chunking**.
- **Root cause:** Multi-hop (tra bảng lương + áp quy tắc 85%); chunk nhỏ tách rời 2 dữ kiện → LLM tự suy số.
- **Suggested fix:** Chunk giữ quy tắc + bảng cùng nhau (structure-aware), temperature=0.

### #4 — score 0.627 · worst: **faithfulness**
- **Question:** Tạm ứng 15 triệu, 20 ngày mới thanh toán, phạt bao nhiêu?
- **Expected:** Hạn 15 ngày; quá 5 ngày, phí 2%/tháng trên 15tr = 300k/tháng → pro-rata ~50.000 VNĐ.
- **Got:** Số phạt sai (không tính đúng pro-rata / mốc 15 ngày).
- **Error Tree:** Output sai → Context đúng? Một phần → Query OK → **fix ở generation (tính toán)**.
- **Root cause:** Multi-hop tính toán trên policy bị vụn; LLM không phải máy tính.
- **Suggested fix:** Giữ policy nguyên vẹn, temperature=0, hoặc thêm tool tính toán.

### #5 — score 0.696 · worst: **faithfulness**
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** 16–30 ngày → **Giám đốc điều hành (CEO)**.
- **Got:** Cấp phê duyệt sai (kéo nhầm bậc thấp hơn).
- **Error Tree:** Output sai → Context đúng? **KHÔNG** (bảng phân tầng phê duyệt bị cắt) → Query OK → **fix ở chunking**.
- **Root cause:** Bảng ngưỡng phê duyệt bị chunk 256 char cắt rời từng dòng.
- **Suggested fix:** Structure-aware chunking giữ nguyên bảng; không cắt giữa list/table.

---

## Case Study (cho presentation)

**Question chọn phân tích:** #1 — "Bao lâu phải đổi mật khẩu một lần?" (faithfulness 0.396, thấp nhất)

**Error Tree walkthrough:**
1. Output đúng? → **Không** — trộn 90/120 ngày.
2. Context đúng? → **Không** — retrieve đồng thời `mat_khau_v1` (90) và `mat_khau_v2` (120).
3. Query rewrite OK? → **Có** — câu hỏi rõ ràng.
4. Fix ở bước: **Retrieval/Indexing** — thêm version-aware metadata filter, không phải sửa prompt.

**Bài học lớn nhất:** "Production" nhiều tầng không tự động thắng baseline. Hierarchical chunking chỉ có giá trị khi **thực sự expand child→parent**; index children rồi dùng luôn children làm context còn *tệ hơn* chunk paragraph đơn giản. Precision cao mà recall thấp = "đúng nhưng thiếu".

**Nếu có thêm 1 giờ, sẽ optimize:**
- Sửa `pipeline.py` để retrieve child → **trả về parent** (đúng thiết kế hierarchical) — kỳ vọng recall/faithfulness vượt baseline.
- Version-aware metadata filter cho document xung đột (mật khẩu, nghỉ phép).
- `temperature=0` + prompt grounding chặt cho các câu multi-hop tính toán.
