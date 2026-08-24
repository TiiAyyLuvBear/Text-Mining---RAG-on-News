# Production RAG evaluation plan

## Scope

Evaluate retrieval relevance, source diversity, lexical answer-support diagnostics, citation completeness, contradiction signals, and abstention recommendation without requiring a ground-truth answer for every production request.

## Signals

`context_relevance` uses lexical question/context overlap as a low-cost diagnostic. It is not retrieval accuracy. `source_diversity` reports unique-article ratio on full pre-dedup rerank pool. `lexical claim-support diagnostic` splits answer into claims and checks claim/context token overlap; claims below 0.35 support score are unsupported. Citation completeness counts claims containing `[Nguồn N]`. `abstention_recommended` combines the evidence gate, relevance, and unsupported-claim rate. No calibrated overall confidence is emitted; legacy API confidence is deprecated BGE-gate state only.

## Production rollout

1. Emit evaluation object alongside existing response fields.
2. Keep full reranked pool for evidence gate; deduplicate article IDs only for prompt contexts.
3. Sample low-score, unsupported, contradictory, and user-negative requests for human review.
4. Build versioned benchmark from reviewed traffic. Use it for regression after model, prompt, index, or threshold changes.
5. Track p50/p95 latency, model load/VRAM, abstention rate, unsupported claims, citation completeness, and source quality.

## Limitations

## Operational policy

Evaluator runs fail-open: answer/API response survives evaluator errors. Inputs are capped before evaluation. Emit aggregate metrics and IDs only; do not log raw question, answer, or context by default. Apply retention, redaction, and role-based access controls to sampled review records.

Lexical overlap misses paraphrases and can reward copied but incorrect wording. It cannot prove factual correctness, detect all contradictions, or replace human/source-authority review. Health/legal/financial answers need authoritative sources and stricter review.

## Verification

```bash
../.venv/bin/python -m py_compile src/backend/evaluation.py
../.venv/bin/python -m pytest -q tests/test_evaluation.py
```

Current unit result: 37 passed.

## Pass-2 review mapping

- Prompt-source ranks and validator ranks: selected contexts are renumbered citation_rank 1..N; original rank preserved; selector/gate share `_source_key`; tested with full-pool gate and dedup.
- Citation quality: presence, validity, and support are separate; cited source must support claim with matching polarity; out-of-range and detached markers are reported.
- Polarity: positive/negative mismatch is contradiction only when lexical evidence is strong; otherwise unknown. Both directions and irrelevant negation are tested.
- WebSocket: token stream remains unchanged; aggregate structured evaluation telemetry is logged without raw text.
- Failure isolation: evaluator caps inputs and fails open; test covers exception path.
- Public telemetry: claim text is omitted; claim indexes/statuses remain. No calibrated confidence is emitted.

## Pass-3 review mapping

- Claim parser now processes lines, bullets, and punctuation-free claims; no first-line loss.
- Prefix, postfix, middle, orphan, and detached citation cases are classified; cited-source support is checked per source.
- Polarity is computed on best matching sentence windows, not whole chunks. Supporting-only, opposing-only, and mixed-source conflict states are distinct.
- Anonymous source identity uses deterministic `anonymous:index+sha1(text)` fallback. `citation_rank` is added while original rerank `rank` is preserved.
- WebSocket evaluation runs via `asyncio.to_thread`; legacy REST reports evaluator latency too.

## Pass-4 review mapping

- Standalone citation markers: postfix markers attach to prior claim; prefix markers attach to next claim; middle markers stay with containing claim; orphan markers remain unvalidated.
- Global claim status: supporting and opposing evidence always produces `conflicting`; citation support is reported separately from global status.
- Evidence matching is sentence/window-based and line-aware; unrelated chunk negation does not alter polarity.
- Anonymous source fallback is deterministic and shared by selector/gate. Original rerank rank is preserved; `citation_rank` is separate.
- Legacy evaluator failure includes `evaluation_latency_ms`; WebSocket evaluation remains off event loop.

## Pass-5 review mapping

- Shared line-aware claim/context segmentation retained; multiline no-punctuation evidence uses each line window.
- Shared namespaced source identity moved to `source_identity.py`: `article:`, `url:`, `chunk:`, deterministic `anonymous:`.
- Orphan citation now separate `citation_index_validity`; orphan error forces aggregate validity false and abstention when citations expected.
- Global claim status remains independent from cited-source status; tied opposing windows produce `conflicting`.
- Ignore rules now ignore `tests/*` and `docs/*`, unignore only two evaluation artifacts.

## Pass-6 exact verification

- `rg -n '_segments|_SENTENCE_RE|def _evidence' src/backend/evaluation.py` confirms `_evidence` calls `_segments`; direct whole-text regex absent there.
- `pytest --collect-only -q` lists `test_websocket_stream_token_protocol_and_telemetry` and `test_legacy_rag_handler_additive_schema_and_latency`.
- Exact regression tests cover multiline first/last context support and both sentence orders for same-source polarity conflict.
- Source identity namespaces: `article:`, `url:`, `chunk:`, `anonymous:`.

## Pass-7 review mapping

- `_segments` strips list prefixes before sentence splitting and is shared by answer claims and context evidence.
- Named `POLARITY_TIE_TOLERANCE = 0.15` controls near-tied polarity windows; version this constant with evaluation changes.
- Same-source opposing windows produce `conflicting`; cited conflicting source cannot count as `citation_support`.
- WebSocket test consumes all three expected token frames and confirms no metadata frame.
- FastAPI fail-open endpoint test confirms HTTP 200, abstained answer, and `evaluation.status=unavailable`.

## Pass-8 production generation fix

HF generation now uses `apply_chat_template(..., add_generation_prompt=True)` when available, extracts string/list/chat outputs, retries one empty generation, then raises stable `LLMUnavailableError` without fabricated text. REST and legacy endpoints return controlled `generation_unavailable` responses. Evaluation metadata includes `evaluation_version=lexical-v7`.

Pass-8 tests cover raw HF success, chat-template use, empty-then-success retry, persistent empty error, numeric 1..10 claim segmentation, generated-path evaluator fail-open, and controlled generation-unavailable HTTP 200.

## Pass-9 production failure handling

HF chat-template, load, runtime, CUDA/OOM, invalid output, and empty output failures normalize to stable `LLMUnavailableError` categories without prompt/secret logging. REST, legacy, and WebSocket paths return controlled output; generation-unavailable responses skip claim evaluation with `status=skipped`, `reason=generation_unavailable`, and shared `evaluation_version`.

## Pass-10 review mapping

- Rendered chat-template input carries `(text, templated=True)`; generation passes `add_special_tokens=False` only for rendered templates, preventing duplicate BOS. Plain fallback keeps default tokenization.
- One shared double-checked load lock protects encoder, reranker, and generator initialization; inference does not hold load lock.
- Nested assistant extraction concatenates all text blocks from last assistant message only.
- WebSocket generation-unavailable test consumes controlled tokens, verifies skipped telemetry, and checks raw error absence.

## Pass-11 regression

`_build_generation_prompt` remains string-only; templated tuple exists only in `_hf_input`. Smoke test covers `NewsPipeline.generate()` API/HF provider paths with string prompt. Assistant nested content concatenates first and last blocks. Skipped evaluation includes `abstention_recommended=true`, shared version, status/reason, and latency.

## Pass-12 ship cleanup

Assistant content blocks join with newline separators, preserving semantic boundaries. Legacy generation-unavailable path now has an actual `RagHandler` test: sufficient evidence, stable internal error, HTTP 200, no leak, skipped evaluation, abstention true, shared version, and latency.
