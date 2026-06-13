# Report

## vLLM serving flags (SLO: P95 end-to-end < 5s @ 10+ RPS, 1x H100 80GB)

| Flag | One-line justification |
| --- | --- |
| `--quantization fp8` | Halves the 30B weights (~60GB→~30GB) so KV cache fits, and ~2x compute throughput on H100's native FP8. |
| `--kv-cache-dtype fp8` | Doubles effective KV capacity → more concurrent sequences before queueing, protecting P95 under load. |
| `--max-model-len 12288` | Caps context far below the 256K native so KV budget isn't wasted, maximizing concurrency (raise/lower to fit real schema sizes). |
| `--gpu-memory-utilization 0.92` | Pushes freed memory into KV cache, where this prefill-bound workload is actually constrained. |
| `--max-num-seqs 64` | Bounds concurrency high enough for ~20–60 in-flight calls (10 RPS × 2–6 calls/run) without batching inflating per-request latency. |
| `--enable-prefix-caching` | Static system prompt + per-DB schema is re-sent every call; caching its KV makes the dominant prefill near-free, cutting TTFT. |
| `--enable-chunked-prefill` | Interleaves a large schema prefill with other requests' decode so one big prompt can't head-of-line-block the batch (smooths the tail). |
| `--max-num-batched-tokens 4096` | Sizes the chunked-prefill budget to balance prefill throughput against decode latency. |

### Client-side (agent/graph.py)
| Change | One-line justification |
| --- | --- |
| `max_tokens` bound (512 gen/revise, 128 verify) | Caps decode so a runaway generation can't blow the per-run latency budget. |
| `guided_json` on verify | Forces parseable `{"ok","issue"}` JSON so the fail-open parser can't silently skip a needed revision. |
