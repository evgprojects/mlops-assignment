# Report

## vLLM serving flags for SLO: P95 end-to-end < 5s @ 10+ RPS, 1x H100 80GB

| Flag | Justification                                                                                                                           |
| --- |-----------------------------------------------------------------------------------------------------------------------------------------|
| `--quantization fp8` | Halves the 30B weights (~60GB→~30GB) so KV cache fits, and ~2x compute throughput on H100's native FP8.                                 |
| `--kv-cache-dtype fp8` | Doubles effective KV capacity → more concurrent sequences before queueing, protecting P95 under load.                                   |
| `--max-model-len 12288` | Caps context far below the 256K native so KV budget isn't wasted, maximizing concurrency (raise/lower to fit real schema sizes).        |
| `--gpu-memory-utilization 0.92` | Pushes freed memory into KV cache, where this prefill-bound workload is actually constrained.                                           |
| `--max-num-seqs 64` | Bounds concurrency high enough for ~20–60 in-flight calls (10 RPS × 2–6 calls/run) without batching inflating per-request latency.      |
| `--enable-prefix-caching` | Static system prompt + per-DB schema is re-sent every call; caching its KV makes the dominant prefill near-free, cutting TTFT.          |
| `--enable-chunked-prefill` | Interleaves a large schema prefill with other requests' decode so one big prompt can't head-of-line-block the batch (smooths the tail). |
| `--max-num-batched-tokens 4096` | Sizes the chunked-prefill budget to balance prefill throughput against decode latency.                                                  |

### Client-side (agent/graph.py)
| Change | One-line justification |
| --- | --- |
| `max_tokens` bound (512 gen/revise, 128 verify) | Caps decode so a runaway generation can't blow the per-run latency budget. |
| `guided_json` on verify | Forces parseable `{"ok","issue"}` JSON so the fail-open parser can't silently skip a needed revision. |

## Analysis

The overall pass rate has not changed between the baseline and tuned version, and remained 0.3333. The latency 
requirement was satisfied, with both baseline and tuned versions showing P95 latency around 2 seconds. RPS varied from
close to 0 and upto 40 for both versions. Average RPS on the tuned version was only slightly better than baseline, i.e.
8.39 VS 8.33. Technically, the SLO was not satisfied because RPS was less than 10. However, I hypothesise that it was 
not due to vLLM bottlenecks, but rather due to the agent server implementation and configuration. Both, baseline and 
tuned runs showed timeouts and HTTP errors. So, not requests reached vLLM, and perhaps that explains the RPS metric 
drops. Also,the performance on both versions is very similar, which also suggests that there was no vLLM bottleneck 
in the first place.

## Future work

If I had more time, I would investigate why the overall pass rate is quite low. I would also investigate timeouts, 
HTTP errors and client errors that occurred during load tests. There errors might interfere with vLLM performance 
metrics collected. 