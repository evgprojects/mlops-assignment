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



```
(EngineCore_DP0 pid=140653) ValueError: To serve at least one request with the models's max seq len (262144), (24.00 GiB KV cache is needed, which is larger than the available KV cache memory (8.68 GiB). Based on the available memory, the estimated maximum model length is 94784. Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
```

baseline
```
{
  "n": 30,
  "overall_pass_rate": 0.3333,
  "overall_correct": 10,
  "agent_errors": 0,
  "per_iteration_pass_rate": [
    {
      "iteration": 0,
      "n_correct": 10,
      "n": 30,
      "pass_rate": 0.3333
    },
    {
      "iteration": 1,
      "n_correct": 10,
      "n": 30,
      "pass_rate": 0.3333
    },
    {
      "iteration": 2,
      "n_correct": 10,
      "n": 30,
      "pass_rate": 0.3333
    }
  ],
  "candidate_count_distribution": {
    "1": 19,
    "2": 5,
    "3": 6
  }
}

```

after tuning

```
{
  "n": 30,
  "overall_pass_rate": 0.3333,
  "overall_correct": 10,
  "agent_errors": 0,
  "per_iteration_pass_rate": [
    {
      "iteration": 0,
      "n_correct": 10,
      "n": 30,
      "pass_rate": 0.3333
    },
    {
      "iteration": 1,
      "n_correct": 11,
      "n": 30,
      "pass_rate": 0.3667
    },
    {
      "iteration": 2,
      "n_correct": 10,
      "n": 30,
      "pass_rate": 0.3333
    }
  ],
  "candidate_count_distribution": {
    "1": 22,
    "2": 1,
    "3": 7
  }
}
```

baseline
```
evgeny@computeinstance-e00m2j5x6myy7h1py7:~/mlops-assignment$ uv run python load_test/driver.py --rps 10 --duration 300
{
  "requested_rps": 10.0,
  "duration_seconds": 300,
  "wall_clock_seconds": 360.0035142329998,
  "total_requests": 3000,
  "achieved_rps": 8.333251986141319,
  "ok": 2359,
  "timeouts": 137,
  "http_errors": 402,
  "client_errors": 102,
  "latency_p50": 79.7310095519988,
  "latency_p95": 117.85794400499981,
  "latency_p99": 119.77713574300105,
  "latency_max": 120.88763312999981
}

```


after
```
evgeny@computeinstance-e00m2j5x6myy7h1py7:~/mlops-assignment$ uv run python load_test/driver.py --rps 10 --duration 300
{
  "requested_rps": 10.0,
  "duration_seconds": 300,
  "wall_clock_seconds": 357.5086048940002,
  "total_requests": 3000,
  "achieved_rps": 8.391406413530905,
  "ok": 2254,
  "timeouts": 157,
  "http_errors": 346,
  "client_errors": 243,
  "latency_p50": 72.88748577699971,
  "latency_p95": 118.42832197000007,
  "latency_p99": 120.09208480599955,
  "latency_max": 120.72303548899981
}
Wrote /home/evgeny/mlops-assignment/results/load_test.json
```