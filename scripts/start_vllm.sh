#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#
# Tuned for the SLO: P95 end-to-end agent latency < 5s at 10+ RPS on 1x H100 80GB.
# Each agent run is 2-6 sequential LLM calls (generate -> verify -> revise loop),
# all re-sending a static system prompt + cached per-DB schema. So the wins are:
# FP8 to fit the 30B MoE with room for KV cache, prefix caching to make the
# repeated schema prefill near-free, and a capped context to maximize concurrency.

set -euo pipefail

# On-the-fly fp8 needs no calibration. For a calibrated variant of the same
# model, swap in "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8" and drop --quantization.
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --quantization fp8 \
    --kv-cache-dtype fp8 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096
