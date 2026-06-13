"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness.

    Execution accuracy: run the gold SQL once to get the reference row set,
    then run *each* SQL the agent emitted (the generate_sql attempt plus any
    revise attempts, pulled from the returned history) and compare canonicalized
    rows. The list of per-iteration verdicts is what summarize() rolls up into
    the per-iteration pass curve.
    """
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]
    q_text = question["question"]

    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    # Call the agent over HTTP. The graph fires up to MAX_ITERATIONS LLM-backed
    # nodes, so give it a generous timeout.
    agent_error: str | None = None
    data: dict = {}
    try:
        resp = httpx.post(agent_url, json={"question": q_text, "db": db_id}, timeout=180.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        agent_error = f"{type(e).__name__}: {e}"

    # Each generate_sql / revise step in the history produced one candidate SQL.
    # Their order is the iteration order: index 0 = generate, 1 = first revise, ...
    history = data.get("history", [])
    candidate_sqls = [h["sql"] for h in history if h.get("node") in ("generate_sql", "revise")]

    per_iteration: list[dict] = []
    for i, sql in enumerate(candidate_sqls):
        exec_ok, rows, exec_err = run_sql(db_id, sql)
        per_iteration.append({
            "iteration": i,
            "sql": sql,
            "exec_ok": exec_ok,
            "exec_error": exec_err,
            "correct": matches(gold_rows, rows),
        })

    # Final served answer is whatever SQL the agent returned at termination.
    final_sql = data.get("sql", "")
    final_correct = per_iteration[-1]["correct"] if per_iteration else False

    return {
        "db_id": db_id,
        "question": q_text,
        "gold_sql": gold_sql,
        "gold_exec_ok": gold_ok,
        "gold_error": gold_err,
        "agent_error": agent_error,
        "agent_ok": data.get("ok"),
        "iterations": data.get("iterations", len(candidate_sqls)),
        "n_candidates": len(candidate_sqls),
        "final_sql": final_sql,
        "final_correct": final_correct,
        "per_iteration": per_iteration,
    }


def _pass_rate_at(results: list[dict], k: int) -> tuple[int, int]:
    """Correct count and denominator at iteration k, with carry-forward.

    For a question that emitted fewer than k+1 candidates (verify accepted
    early, or it hit the cap), the iteration-k verdict is its last emitted
    verdict - that's what would have been served had we polled at iteration k.
    Questions that emitted nothing (agent error) count as incorrect.
    """
    correct = 0
    for r in results:
        pis = r.get("per_iteration", [])
        if not pis:
            continue  # no candidate ever produced -> incorrect, denominator unchanged
        idx = min(k, len(pis) - 1)
        if pis[idx]["correct"]:
            correct += 1
    return correct, len(results)


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {"n": 0}

    overall_correct = sum(1 for r in results if r.get("final_correct"))
    agent_errors = sum(1 for r in results if r.get("agent_error"))

    max_iters = max((r.get("n_candidates", 0) for r in results), default=0)
    per_iteration_pass = []
    for k in range(max_iters):
        correct, denom = _pass_rate_at(results, k)
        per_iteration_pass.append({
            "iteration": k,
            "n_correct": correct,
            "n": denom,
            "pass_rate": round(correct / denom, 4),
        })

    # Distribution of how many candidate SQLs each question went through.
    iteration_counts: dict[int, int] = {}
    for r in results:
        c = r.get("n_candidates", 0)
        iteration_counts[c] = iteration_counts.get(c, 0) + 1

    return {
        "n": n,
        "overall_pass_rate": round(overall_correct / n, 4),
        "overall_correct": overall_correct,
        "agent_errors": agent_errors,
        "per_iteration_pass_rate": per_iteration_pass,
        "candidate_count_distribution": {str(k): v for k, v in sorted(iteration_counts.items())},
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
