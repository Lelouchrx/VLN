"""
probe_vllm_speed.py — sample vLLM /metrics over a window and print the per-REQUEST
time breakdown (prefill / decode / pure-inference / queue / TTFT).

These are SERVER-side counters: prefill/decode/inference EXCLUDE queue wait.
  pure inference = prefill + decode        (no queue)
  TTFT           = queue + prefill          (time to first token)
Caveat: per-request prefill/decode still vary with batching (concurrency); for a
fully concurrency-free number probe while running a single-request (CHUNKS=1) eval.

Usage:
  python vln/probe_vllm_speed.py --base http://127.0.0.1:8001 --seconds 30
"""
import argparse, re, time, urllib.request

KEYS = {
    "request_prefill_time_seconds": "prefill",
    "request_decode_time_seconds": "decode",
    "request_inference_time_seconds": "inference (prefill+decode, excl queue)",
    "request_queue_time_seconds": "queue",
    "time_to_first_token_seconds": "TTFT (queue+prefill)",
}


def snap(base):
    txt = urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=5).read().decode()
    out = {}
    for k in KEYS:
        s = re.search(rf"^vllm:{k}_sum\S* ([0-9.eE+-]+)", txt, re.M)
        c = re.search(rf"^vllm:{k}_count\S* ([0-9.eE+-]+)", txt, re.M)
        if s and c:
            out[k] = (float(s.group(1)), float(c.group(1)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8001", help="vLLM host (no /v1)")
    p.add_argument("--seconds", type=int, default=30)
    a = p.parse_args()
    a0 = snap(a.base); time.sleep(a.seconds); a1 = snap(a.base)
    n = int(a1["request_inference_time_seconds"][1] - a0["request_inference_time_seconds"][1])
    print(f"window={a.seconds}s  requests_completed={n}\n")
    print(f"{'stage':<40}{'s/req':>10}")
    print("-" * 50)
    for k, label in KEYS.items():
        if k in a0 and k in a1:
            ds = a1[k][0] - a0[k][0]; dc = a1[k][1] - a0[k][1]
            if dc > 0:
                print(f"{label:<40}{ds/dc:>10.4f}")


if __name__ == "__main__":
    main()
