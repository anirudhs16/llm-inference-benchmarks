import argparse
import requests
import time
import json
import statistics

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Inference Benchmark CLI")
    parser.add_argument("--model", type=str, default="llama3.2")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="Explain what a KV cache is in two sentences.")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama"])
    return parser.parse_args()

def run_single(model, prompt, backend="ollama"):
    """
    Send one request, measure TTFT and total time.
    Returns dict with ttft, total_time, tokens_generated, tokens_per_sec
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True
    }

    first_token_time = None
    token_count = 0
    full_response = ""

    start = time.time()

    with requests.post(url, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                if first_token_time is None and token:
                    first_token_time = time.time() - start
                full_response += token
                token_count += 1
                if data.get("done"):
                    break

    total_time = time.time() - start
    decode_time = total_time - first_token_time

    return {
        "ttft": first_token_time,
        "total_time": total_time,
        "tokens_generated": token_count,
        "tokens_per_sec": token_count / decode_time if decode_time > 0 else 0
    }

def run_benchmark(model, prompt, runs, backend="ollama"):
    """
    Run N requests and collect statistics.
    P50 = median, P95 = 95th percentile (worst case in production)
    """
    results = []
    
    print(f"\nRunning {runs} requests...")
    for i in range(runs):
        result = run_single(model, prompt, backend)
        results.append(result)
        print(f"  Run {i+1}/{runs}: TTFT={result['ttft']:.2f}s | tokens/sec={result['tokens_per_sec']:.1f}")
    
    ttfts = sorted([r["ttft"] for r in results])
    tokens_per_sec = [r["tokens_per_sec"] for r in results]
    
    def percentile(data, p):
        index = int(len(data) * p / 100)
        return data[min(index, len(data)-1)]
    
    return {
        "model": model,
        "runs": runs,
        "p50_ttft": percentile(ttfts, 50),
        "p95_ttft": percentile(ttfts, 95),
        "avg_tokens_per_sec": statistics.mean(tokens_per_sec),
        "avg_total_time": statistics.mean([r["total_time"] for r in results])
    }

def print_results(results, prompt):
    """Print formatted benchmark summary."""
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Model:           {results['model']}")
    print(f"Backend:         Ollama (local)")
    print(f"Runs:            {results['runs']}")
    print(f"Prompt:          {prompt[:50]}...")
    print("-" * 50)
    print(f"P50 TTFT:        {results['p50_ttft']:.2f}s")
    print(f"P95 TTFT:        {results['p95_ttft']:.2f}s")
    print(f"Avg tokens/sec:  {results['avg_tokens_per_sec']:.1f}")
    print(f"Avg total time:  {results['avg_total_time']:.2f}s")
    print("-" * 50)
    print(f"Cost/1K tokens:  $0.00 (local)")
    print("=" * 50)

if __name__ == "__main__":
    args = parse_args()
    results = run_benchmark(args.model, args.prompt, args.runs)
    print_results(results, args.prompt)