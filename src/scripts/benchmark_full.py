from pathlib import Path


from euroeval import Benchmarker


def main(
    model: str,
    dataset: str,
    output_file: str = "benchmark_results.jsonl",
    cache_dir: str | None = None,
):
    pfout = Path(output_file)
    pfcache = pfout.parent / "cache" if cache_dir is None else Path(cache_dir)
    pfcache.mkdir(parents=True, exist_ok=True)

    benchmarker = Benchmarker(cache_dir=str(pfcache))
    for result in benchmarker.benchmark(
        model=model,
        dataset=dataset,
        save_results=False,
    ):
        result.append_to_results(pfout)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark pretrained language models on language tasks.")
    parser.add_argument("--model", type=str, required=True, help="The model to benchmark, e.g., 'Qwen/Qwen3.5-0.8B'.")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset to benchmark on, e.g., 'mmlu-nl-full'.")
    parser.add_argument("-o", "--output-file", type=str, default="benchmark_results.jsonl", help="The file to save the benchmark results to.")
    parser.add_argument("--cache-dir", type=str, help="The directory to cache the benchmark results in.")
    args = parser.parse_args()
    main(
        model=args.model,
        dataset=args.dataset,
        output_file=args.output_file,
        cache_dir=args.cache_dir,
    )