from pathlib import Path


from euroeval import Benchmarker


def main(
    model: str,
    dataset: str,
    output_file: str = "benchmark_results.jsonl",
):
    pfout = Path(output_file)
    benchmarker = Benchmarker()
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
    args = parser.parse_args()
    main(
        model=args.model,
        dataset=args.dataset,
        output_file=args.output_file,
    )