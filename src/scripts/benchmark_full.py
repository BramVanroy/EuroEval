"""Custom script so I can change output file/directory in SLURM/container context. CLI has no option for that."""
from pathlib import Path
import json

from euroeval import Benchmarker


def main(
    model: str,
    dataset: str,
    output_file: str = "benchmark_results.jsonl",
    cache_dir: str | None = None,
    force: bool = False,
):
    pfout = Path(output_file)
    pfcache = pfout.parent / "cache" if cache_dir is None else Path(cache_dir)
    pfcache.mkdir(parents=True, exist_ok=True)

    result_exists = False
    if pfout.exists():
        keep_data = []
        for line in pfout.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            result = json.loads(line)
            try:
                res_model = result["model_info"]["id"]
                res_dataset = result["evaluation_results"][0]["source_data"]["dataset_name"]
            except KeyError:
                continue
            else:
                if res_model == model and res_dataset == dataset:
                    result_exists = True
                    # Remove/overwrite existing results for the same model and dataset
                    if force:
                        continue
                    else:
                        keep_data.append(result)
                else:
                    keep_data.append(result)
        if keep_data:
            with pfout.open("w", encoding="utf-8") as fhout:
                for result in keep_data:
                    fhout.write(json.dumps(result) + "\n")

    if result_exists and not force:
        print(f"Results for model '{model}' on dataset '{dataset}' already exist in '{pfout}'. Use --force to overwrite.")
        return

    benchmarker = Benchmarker(cache_dir=str(pfcache), verbose=True, evaluate_test_split=True)
    for result in benchmarker.benchmark(
        model=model,
        dataset=dataset,
        save_results=False,
        evaluate_test_split=True,
    ):
        result.append_to_results(pfout)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark pretrained language models on language tasks.")
    parser.add_argument("--model", type=str, required=True, help="The model to benchmark, e.g., 'Qwen/Qwen3.5-0.8B'.")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset to benchmark on, e.g., 'mmlu-nl-full'.")
    parser.add_argument("-o", "--output-file", type=str, default="benchmark_results.jsonl", help="The file to save the benchmark results to.")
    parser.add_argument("--cache-dir", type=str, help="The directory to cache the benchmark results in.")
    parser.add_argument("--force", action="store_true", help="Whether to overwrite existing results for the same model and dataset.")
    args = parser.parse_args()
    main(
        model=args.model,
        dataset=args.dataset,
        output_file=args.output_file,
        cache_dir=args.cache_dir,
        force=args.force
    )