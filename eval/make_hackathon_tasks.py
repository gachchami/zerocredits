import argparse
import json
from pathlib import Path

from task_schema import BenchmarkTask


def load_benchmark(path: str) -> list[BenchmarkTask]:
    records: list[str] = []
    with open(path, "r", encoding="utf-8") as benchmark_file:
        records = json.load(benchmark_file)

    return [BenchmarkTask.from_dict(record) for record in records]


def write_hackathon_tasks(
    benchmark_tasks: list[BenchmarkTask], hackathon_tasks_path: str
):
    hackathon_tasks_path_obj = Path(hackathon_tasks_path)
    hackathon_tasks_path_obj.parent.mkdir(parents=True, exist_ok=True)

    hackathon_tasks = [task.to_hackathon_task() for task in benchmark_tasks]

    with open(hackathon_tasks_path_obj, "w", encoding="utf-8") as hackathon_tasks_file:
        json.dump(hackathon_tasks, hackathon_tasks_file, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-path", default="eval/local_benchmark.json")
    parser.add_argument(
        "--hackathon-tasks-path", default="eval/local_hackathon_tasks.json"
    )
    args = parser.parse_args()
    benchmark_tasks = load_benchmark(path=args.benchmark_path)
    write_hackathon_tasks(
        benchmark_tasks=benchmark_tasks, hackathon_tasks_path=args.hackathon_tasks_path
    )

    print(
        f"Wrote {len(benchmark_tasks)} hackathon tasks to {args.hackathon_tasks_path}"
    )


if __name__ == "__main__":
    main()
