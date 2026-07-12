import json
from pathlib import Path

from task_schema import BenchmarkTask

BENCHMARK_FILE = "eval/local_benchmark.json"


def make_math_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="math_001",
            category="math",
            prompt="Calculate 15% of 240. Return the final answer.",
            gold="36",
            scorer="numeric_exact",
        ),
        BenchmarkTask(
            task_id="math_002",
            category="math",
            prompt="A price increases from 80 to 100. What is the percentage increase?",
            gold="25",
            scorer="numeric_exact",
        ),
        BenchmarkTask(
            task_id="math_003",
            category="math",
            prompt="If revenue grows from 500 to 650, what is the growth percentage?",
            gold="30",
            scorer="numeric_exact",
        ),
    ]


def make_sentiment_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="sentiment_001",
            category="sentiment",
            prompt="Classify the sentiment as Positive, Negative, or Neutral and justify briefly: I loved the new laptop. It is fast and reliable.",
            gold="Positive",
            scorer="label_exact",
        ),
        BenchmarkTask(
            task_id="sentiment_002",
            category="sentiment",
            prompt="Classify the sentiment as Positive, Negative, or Neutral and justify briefly: The app crashes every time I open it.",
            gold="Negative",
            scorer="label_exact",
        ),
        BenchmarkTask(
            task_id="sentiment_003",
            category="sentiment",
            prompt="Classify the sentiment as Positive, Negative, or Neutral and justify briefly: The package arrived yesterday.",
            gold="Neutral",
            scorer="label_exact",
        ),
    ]


def make_ner_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="ner_001",
            category="ner",
            prompt="Extract named entities from this text: Lisa Patel joined AMD in Bengaluru on July 12, 2026.",
            gold={
                "PERSON": ["Lisa Patel"],
                "ORG": ["AMD"],
                "LOCATION": ["Bengaluru"],
                "DATE": ["July 12, 2026"],
            },
            scorer="entity_exact",
        ),
        BenchmarkTask(
            task_id="ner_002",
            category="ner",
            prompt="Extract named entities from this text: Microsoft opened a new AI lab in London in March 2025.",
            gold={
                "PERSON": [],
                "ORG": ["Microsoft"],
                "LOCATION": ["London"],
                "DATE": ["March 2025"],
            },
            scorer="entity_exact",
        ),
    ]


def make_summarization_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="summary_001",
            category="summarization",
            prompt="Summarize the following text in one sentence: GPUs are designed for parallel computation. They are widely used in graphics, scientific computing, and AI workloads.",
            gold="GPUs are parallel processors used in graphics, scientific computing, and AI.",
            scorer="format_and_semantic",
        ),
        BenchmarkTask(
            task_id="summary_002",
            category="summarization",
            prompt="Summarize the following text in one sentence: Docker containers package applications with their dependencies. This makes software easier to move across machines and environments.",
            gold="Docker containers make applications portable by packaging them with dependencies.",
            scorer="format_and_semantic",
        ),
    ]


def make_factual_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="factual_001",
            category="factual",
            prompt="Explain what a GPU is in two concise sentences.",
            gold="A GPU is a processor designed for parallel computation, often used for graphics and AI workloads.",
            scorer="llm_or_manual",
        ),
        BenchmarkTask(
            task_id="factual_002",
            category="factual",
            prompt="What is Docker? Answer in one sentence.",
            gold="Docker packages applications and dependencies into portable containers.",
            scorer="llm_or_manual",
        ),
        BenchmarkTask(
            task_id="factual_003",
            category="factual",
            prompt="Explain what an API is in simple terms.",
            gold="An API is a way for software programs to communicate with each other.",
            scorer="llm_or_manual",
        ),
    ]


def make_code_generation_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="codegen_001",
            category="code_generation",
            prompt="Write a Python function add(a, b) that returns the sum of a and b. Return code only.",
            gold="unit_tests",
            scorer="python_unit_tests",
            tests=[
                "assert add(2, 3) == 5",
                "assert add(-1, 1) == 0",
            ],
        ),
        BenchmarkTask(
            task_id="codegen_002",
            category="code_generation",
            prompt="Write a Python function is_even(n) that returns True if n is even, otherwise False. Return code only.",
            gold="unit_tests",
            scorer="python_unit_tests",
            tests=[
                "assert is_even(2) is True",
                "assert is_even(7) is False",
            ],
        ),
    ]


def make_code_debug_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="codedebug_001",
            category="code_debug",
            prompt="Fix this Python code and return corrected code only:\n\ndef add(a, b):\n    return a - b",
            gold="unit_tests",
            scorer="python_unit_tests",
            tests=[
                "assert add(2, 3) == 5",
                "assert add(-1, 1) == 0",
            ],
        ),
        BenchmarkTask(
            task_id="codedebug_002",
            category="code_debug",
            prompt="Fix this Python code and return corrected code only:\n\ndef square(x):\n    return x + x",
            gold="unit_tests",
            scorer="python_unit_tests",
            tests=[
                "assert square(3) == 9",
                "assert square(-4) == 16",
            ],
        ),
    ]


def make_logic_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="logic_001",
            category="logic",
            prompt="Logic puzzle: A is taller than B. B is taller than C. Who is the shortest? Return only the name.",
            gold="C",
            scorer="exact_text",
        ),
        BenchmarkTask(
            task_id="logic_002",
            category="logic",
            prompt="Logic puzzle: Red is left of Blue. Green is right of Blue. Which color is in the middle? Return only the color.",
            gold="Blue",
            scorer="exact_text",
        ),
    ]


def build_dataset() -> list[BenchmarkTask]:
    tasks = []
    tasks.extend(make_factual_tasks())
    tasks.extend(make_math_tasks())
    tasks.extend(make_sentiment_tasks())
    tasks.extend(make_summarization_tasks())
    tasks.extend(make_ner_tasks())
    tasks.extend(make_code_generation_tasks())
    tasks.extend(make_code_debug_tasks())
    tasks.extend(make_logic_tasks())
    return tasks


def write_benchmark(tasks: list[BenchmarkTask], output_path: str):
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    records = [task.to_dict() for task in tasks]
    with open(output_path_obj, "w", encoding="utf-8") as output:
        json.dump(records, output, indent=2, ensure_ascii=False)


def main():
    tasks = build_dataset()
    write_benchmark(tasks, BENCHMARK_FILE)
    print(f"Benchmark file is {BENCHMARK_FILE} with {len(tasks)} tasks")


if __name__ == "__main__":
    main()
