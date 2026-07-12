from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Category = Literal[
    "factual",
    "math",
    "sentiment",
    "summarization",
    "ner",
    "code_debug",
    "logic",
    "code_generation",
]

Scorer = Literal[
    "llm_or_manual",
    "numeric_exact",
    "label_exact",
    "format_and_semantic",
    "entity_exact",
    "python_unit_tests",
    "exact_text",
]


@dataclass
class BenchmarkTask:
    task_id: str
    category: Category
    prompt: str
    gold: Any
    scorer: Scorer
    tests: list[str] = field(default_factory=list)
    source: str = "local_generated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_hackathon_task(self) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkTask":
        return cls(
            task_id=data["task_id"],
            category=data["category"],
            prompt=data["prompt"],
            gold=data["gold"],
            scorer=data["scorer"],
            tests=data.get("tests", []),
            source=data.get("source", "unknown"),
        )
