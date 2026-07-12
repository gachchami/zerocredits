from dataclasses import dataclass


@dataclass
class HackathonTask:
    task_id: str
    prompt: str

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "HackathonTask":
        return cls(
            task_id=data["task_id"],
            prompt=data["prompt"],
        )

    def to_result(self, answer: str) -> dict[str, str]:
        return {
            "task_id": self.task_id,
            "answer": answer,
        }
