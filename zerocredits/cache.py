import hashlib


class RuntimeCache:
    def __init__(self):
        self._answers: dict[str, str] = {}

    def key_for_prompt(self, prompt: str) -> str:
        normalized_prompt = " ".join(prompt.strip().lower().split())
        return hashlib.sha256(normalized_prompt.encode("utf-8")).hexdigest()

    def get(self, prompt: str) -> str | None:
        key = self.key_for_prompt(prompt=prompt)
        return self._answers.get(key, None)

    def set(self, prompt: str, answer: str):
        key = self.key_for_prompt(prompt=prompt)
        self._answers[key] = answer
