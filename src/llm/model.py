import requests

from src.config import LM_STUDIO_BASE_URL, LM_STUDIO_MODEL


class LMStudioPipeline:
    """Callable client for LM Studio's local OpenAI-compatible /v1/completions endpoint."""

    def __init__(self, base_url: str = LM_STUDIO_BASE_URL, model: str = LM_STUDIO_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def __call__(self, prompt: str, max_new_tokens: int = 512, do_sample: bool = True, temperature: float = 0.7):
        response = requests.post(
            f"{self.base_url}/completions",
            json={
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_new_tokens,
                "temperature": temperature if do_sample else 0.0,
            },
            timeout=300,
        )
        response.raise_for_status()
        completion = response.json()["choices"][0]["text"]
        return [{"generated_text": completion}]


def load_pipeline():
    print(f"Connecting to LM Studio at {LM_STUDIO_BASE_URL} (model: {LM_STUDIO_MODEL})")
    return LMStudioPipeline()
