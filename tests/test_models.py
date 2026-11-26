from types import SimpleNamespace

from sycophancy.models import VLLMBackend


class _Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[0]["content"]


class _LLM:
    def __init__(self):
        self.batch_sizes = []

    def get_tokenizer(self):
        return _Tokenizer()

    def generate(self, prompts, _params):
        self.batch_sizes.append(len(prompts))
        return [
            SimpleNamespace(outputs=[SimpleNamespace(text=f"response:{prompt}")])
            for prompt in prompts
        ]


def test_vllm_generation_batch_size_preserves_order(monkeypatch):
    monkeypatch.setenv("VLLM_GENERATION_BATCH_SIZE", "2")
    backend = object.__new__(VLLMBackend)
    backend._llm = _LLM()
    backend._sampling_params_type = lambda **kwargs: kwargs
    backend._max_tokens = 32
    conversations = [[{"role": "user", "content": str(index)}] for index in range(5)]

    responses = backend.generate_batch(conversations)

    assert backend._llm.batch_sizes == [2, 2, 1]
    assert responses == [f"response:{index}" for index in range(5)]
