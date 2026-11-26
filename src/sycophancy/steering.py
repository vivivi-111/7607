from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .models import Messages, ModelBackend


class SteeringBackend(ModelBackend):
    """Transformers generation with a residual-stream steering vector."""

    def __init__(
        self,
        model: str,
        direction_path: str | Path,
        layer: int,
        alpha: float,
        max_tokens: int = 512,
        batch_size: int = 8,
        device: str = "cuda",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("install torch and transformers to use steering") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.bfloat16
        ).to(device)
        self._model.eval()
        payload = torch.load(direction_path, map_location="cpu", weights_only=True)
        directions = payload["directions"] if "directions" in payload else payload
        self._directions = {int(key): value.float() for key, value in directions.items()}
        self._layer = layer
        self._alpha = alpha
        self._max_tokens = max_tokens
        self._batch_size = batch_size
        self._device = device

    def set_steering(self, layer: int, alpha: float) -> None:
        if layer not in self._directions:
            raise ValueError(f"direction for layer {layer} is unavailable")
        self._layer = layer
        self._alpha = alpha

    @property
    def layers(self) -> list[int]:
        return sorted(self._directions)

    @contextmanager
    def _steering_hook(self) -> Iterator[None]:
        direction = (
            self._directions[self._layer].to(self._device) * self._alpha
        )

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            shifted = hidden + direction.to(dtype=hidden.dtype)
            if isinstance(output, tuple):
                return (shifted, *output[1:])
            return shifted

        layer = self._model.model.layers[self._layer]
        handle = layer.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def generate_batch(self, conversations: list[Messages]) -> list[str]:
        responses: list[str] = []
        with self._steering_hook():
            for start in range(0, len(conversations), self._batch_size):
                batch = conversations[start : start + self._batch_size]
                prompts = [
                    self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    for messages in batch
                ]
                inputs = self._tokenizer(
                    prompts, return_tensors="pt", padding=True
                ).to(self._device)
                input_width = inputs["input_ids"].shape[1]
                with self._torch.inference_mode():
                    generated = self._model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=self._max_tokens,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )
                responses.extend(
                    self._tokenizer.batch_decode(
                        generated[:, input_width:], skip_special_tokens=True
                    )
                )
        return responses
