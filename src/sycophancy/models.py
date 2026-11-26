from __future__ import annotations

import asyncio
import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

Messages = list[dict[str, str]]


class ModelBackend(ABC):
    @abstractmethod
    def generate_batch(self, conversations: list[Messages]) -> list[str]:
        """Generate one response for each conversation, preserving order."""


class VLLMBackend(ModelBackend):
    def __init__(self, model: str, max_tokens: int = 1024, **engine_kwargs: Any):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError("install the 'vllm' extra to use VLLMBackend") from exc
        self._sampling_params_type = SamplingParams
        self._max_tokens = max_tokens
        self._llm = LLM(model=model, **engine_kwargs)

    def generate_batch(self, conversations: list[Messages]) -> list[str]:
        prompts = [
            self._llm.get_tokenizer().apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in conversations
        ]
        params = self._sampling_params_type(temperature=0, max_tokens=self._max_tokens)
        batch_size = int(os.environ.get("VLLM_GENERATION_BATCH_SIZE", "0"))
        if batch_size <= 0:
            batch_size = len(prompts) or 1
        responses: list[str] = []
        for start in range(0, len(prompts), batch_size):
            outputs = self._llm.generate(prompts[start : start + batch_size], params)
            responses.extend(item.outputs[0].text for item in outputs)
        return responses


class TransformersBackend(ModelBackend):
    def __init__(
        self,
        model: str,
        max_tokens: int = 1024,
        batch_size: int = 8,
        device: str = "cuda",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("install torch and transformers to use this backend") from exc
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto"
        ).to(device)
        self._model.eval()
        self._device = device
        self._max_tokens = max_tokens
        self._batch_size = batch_size

    def generate_batch(self, conversations: list[Messages]) -> list[str]:
        responses: list[str] = []
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


class OpenAICompatibleBackend(ModelBackend):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        cache_dir: str | Path = ".cache/api",
        concurrency: int = 8,
        max_retries: int = 5,
        request_options: dict[str, Any] | None = None,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.request_options = {"temperature": 0, **(request_options or {})}

    def _request_hash(self, messages: Messages) -> str:
        payload = {
            "base_url": self.base_url,
            "model": self.model,
            "messages": messages,
            "options": self.request_options,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    async def _generate_all(self, conversations: list[Messages]) -> list[str]:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("install the 'api' extra to use the API backend") from exc
        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        semaphore = asyncio.Semaphore(self.concurrency)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        async def one(messages: Messages) -> str:
            request_hash = self._request_hash(messages)
            cache_path = self.cache_dir / f"{request_hash}.json"
            if cache_path.exists():
                with cache_path.open(encoding="utf-8") as handle:
                    return str(json.load(handle)["text"])
            async with semaphore:
                for attempt in range(self.max_retries + 1):
                    try:
                        response = await client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            **self.request_options,
                        )
                        text = response.choices[0].message.content or ""
                        temporary = cache_path.with_suffix(".tmp")
                        with temporary.open("w", encoding="utf-8") as handle:
                            json.dump({"text": text}, handle, ensure_ascii=False)
                        temporary.replace(cache_path)
                        return text
                    except Exception:
                        if attempt >= self.max_retries:
                            raise
                        await asyncio.sleep(2**attempt)
            raise AssertionError("unreachable")

        try:
            return list(await asyncio.gather(*(one(messages) for messages in conversations)))
        finally:
            await client.close()

    def generate_batch(self, conversations: list[Messages]) -> list[str]:
        return asyncio.run(self._generate_all(conversations))
