import asyncio
import dataclasses
import hashlib
import json
import os
import threading
from collections.abc import Iterable
from typing import Any, cast

from diskcache import FanoutCache
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

_CACHE = None  # lazy-initialized
_CACHE_LOCK = threading.Lock()


def _cache() -> FanoutCache:
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            _CACHE = FanoutCache(  # pyright: ignore [reportConstantRedefinition]
                directory=".llm_cache", shards=32
            )
    return _CACHE


def _cache_key(**kwargs: Any) -> str:
    m = hashlib.sha1()
    m.update(json.dumps(kwargs, sort_keys=True).encode())
    return m.hexdigest()


@dataclasses.dataclass(kw_only=True, slots=True)
class Message:
    content: str


@dataclasses.dataclass(kw_only=True, slots=True)
class SystemMessage(Message):
    pass


@dataclasses.dataclass(kw_only=True, slots=True)
class UserMessage(Message):
    pass


@dataclasses.dataclass(kw_only=True, slots=True)
class AssistantMessage(Message):
    pass


def enc_msg(msg: Message) -> dict[str, str]:
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    elif isinstance(msg, UserMessage):
        return {"role": "user", "content": msg.content}
    else:
        assert isinstance(msg, AssistantMessage)
        return {"role": "assistant", "content": msg.content}


class Client:
    def __init__(self, provider: str) -> None:
        max_concurrency = 50
        if provider == "openai":
            base_url = None
            api_key = None
        elif provider == "anthropic":
            base_url = "https://api.anthropic.com/v1"
            api_key = os.environ["ANTHROPIC_API_KEY"]
        elif provider == "gemini":
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
            api_key = os.environ["GEMINI_API_KEY"]
        else:
            assert provider == "together"
            base_url = "https://api.together.xyz/v1"
            api_key = os.environ["TOGETHER_API_KEY"]

        self._provider: str = provider
        self._async_openai: AsyncOpenAI = AsyncOpenAI(
            base_url=base_url, api_key=api_key
        )
        self._sema: asyncio.Semaphore = asyncio.Semaphore(max_concurrency)

    def provider(self) -> str:
        return self._provider

    def async_openai(self) -> AsyncOpenAI:
        return self._async_openai

    async def ainvoke(
        self, model: str, input: Iterable[Message], **kwargs: Any
    ) -> AssistantMessage:
        enc_msgs = [cast(ChatCompletionMessageParam, enc_msg(x)) for x in input]
        cache_key = _cache_key(
            provider=self._provider,
            model=model,
            msgs=enc_msgs,
            **kwargs,
        )
        response = _cache().get(cache_key)
        if response is None:
            async with self._sema:
                response = await self._async_openai.chat.completions.create(
                    model=model,
                    messages=enc_msgs,
                    **kwargs,
                )
            _cache().set(cache_key, response)
        content = cast(ChatCompletion, response).choices[0].message.content
        assert content is not None
        return AssistantMessage(content=content)

    async def abatch(
        self, model: str, inputs: Iterable[Iterable[Message]], **kwargs: Any
    ) -> list[AssistantMessage]:
        return await asyncio.gather(*(self.ainvoke(model, x, **kwargs) for x in inputs))
