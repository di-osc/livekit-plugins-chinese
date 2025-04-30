import asyncio
import codecs
import json
import os
from typing import MutableSet, Literal, Any

import aiohttp
from attrs import define
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents import APIConnectionError, APIStatusError, APITimeoutError, llm
from livekit.agents.llm import ToolChoice
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import FunctionTool

from .utils import to_chat_ctx, to_fnc_ctx


ChatModels = Literal["qwen-plus", "qwen-max"]


@define
class LLMOptions:
    model: ChatModels
    base_url: str
    api_key: str


class LLM(llm.LLM):
    def __init__(
            self,
            *,
            model: ChatModels = "qwen-plus",
            api_key: str = "",
            base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    ) -> None:
        api_key = api_key or os.environ.get("ALIYUN_LLM_API_KEY")
        base_url = base_url or os.environ.get("ALIYUN_LLM_BASE_URL")
        if not api_key or not base_url:
            dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY")
            if not dashscope_api_key:
                raise ValueError("both DASHSCOPE_API_KEY or other  llm base_url and api_key must be set")
            else:
                self._opts = LLMOptions(model=model, base_url="xxxx", api_key=dashscope_api_key)
        else:
            self._opts = LLMOptions(model=model, base_url=base_url, api_key=api_key)
        self._running_fncs: MutableSet[asyncio.Task] = set()

    async def fetch_stream(self, url, headers, data):
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=json.dumps(data), timeout=60000) as resp:
                lines = []
                async for block in resp.content.iter_any():
                    lines = block.decode('utf-8').splitlines(keepends=False)
                    for line in lines:
                        if line:
                            yield line.encode('utf-8')

    async def chat(
            self,
            chat_ctx: ChatContext,
            tools: list[FunctionTool] | None = None,
            conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
            parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
            tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
            extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "LLMStream":
        pass
        # llm_config = self._opts
        # url = llm_config.base_url
        # headers = {
        #     'Authorization': f"Bearer {llm_config.api_key}",
        #     'Content-Type': 'application/json'
        # }

        # req_data = {
        #     "messages": to_openai_ctx(history),
        #     "stream": True,
        #     "max_tokens": 500,
        #     "model": self._opts.model,
        # }
        # async_gen = self.fetch_stream(url, headers, req_data)
        # return LLMStream(async_gen, fnc_ctx)


class LLMStream(llm.LLMStream):
    def __init__(
            self, oai_stream, fnc_ctx: llm.FunctionContext | None
    ) -> None:
        super().__init__()
        self._oai_stream = oai_stream
        self._fnc_ctx = fnc_ctx
        self._running_fncs: MutableSet[asyncio.Task] = set()

    def __aiter__(self) -> "LLMStream":
        return self

    async def __anext__(self) -> llm.ChatChunk:
        answer = ""
        async for line in self._oai_stream:
            if line:
                line = codecs.decode(line)
                if line.startswith("data:"):
                    line = line[5:].strip()
                    try:
                        chunk = json.loads(line)
                        if "choices" in chunk and len(
                                chunk["choices"]) > 0 and "delta" in chunk["choices"][0] and "content" in \
                                chunk["choices"][0]["delta"]:
                            delta = chunk["choices"][0]["delta"]
                            answer += delta["content"]
                            print(answer)
                            return llm.ChatChunk(
                                choices=[
                                    llm.Choice(
                                        delta=llm.ChoiceDelta(
                                            content=delta["content"],
                                            role='assistant',
                                        ),
                                        index=0,
                                    )
                                ]
                            )
                    except json.JSONDecodeError as err:
                        # print(f"error : {err}")
                        raise StopAsyncIteration
            else:
                raise StopAsyncIteration
        raise StopAsyncIteration

    async def aclose(self, wait: bool = True) -> None:
        if not wait:
            for task in self._running_fncs:
                task.cancel()

        await asyncio.gather(*self._running_fncs, return_exceptions=True)


def to_openai_ctx(chat_ctx: llm.ChatContext) -> list:
    print(chat_ctx)
    content = '''
    # 角色
    你是小六，是由六六六研发的智能助手，能够与用户交流,回答问题。

    ## 技能
    ### 技能 1: 正确理解用户的问题
    用户的问题由语音转文字而来,可能存在错字别字,请你正确理解相关问题
    ### 技能 2: 回答用户的问题
    1. 你的回答应当简洁明了,避免过分冗长,尽量以50字内的回答回复用户。
    ### 技能 3: 当用户叫你名字"小六"进行对话唤醒时,回答"您好，我是小六，请问有什么可以帮助您的？"
    
    ## 限制:
    - 当用户询问你的身份信息等敏感问题时,回答你的名字"我是小六，使用六六六全新一代AI视觉大模型，欢迎随时找我聊天"或者""对不起,我还不太了解""
    '''
    messages = [
        {'role': 'system', 'content': content}
    ]
    for msg in chat_ctx.messages:
        messages.append(
            {
                    "role": 'user',
                    "content": msg.text,
            }
        )

    print(messages)
    return messages