from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIStatusError,
    llm,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectOptions,
    NotGivenOr,
    NOT_GIVEN,
)
from livekit.agents.llm import FunctionTool, ToolChoice

from .log import logger


@dataclass
class LLMOptions:
    api_key: str
    api_base: str = "https://api.dify.ai/v1"
    user: str | None = None
    _conversation_id: str = ""

    def get_parameters_url(self) -> str:
        return f"{self.api_base}/parameters"

    def get_message_history_url(self) -> str:
        return f"{self.api_base}/messages"

    def get_conversation_history_url(self) -> str:
        return f"{self.api_base}/conversations"

    def get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


class LLM(llm.LLM):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        user: str | None = None,
    ) -> None:
        """
        Create a new instance of Dify LLM.

        api_key (str | None): The Dify API key. Defaults to DIFY_API_KEY env var.
        api_base (str | None): The base URL for the Dify API. Defaults to https://api.dify.ai/v1.
        temperature (float | None): The temperature for generation. Defaults to None.
        conversation_id (str | None): The conversation ID to continue. Defaults to None.
        """
        super().__init__()
        self._session: Optional[aiohttp.ClientSession] = None

        api_key = api_key or os.environ.get("DIFY_API_KEY")
        if api_key is None:
            raise ValueError("Dify API key is required")

        api_base = api_base or os.environ.get("DIFY_API_BASE", "https://api.dify.ai/v1")

        if user is None:
            user = os.environ.get("DIFY_USER", "test_user")

        self._opts = LLMOptions(
            api_key=api_key,
            api_base=api_base,
            user=user,
        )

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        tools: list[FunctionTool] | None = None,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> "LLMStream":
        """Start a chat completion stream"""
        # Extract the last user message
        last_message = next(
            (msg for msg in reversed(chat_ctx.items) if msg.role == "user"), None
        )
        if not last_message:
            raise ValueError("No user message found in chat context")

        # Prepare the request payload
        payload = {
            "inputs": {},
            "query": last_message.content[0],
            "response_mode": "streaming",
            "conversation_id": self._opts._conversation_id,
            "user": self._opts.user,
        }
        # Create headers
        headers = {
            "Authorization": f"Bearer {self._opts.api_key}",
            "Content-Type": "application/json",
        }

        # Create or reuse the session
        if self._session is None:
            self._session = aiohttp.ClientSession()
        logger.info("llm start", extra={"query": last_message.content[0]})
        # Create the stream
        stream = self._session.post(
            f"{self._opts.api_base}/chat-messages", headers=headers, json=payload
        )

        return LLMStream(
            self,
            dify_stream=stream,
            chat_ctx=chat_ctx,
            conn_options=conn_options,
            tools=tools,
        )

    def ensure_session(self) -> None:
        """Ensure that the session is created"""
        if self._session is None:
            self._session = aiohttp.ClientSession()

    async def get_parameters(self) -> Dict:
        """Get the parameters for the LLM"""
        self.ensure_session()
        res = await self._session.get(
            url=self._opts.get_parameters_url(), headers=self._opts.get_headers()
        )
        return await res.json()

    async def get_opening_words(self) -> str:
        """Get the introduction text for the LLM"""
        params = await self.get_parameters()
        return params.get("opening_statement", "")

    async def is_chatable(self) -> bool:
        """Check if the LLM is chatable"""
        url = self._opts.get_conversation_history_url()
        url = f"{url}?&user={self._opts.user}"
        self.ensure_session()
        res = await self._session.get(url=url, headers=self._opts.get_headers())
        if res.status != 200:
            return False
        return True

    async def required_inputs(self) -> bool:
        """Check if the LLM requires inputs"""
        paras = await self.get_parameters()
        user_input_form = paras.get("user_input_form", [])
        if len(user_input_form) == 0:
            return False
        else:
            required = False
            for item in user_input_form:
                key = list(item.items())[0][0]
                _item = item.get(key, {})
                required = _item.get("required", False)
                break
            return required

    async def close(self) -> None:
        """Close the LLM client and cleanup resources"""
        if self._session is not None:
            await self._session.close()
            self._session = None

    @classmethod
    def from_env(cls) -> "LLM":
        """Create a DifyLLM instance from environment variables"""
        api_key = os.getenv("DIFY_API_KEY")
        if not api_key:
            raise ValueError("DIFY_API_KEY environment variable is required")

        api_base = os.getenv("DIFY_API_BASE", "https://api.dify.ai/v1")

        return cls(
            api_key=api_key,
            api_base=api_base,
        )


class LLMStream(llm.LLMStream):
    def __init__(
        self,
        llm: LLM,
        *,
        dify_stream: aiohttp.ClientResponse,
        chat_ctx: llm.ChatContext,
        conn_options: APIConnectOptions,
        tools: list[FunctionTool] | None = None,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, conn_options=conn_options, tools=tools)
        self._awaitable_dify_stream = dify_stream
        self._dify_stream: aiohttp.ClientResponse | None = None
        self._request_id: str = ""
        self._input_tokens = 0
        self._output_tokens = 0

    async def _run(self) -> None:
        """Run the LLM stream"""
        is_chatable = await self._llm.is_chatable()
        required_inputs = await self._llm.required_inputs()
        is_available = is_chatable and (not required_inputs)
        if not is_available:
            error_text = "Dify is not available"
            raise APIConnectionError(error_text, retryable=False)

        retryable = True
        first_response = True
        try:
            if not self._dify_stream:
                self._dify_stream = await self._awaitable_dify_stream
            async with self._dify_stream as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise APIStatusError(
                        f"Dify API error: {error_text}",
                        status_code=response.status,
                        body=error_text,
                    )
                async for line in response.content:
                    if line:
                        try:
                            line = line.decode("utf-8").strip()
                            if line.startswith("data: "):
                                data = json.loads(line[6:])
                                chat_chunk = self._parse_event(data, self._llm._opts)
                                if chat_chunk is not None:
                                    self._event_ch.send_nowait(chat_chunk)
                                    retryable = False
                                    if first_response:
                                        logger.info("llm first response")
                                        first_response = False
                        except Exception as e:
                            logger.error(f"Error processing stream: {e}")
                            continue

                # Send final usage stats
                self._event_ch.send_nowait(
                    llm.ChatChunk(
                        id=self._request_id,
                        usage=llm.CompletionUsage(
                            completion_tokens=self._output_tokens,
                            prompt_tokens=self._input_tokens,
                            total_tokens=self._input_tokens + self._output_tokens,
                        ),
                    )
                )
                logger.info("llm end")

        except aiohttp.ClientError as e:
            raise APIConnectionError(retryable=retryable) from e
        except Exception as e:
            raise APIConnectionError(retryable=retryable) from e

    def _parse_event(
        self, event: Dict[str, Any], ops: LLMOptions
    ) -> llm.ChatChunk | None:
        """Parse a Dify event into a ChatChunk"""
        event_type = event.get("event")
        if event_type == "message_end":
            # Update usage statistics
            if "metadata" in event and "usage" in event["metadata"]:
                usage = event["metadata"]["usage"]
                self._input_tokens = usage.get("prompt_tokens", 0)
                self._output_tokens = usage.get("completion_tokens", 0)
            ops._conversation_id = event.get("conversation_id", "")
            return None

        elif event_type == "agent_message" or event_type == "message":
            # Extract message content
            answer = event.get("answer", "")
            if not answer:
                return None

            return llm.ChatChunk(
                id=event.get("message_id", ""),
                delta=llm.ChoiceDelta(
                    role="assistant",
                    content=answer,
                ),
            )

        return None
