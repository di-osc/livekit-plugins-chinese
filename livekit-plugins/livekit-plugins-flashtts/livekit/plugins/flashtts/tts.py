from __future__ import annotations

import time
from typing import Dict, Literal, Optional
import os

import aiohttp
import weakref
from pydantic import BaseModel, Field
from osc_data.text_stream import TextStreamSentencizer
from osc_data.text import TextNormalizer

from livekit.agents import (
    APIConnectOptions,
    tts,
    utils,
    tokenize,
)


from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .log import logger


class TTSOptions(BaseModel):
    base_url: str | None = None
    api_key: str | None = None
    sample_rate: int = 16000
    name: Optional[str] = Field(
        default=None,
        description="The name of the voice character to be used for speech synthesis.",
    )
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = (
        Field(
            default=None,
            description="Specifies the pitch level for the generated audio. Valid options: 'very_low', 'low', 'moderate', 'high', 'very_high'.",
        )
    )
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = (
        Field(
            default=None,
            description="Specifies the speed level of the audio output. Valid options: 'very_low', 'low', 'moderate', 'high', 'very_high'.",
        )
    )
    temperature: float = Field(
        default=0.9,
        description="Controls the randomness of the speech synthesis. A higher temperature produces more diverse outputs.",
    )
    top_k: int = Field(
        default=50,
        description="Limits the sampling to the top 'k' most probable tokens during generation.",
    )
    top_p: float = Field(
        default=0.95,
        description="Nucleus sampling threshold: only tokens with a cumulative probability up to 'top_p' are considered.",
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="Controls the repetition penalty applied to the generated text. "
        "Higher values penalize repeated words and phrases.",
    )
    max_tokens: int = Field(
        default=32768,
        description="Specifies the maximum number of tokens to generate in the output.",
    )
    length_threshold: int = Field(
        default=1000000,
        description="If the input text exceeds this token length threshold, it will be split into multiple segments for synthesis.",
    )
    window_size: int = Field(
        default=100000,
        description="Determines the window size for each text segment when performing segmentation on longer texts.",
    )
    stream: bool = Field(
        default=True,
        description="Indicates whether the audio output should be streamed in real-time (True) or returned only after complete synthesis (False).",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="pcm",
        description=(
            "The format in which to return audio. Supported formats: mp3, opus, aac, flac, wav, pcm. "
            "Note: PCM returns raw 16-bit samples without headers and AAC is not currently supported."
        ),
    )
    extra_headers: Dict[str, str] = Field(
        default={},
        description="Extra headers to be sent with the request.",
    )

    def get_http_url(self) -> str:
        return f"{self.base_url}/speak"

    def get_http_headers(self) -> Dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.extra_headers,
        }

    def get_query_params(self, text: str) -> Dict:
        if self.api_key is None:
            self.api_key = os.environ.get("FLASHTTS_API_KEY", None)
        if self.base_url is None:
            self.base_url = os.environ.get("FLASHTTS_BASE_URL", "http://localhost:8000")
        params = self.model_dump()
        params["text"] = text
        return params


class TTS(tts.TTS):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        sample_rate: int = 16000,
        voice: Optional[str] = "female",
        pitch: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        speed: Optional[
            Literal["very_low", "low", "moderate", "high", "very_high"]
        ] = None,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        max_tokens: int = 32768,
        http_session: aiohttp.ClientSession | None = None,
        extra_headers: Dict[str, str] = {},
    ):
        """flashtts

        Args:
            base_url (str | None, optional): Base URL. Defaults to None.
            api_key (str | None, optional): API key. Defaults to None.
            sample_rate (int, optional): Sample rate. Defaults to 16000.
            voice (Optional[str], optional): voice name. Defaults to "female".
            pitch (Optional[Literal[ "very_low", "low", "moderate", "high", "very_high" ]], optional): Pitch. Defaults to None.
            speed (Optional[Literal[ "very_low", "low", "moderate", "high", "very_high" ]], optional): Speed. Defaults to None.
            temperature (float, optional): Temperature. Defaults to 0.9.
            top_k (int, optional): Top k. Defaults to 50.
            top_p (float, optional): Top p. Defaults to 0.95.
            repetition_penalty (float, optional): Repetition penalty. Defaults to 1.0.
            max_tokens (int, optional): Max tokens. Defaults to 4096.
            stream (bool, optional): Stream. Defaults to False.
            http_session (aiohttp.ClientSession | None, optional): HTTP session. Defaults to None.
            extra_headers (Dict[str, str], optional): Extra headers to be sent with the request. Defaults to {}.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = TTSOptions(
            base_url=base_url,
            api_key=api_key,
            sample_rate=sample_rate,
            name=voice,
            pitch=pitch,
            speed=speed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        raise NotImplementedError

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        stream = SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()

        self._streams.clear()


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        opts: TTSOptions,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts, self._session = opts, session
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

    async def _run(self, emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            mime_type="audio/pcm",
            stream=True,
            num_channels=1,
        )
        splitter = TextStreamSentencizer()
        tn = TextNormalizer(remove_erhua=False)
        first_sentence_spend = None
        start_time = time.perf_counter()
        async for token in self._input_ch:
            if isinstance(token, self._FlushSentinel):
                sentences = splitter.flush()

            else:
                sentences = splitter.push(text=token)
            for sentence in sentences:
                if len(sentence.strip()) > 0:
                    sentence = tn.normalize(sentence)
                    if first_sentence_spend is None:
                        first_sentence_spend = time.perf_counter() - start_time
                        logger.info(
                            "llm first sentence",
                            extra={"spent": str(first_sentence_spend)},
                        )
                    first_response_spend = None
                    emitter.start_segment(segment_id=utils.shortuuid)
                    logger.info("tts start", extra={"sentence": sentence})
                    data = self._opts.get_query_params(text=sentence)
                    if first_response_spend is None:
                        start_time = time.perf_counter()
                    async with self._session.post(
                        self._opts.get_http_url(),
                        json=data,
                        timeout=aiohttp.ClientTimeout(
                            total=30,
                            sock_connect=self._conn_options.timeout,
                        ),
                        headers=self._opts.get_http_headers(),
                    ) as resp:
                        resp.raise_for_status()
                        async for data in resp.content:
                            if first_response_spend is None:
                                first_response_spend = time.perf_counter() - start_time
                                logger.info(
                                    "tts first response",
                                    extra={"spent": str(first_response_spend)},
                                )
                            emitter.push(data=data)
                        self._pushed_text = ""
                        emitter.end_segment()
                    logger.info("tts end")
