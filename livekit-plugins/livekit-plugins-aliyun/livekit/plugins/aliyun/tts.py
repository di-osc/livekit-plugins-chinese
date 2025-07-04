import os
from dataclasses import dataclass
from typing import AsyncIterable, Optional, Dict
import time
import aiohttp
import asyncio
import json

from livekit.agents import tts, APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS, utils
from osc_data.text_stream import TextStreamSentencizer

from .log import logger


STREAM_EOS = "EOS"


@dataclass
class TTSOptions:
    api_key: str
    model: str
    # 语速，取值范围：0.5~2。
    rate: float
    # 音色
    voice: str
    # 合成音频的语速，取值范围：0.5~2。
    speech_rate: int
    # 合成音频的音量，取值范围：0~100。
    volume: int
    # 采样率，取值范围：8000, 16000, 22050, 24000, 44100, 48000
    sample_rate: int
    # 音调，取值范围：0.5~2。
    pitch: float = 1.0

    def get_ws_url(self) -> str:
        return "wss://dashscope.aliyuncs.com/api-ws/v1/inference"

    def get_ws_header(self) -> Dict[str, str]:
        return {
            "Authorization": f"bearer {self.api_key}",
            "X-DashScope-DataInspection": "enable",
        }

    def get_run_task_params(self) -> Dict[str, str]:
        params = {
            "header": {
                "action": "run-task",
                "task_id": utils.shortuuid(),
                "streaming": "duplex",
            },
            "payload": {
                "task_group": "audio",
                "task": "tts",
                "function": "SpeechSynthesizer",
                "model": self.model,
                "parameters": {
                    "text_type": "PlainText",
                    "voice": self.voice,
                    "format": "pcm",
                    "sample_rate": self.sample_rate,
                    "volume": self.volume,
                    "rate": self.rate,
                    "pitch": self.pitch,
                },
                "input": {},
            },
        }
        return params

    def get_continue_task_params(self, text: str) -> Dict[str, str]:
        params = {
            "header": {
                "action": "continue-task",
                "task_id": utils.shortuuid(),
                "streaming": "duplex",
            },
            "payload": {
                "input": {
                    "text": text,
                }
            },
        }
        return params

    def get_finish_task_params(self) -> Dict[str, str]:
        params = {
            "header": {
                "action": "finish-task",
                "task_id": utils.shortuuid(),
                "streaming": "duplex",
            },
            "payload": {"input": {}},
        }
        return params


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        sample_rate: int = 24000,
        voice: str = "longcheng",
        model: str = "cosyvoice-v2",
        speech_rate: int = 1,
        volume: int = 100,
        rate: float = 1.0,
        pitch: float = 1.0,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: float = 600,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY must be set")
        self._session = http_session
        self._opts = TTSOptions(
            model=model,
            api_key=api_key,
            voice=voice,
            speech_rate=speech_rate,
            volume=volume,
            sample_rate=sample_rate,
            rate=rate,
            pitch=pitch,
        )
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=max_session_duration,
            mark_refreshed_on_get=True,
        )

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()

        return self._session

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url()
        headers = self._opts.get_ws_header()
        return await asyncio.wait_for(
            session.ws_connect(url, headers=headers),
            timeout=timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

    def synthesize(
        self,
        text: str,
    ) -> AsyncIterable[tts.SynthesizedAudio]:
        raise NotImplementedError

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> "SynthesizeStream":
        return SynthesizeStream(tts=self, opts=self._opts, conn_options=conn_options)


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        opts: TTSOptions,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts = opts

    async def _run(self, emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            mime_type="audio/pcm",
            stream=True,
            num_channels=1,
            frame_size_ms=200,
        )

        async def _send_task(sentence: str, ws: aiohttp.ClientWebSocketResponse):
            run_task_params = self._opts.get_run_task_params()
            await ws.send_json(run_task_params)
            continue_task_params = self._opts.get_continue_task_params(text=sentence)
            await ws.send_json(continue_task_params)
            finish_task_params = self._opts.get_finish_task_params()
            await ws.send_json(finish_task_params)

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
            is_first_response = True
            start_time = time.perf_counter()
            while True:
                try:
                    msg = await ws.receive()
                except Exception as e:
                    logger.warning(f"Error while receiving bytes: {e}")
                    break
                if msg.type == aiohttp.WSMsgType.BINARY:
                    if is_first_response:
                        elapsed_time = time.perf_counter() - start_time
                        logger.info(
                            "tts first response",
                            extra={"spent": round(elapsed_time, 4)},
                        )
                        is_first_response = False
                    emitter.push(data=msg.data)
                elif msg.type == aiohttp.WSMsgType.TEXT:
                    msg_json = json.loads(msg.data)
                    if "header" in msg_json:
                        header = msg_json["header"]
                        if "event" in header:
                            event = header["event"]
                            if event == "task-finished":
                                break
                            if event == "task-failed":
                                logger.error(f"tts task failed: {msg_json}")
                                break

        splitter = TextStreamSentencizer(remove_emoji=True)
        is_first_sentence = True
        start_time = time.perf_counter()
        async for token in self._input_ch:
            if isinstance(token, self._FlushSentinel):
                sentences = splitter.flush()
            else:
                sentences = splitter.push(text=token)
            for sentence in sentences:
                if is_first_sentence:
                    first_sentence_spend = time.perf_counter() - start_time
                    logger.info(
                        "llm first sentence",
                        extra={"spent": str(first_sentence_spend)},
                    )
                    is_first_sentence = False
                logger.info("tts start", extra={"sentence": sentence})
                emitter.start_segment(segment_id=utils.shortuuid())
                async with self._tts._pool.connection(
                    timeout=self._conn_options.timeout
                ) as ws:
                    assert not ws.closed, "WebSocket connection is closed"
                    tasks = [
                        asyncio.create_task(_send_task(sentence=sentence, ws=ws)),
                        asyncio.create_task(_recv_task(ws=ws)),
                    ]
                    await asyncio.gather(*tasks)
                    emitter.end_segment()
                    logger.info("tts end", extra={"sentence": sentence})
                    self._pushed_text = self._pushed_text.replace(sentence, "")
                    await utils.aio.gracefully_cancel(*tasks)
