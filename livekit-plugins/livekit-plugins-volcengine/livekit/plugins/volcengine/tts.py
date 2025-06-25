from __future__ import annotations

import asyncio
import gzip
import json
import os
import time
from collections.abc import ByteString
from typing import Literal, Tuple

import aiohttp
from pydantic import BaseModel, Field
from osc_data.text_stream import TextStreamSentencizer

from livekit.agents import (
    APIConnectOptions,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .log import logger


class _TTSOptions(BaseModel):
    app_id: str
    cluster: str
    access_token: str | None = None
    voice: str = "BV001_V2_streaming"
    base_url: str = "wss://openspeech.bytedance.com/api/v1"
    sample_rate: Literal[24000, 16000, 8000] = 24000
    encoding: Literal["mp3", "pcm"] = "pcm"
    speed: float = Field(1.0, ge=0.2, le=3.0)
    volume: float = Field(1.0, gt=0.1, le=3.0)
    pitch: float = Field(1.0, ge=0.1, le=3.0)

    def get_ws_url(self):
        return f"{self.base_url}/tts/ws_binary"

    def get_ws_query_params(self, text: str, uid: str | None = None) -> bytearray:
        if uid is None:
            uid = utils.shortuuid()
        submit_request_json = {
            "app": {
                "appid": self.app_id,
                "token": self.access_token,
                "cluster": self.cluster,
            },
            "user": {"uid": uid},
            "audio": {
                "voice_type": self.voice,
                "encoding": self.encoding,
                "speed_ratio": self.speed,
                "volume_ratio": self.volume,
                "pitch_ratio": self.pitch,
                "rate": self.sample_rate,
            },
            "request": {
                "reqid": utils.shortuuid(),
                "text": text,
                "text_type": "plain",
                "operation": "submit",
                "with_frontend": 1,
                "frontend_type": "unitTson",
            },
        }
        default_header = bytearray(b"\x11\x10\x11\x00")
        payload_bytes = str.encode(json.dumps(submit_request_json))
        payload_bytes = gzip.compress(
            payload_bytes
        )  # if no compression, comment this line
        full_client_request = bytearray(default_header)
        full_client_request.extend(
            (len(payload_bytes)).to_bytes(4, "big")
        )  # payload size(4 bytes)
        full_client_request.extend(payload_bytes)  # payload
        return full_client_request

    def get_ws_header(self):
        if self.access_token is None:
            self.access_token = os.getenv("VOLCENGINE_TTS_ACCESS_TOKEN")
            if self.access_token is None:
                raise ValueError("VOLCENGINE_TTS_ACCESS_TOKEN is not set")
        return {
            "Authorization": f"Bearer;{self.access_token}",
        }


class TTS(tts.TTS):
    def __init__(
        self,
        app_id: str,
        cluster: str,
        access_token: str | None = None,
        voice: str = "BV001_V2_streaming",
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: float = 1.0,
        sample_rate: Literal[24000, 16000, 8000] = 16000,
        http_session: aiohttp.ClientSession | None = None,
    ):
        """VolcEngine TTS

        Args:
            app_id (str): the app id of the tts, you can get it from the console.
            cluster (str): the cluster of the tts, you can get it from the console.
            access_token (str | None, optional): the access token of the tts, if not provided, the value of the environment variable VOLCENGINE_TTS_ACCESS_TOKEN will be used. Defaults to None.
            voice_type (str, optional): the voice type of the tts, you can get it from https://www.volcengine.com/docs/6561/97465. Defaults to "BV001_V2_streaming". if you want to use the streaming api, you must ensure the voice type is end with "_streaming".
            sample_rate (Literal[24000, 16000, 8000], optional): the sample rate of the tts. Defaults to 24000.
            streaming (bool, optional): whether to use the streaming api. Defaults to True.
            http_session (aiohttp.ClientSession | None, optional): the http session to use. Defaults to None.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = _TTSOptions(
            app_id=app_id,
            cluster=cluster,
            access_token=access_token,
            voice=voice,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
            pitch=pitch,
        )
        self._session = http_session

        self._pool = utils.ConnectionPool[
            aiohttp.ClientWebSocketResponse
        ](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=30,  # 火山ws30s会自动关闭，所以至少30s以内自动建立新的连接。
            mark_refreshed_on_get=False,
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
        self, text, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ):
        raise NotImplementedError

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
            pool=self._pool,
            session=self._ensure_session(),
        )


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
        tts: TTS,
        conn_options=None,
    ):
        super().__init__(tts=tts, conn_options=conn_options)
        self._opts: _TTSOptions = opts
        self._session = session
        self._pool = pool

    async def _run(self, emitter: tts.AudioEmitter):
        request_id = utils.shortuuid()

        sentence_splitter = TextStreamSentencizer()
        emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=200,
            stream=True,
        )

        async def _send_task(sentence: str, ws: aiohttp.ClientWebSocketResponse):
            if len(sentence) > 0:
                data = self._opts.get_ws_query_params(text=sentence)
                await ws.send_bytes(data)

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
            is_first_response = True
            start_time = time.perf_counter()
            while True:
                try:
                    res = await ws.receive_bytes()
                except Exception as e:
                    logger.warning(f"Error while receiving bytes: {e}")
                    break
                done, data = parse_response(res)
                if data is not None:
                    if is_first_response:
                        elapsed_time = time.perf_counter() - start_time
                        logger.info(
                            "tts first response",
                            extra={"spent": round(elapsed_time, 4)},
                        )
                        is_first_response = False
                    emitter.push(data=data)
                if done:
                    break

        is_first_sentence = True
        start = time.perf_counter()
        async for token in self._input_ch:
            if isinstance(token, self._FlushSentinel):
                sentences = sentence_splitter.flush()
            else:
                sentences = sentence_splitter.push(text=token)
            for sentence in sentences:
                if len(sentence.strip()) == 0:
                    continue
                if is_first_sentence:
                    is_first_sentence = False
                    elapsed_time = time.perf_counter() - start
                    logger.info(
                        "llm first sentence", extra={"spent": round(elapsed_time, 4)}
                    )
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
                    try:
                        await asyncio.gather(*tasks)
                    finally:
                        await utils.aio.gracefully_cancel(*tasks)
                emitter.end_segment()
                logger.info("tts end")
                self._pushed_text = self._pushed_text.replace(sentence, "")


def parse_response(res) -> Tuple[bool, ByteString | None]:
    header_size = res[0] & 0x0F
    message_type = res[1] >> 4
    message_type_specific_flags = res[1] & 0x0F
    message_compression = res[2] & 0x0F
    payload = res[header_size * 4 :]
    if message_type == 0xB:  # audio-only server response
        if message_type_specific_flags == 0:  # no sequence number as ACK
            return False, None
        else:
            sequence_number = int.from_bytes(payload[:4], "big", signed=True)
            payload = payload[8:]
        if sequence_number < 0:
            return True, payload
        else:
            return False, payload
    elif message_type == 0xF:
        error_msg = payload[8:]
        if message_compression == 1:
            error_msg = gzip.decompress(error_msg)
        error_msg = str(error_msg, "utf-8")

        return True, None
    elif message_type == 0xC:
        payload = payload[4:]
        if message_compression == 1:
            payload = gzip.decompress(payload)
        return False, None
    else:
        return True, None
