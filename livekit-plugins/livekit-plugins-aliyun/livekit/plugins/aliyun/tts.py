from __future__ import annotations

import asyncio
import base64
import gzip
import json
import os
from collections.abc import ByteString
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import aiohttp
from pydantic import BaseModel, Field

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .log import logger


class _TTSOptions(BaseModel):
    app_id: str
    cluster: str
    access_token: str | None = None
    voice_type: str = "BV001_V2_streaming"
    base_url: str = "https://openspeech.bytedance.com/api/v1"
    sample_rate: Literal[24000, 16000, 8000] = 24000
    encoding: Literal["mp3", "pcm"] = "pcm"
    speed: float = Field(1.0, ge=0.2, le=3.0)
    volume: float = Field(1.0, gt=0.1, le=3.0)
    pitch: float = Field(1.0, ge=0.1, le=3.0)

    def get_http_url(self):
        return f"{self.base_url}/tts"

    def get_http_header(self):
        if self.access_token is None:
            self.access_token = os.getenv("VOLCENGINE_TTS_ACCESS_TOKEN")
            if self.access_token is None:
                raise ValueError("VOLCENGINE_TTS_ACCESS_TOKEN is not set")
        return {
            "Authorization": f"Bearer;{self.access_token}",
        }

    def get_http_query_params(self, text: str, uid: str | None = None) -> Dict:
        if uid is None:
            uid = utils.shortuuid()
        request_json = {
            "app": {
                "appid": self.app_id,
                "token": self.access_token,
                "cluster": self.cluster,
            },
            "user": {"uid": uid},
            "audio": {
                "voice_type": self.voice_type,
                "encoding": self.encoding,
                "speed_ratio": self.speed,
                "volume_ratio": self.volume,
                "pitch_ratio": 1.0,
                "rate": self.sample_rate,
            },
            "request": {
                "reqid": utils.shortuuid(),
                "text": text,
                "text_type": "plain",
                "operation": "query",
                "with_frontend": self.pitch,
                "frontend_type": "unitTson",
            },
        }
        return request_json

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
                "voice_type": self.voice_type,
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
        voice_type: str = "BV001_V2_streaming",
        sample_rate: Literal[24000, 16000, 8000] = 24000,
        streaming: bool = True,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: float = 600,
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
            max_session_duration (float, optional): the max duration of the http session. Defaults to 600.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=streaming),
            sample_rate=sample_rate,
            num_channels=1,
        )
        self._opts = _TTSOptions(
            app_id=app_id,
            cluster=cluster,
            access_token=access_token,
            voice_type=voice_type,
            sample_rate=sample_rate,
        )
        self._session = http_session

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

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url()
        headers = self._opts.get_ws_header()
        return await asyncio.wait_for(
            session.ws_connect(url, headers=headers), self._conn_options.timeout
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

    def synthesize(
        self, text, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> ChunedStream:
        return ChunedStream(
            opts=self._opts,
            session=self._ensure_session(),
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
            pool=self._pool,
            session=self._ensure_session(),
        )


class ChunedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        opts: _TTSOptions,
        session: aiohttp.ClientSession,
        tts: TTS,
        input_text,
        conn_options=None,
    ):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts: _TTSOptions = opts
        self._session = session

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=1
        )
        data = self._opts.get_http_query_params(text=self._input_text)
        headers = self._opts.get_http_header()
        try:
            async with self._session.post(
                self._opts.get_http_url(),
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=30,
                    sock_connect=self._conn_options.timeout,
                ),
            ) as resp:
                resp.raise_for_status()
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                )
                data = await resp.json()
                if "data" in data:
                    data = data["data"]
                    data = base64.b64decode(data)
                    frames = bstream.write(data)
                    for frame in frames:
                        emitter.push(frame)
                    for frame in bstream.flush():
                        emitter.push(frame)
                    emitter.flush()
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            emitter.flush()


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

    async def _run(self):
        request_id = utils.shortuuid()

        sentence_splitter = ChineseSentenceSplitter()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
        )
        emitter = tts.SynthesizedAudioEmitter(
            event_ch=self._event_ch,
            request_id=request_id,
        )

        async def _send_task(sentence: str, ws: aiohttp.ClientWebSocketResponse):
            if len(sentence) > 0:
                data = self._opts.get_ws_query_params(text=sentence)
                await ws.send_bytes(data)

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
            is_first_response = True
            while True:
                try:
                    res = await ws.receive_bytes()
                except Exception as e:
                    logger.warning(f"Error while receiving bytes: {e}")
                    break
                done, data = parse_response(res)
                if data is not None:
                    if is_first_response:
                        logger.info("tts first response")
                        is_first_response = False
                    frames = bstream.write(data)
                    for frame in frames:
                        emitter.push(frame)
                if done:
                    for frame in bstream.flush():
                        emitter.push(frame)
                    emitter.flush()
                    break

        is_first_sentence = True
        async for token in self._input_ch:
            if isinstance(token, self._FlushSentinel):
                sentences = sentence_splitter.process_text(text="", is_last=True)
            else:
                sentences = sentence_splitter.process_text(text=token, is_last=False)
            for sentence in sentences:
                if len(sentence.strip()) == 0:
                    continue
                if is_first_sentence:
                    logger.info("llm first sentence")
                logger.info("tts start", extra={"sentence": sentence})
                ws: aiohttp.ClientWebSocketResponse = await self._tts._connect_ws()
                assert not ws.closed, "WebSocket connection is closed"
                tasks = [
                    asyncio.create_task(_send_task(sentence=sentence, ws=ws)),
                    asyncio.create_task(_recv_task(ws=ws)),
                ]
                await asyncio.gather(*tasks)
                await utils.aio.gracefully_cancel(*tasks)
                await self._tts._close_ws(ws)
                logger.info("tts end", extra={"sentence": sentence})
                if is_first_sentence:
                    is_first_sentence = False


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


@dataclass
class ChineseSentenceSplitter:
    buffer: str = ""
    use_level2_threshold: int = 100
    use_level3_threshold: int = 200

    def process_text(
        self,
        text: str,
        is_last: bool = False,
        special_text: str | None = None,
    ) -> List[str]:
        self.buffer = self.buffer + text
        if special_text is not None:
            if self.buffer.endswith(special_text):
                return [self.buffer]
        sentences, indices = self.split_sentences(self.buffer)
        assert len(sentences) == len(indices), (
            "The number of sentences and indices do not match"
        )
        if not is_last:
            if len(indices) != 0:
                self.buffer = self.buffer[indices[-1] + 1 :]
            return sentences
        else:
            if len(sentences) == 0:
                sentences = [self.buffer]
                self.buffer = ""
                return sentences
            if indices[-1] == len(self.buffer) - 1:
                self.buffer = ""
                return sentences
            else:
                self.buffer = ""
                return sentences + [text[indices[-1] + 1 :]]

    def split_sentences(self, text: str) -> List[str]:
        indices = self.get_sentence_end_indices(text)
        sentences = []
        start = 0
        for i in indices:
            t = text[start : i + 1]
            if len(t) > 0:
                sentences.append(t)
                start = i + 1
        return sentences, indices

    def is_sentence_end_level1(self, text: str) -> bool:
        return text.endswith(
            (
                "!",
                "?",
                "。",
                "？",
                "！",
                "；",
                ";",
            )
        )

    def is_sentence_end_level2(self, text: str) -> bool:
        return text.endswith(
            (
                "、",
                "...",
                "…",
                ",",
                "，",
            )
        )

    def is_sentence_end_level3(self, text: str) -> bool:
        return text.endswith(
            (
                ":",
                "：",
            )
        )

    def get_sentence_end_indices(self, text: str) -> List[int]:
        sents_l1 = [i for i, c in enumerate(text) if self.is_sentence_end_level1(c)]
        if len(sents_l1) == 0 and len(text) > self.use_level2_threshold:
            sents_l2 = [i for i, c in enumerate(text) if self.is_sentence_end_level2(c)]
            if len(sents_l2) == 0 and len(text) > self.use_level3_threshold:
                sents_l3 = [
                    i for i, c in enumerate(text) if self.is_sentence_end_level3(c)
                ]
                return sents_l3
            else:
                return sents_l2

        else:
            return sents_l1

    def reset(self):
        self.buffer = ""
