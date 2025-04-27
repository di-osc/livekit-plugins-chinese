from __future__ import annotations

import os
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Literal

import aiohttp
from pydantic import BaseModel, Field

from livekit.agents import (
    APIConnectOptions,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .log import logger


class TTSOptions(BaseModel):
    api_key: str
    group_id: str
    base_url: str = "https://api.minimax.chat/v1/t2a_v2"
    model: Literal[
        "speech-02-hd",
        "speech-02-turbo",
        "speech-01-hd",
        "speech-01-turbo",
        "speech-01-240228",
        "speech-01-turbo-240228",
    ] = "speech-02-hd"
    language_boost: Literal[
        "Chinese",
        "Chinese,Yue",
        "English",
        "Arabic",
        "Russian",
        "Spanish",
        "French",
        "Portuguese",
        "German",
        "Turkish",
        "Dutch",
        "Ukrainian",
        "Vietnamese",
        "Indonesian",
        "Japanese",
        "Italian",
        "Korean",
        "Thai",
        "Polish",
        "Romanian",
        "Greek",
        "Czech",
        "Finnish",
        "Hindi",
        "auto",
    ] = "auto"

    # audio
    sample_rate: Literal[8000, 16000, 22050, 24000, 32000, 44100] = 16000
    bitrate: Literal[32000, 64000, 128000, 256000] = 128000  # 该参数仅对mp3格式有效
    audio_format: Literal["mp3", "pcm", "flac", "wav"] = "pcm"
    num_channels: Literal[1, 2] = 1

    # speech
    speed: float = Field(1.0, ge=0.5, le=2.0)
    volume: float = Field(1.0, gt=0.0, le=10.0)
    pitch: int = Field(0, ge=-12, le=12)
    voice_id: Literal[
        "male-qn-qingse",
        "male-qn-jingying",
        "male-qn-badao",
        "male-qn-daxuesheng",
        "female-shaonv",
        "female-yujie",
        "female-chengshu",
        "female-tianmei",
        "presenter_male",
        "presenter_female",
        "audiobook_male_1",
        "audiobook_male_2",
        "audiobook_female_1",
        "audiobook_female_2",
        "male-qn-qingse-jingpin",
        "male-qn-jingying-jingpin",
        "male-qn-badao-jingpin",
        "male-qn-daxuesheng-jingpin",
        "female-shaonv-jingpin",
        "female-yujie-jingpin",
        "female-chengshu-jingpin",
        "female-tianmei-jingpin",
        "clever_boy",
        "cute_boy",
        "lovely_girl",
        "cartoon_pig",
        "bingjiao_didi",
        "junlang_nanyou",
        "chunzhen_xuedi",
        "lengdan_xiongzhang",
        "badao_shaoye",
        "tianxin_xiaoling",
        "qiaopi_mengmei",
        "wumei_yujie",
        "diadia_xuemei",
        "danya_xuejie",
        "Santa_Claus",
        "Grinch",
        "Rudolph",
        "Arnold",
        "Charming_Santa",
        "Charming_Lady",
        "Sweet_Girl",
        "Cute_Elf",
        "Attractive_Girl",
        "Serene_Woman",
    ] = "male-qn-jingying"

    def get_http_url(self):
        return f"{self.base_url}?GroupId={self.group_id}"

    def get_http_header(self):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return headers

    def get_query_params(self, text: str) -> Dict:
        request_json = {
            "model": "speech-02-turbo",
            "text": text,
            "stream": True,
            "language_boost": self.language_boost,
            "output_format": "hex",
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": self.speed,
                "vol": self.volume,
                "pitch": self.pitch,
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": self.bitrate,
                "format": self.audio_format,
            },
        }
        return request_json


class TTS(tts.TTS):
    def __init__(
        self,
        api_key: str | None = None,
        group_id: str | None = None,
        model: Literal[
            "speech-02-hd",
            "speech-02-turbo",
            "speech-01-hd",
            "speech-01-turbo",
            "speech-01-240228",
            "speech-01-turbo-240228",
        ] = "speech-02-turbo",
        language_boost: Literal[
            "Chinese",
            "Chinese,Yue",
            "English",
            "Arabic",
            "Russian",
            "Spanish",
            "French",
            "Portuguese",
            "German",
            "Turkish",
            "Dutch",
            "Ukrainian",
            "Vietnamese",
            "Indonesian",
            "Japanese",
            "Italian",
            "Korean",
            "Thai",
            "Polish",
            "Romanian",
            "Greek",
            "Czech",
            "Finnish",
            "Hindi",
            "auto",
        ] = "auto",
        sample_rate: Literal[8000, 16000, 22050, 24000, 32000, 44100] = 8000,
        bitrate: Literal[
            32000, 64000, 128000, 256000
        ] = 128000,  # 该参数仅对mp3格式有效
        audio_format: Literal["mp3", "pcm", "flac", "wav"] = "pcm",
        num_channels: Literal[1, 2] = 1,
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: float = 0.0,
        voice_id: Literal[
            "male-qn-qingse",
            "male-qn-jingying",
            "male-qn-badao",
            "male-qn-daxuesheng",
            "female-shaonv",
            "female-yujie",
            "female-chengshu",
            "female-tianmei",
            "presenter_male",
            "presenter_female",
            "audiobook_male_1",
            "audiobook_male_2",
            "audiobook_female_1",
            "audiobook_female_2",
            "male-qn-qingse-jingpin",
            "male-qn-jingying-jingpin",
            "male-qn-badao-jingpin",
            "male-qn-daxuesheng-jingpin",
            "female-shaonv-jingpin",
            "female-yujie-jingpin",
            "female-chengshu-jingpin",
            "female-tianmei-jingpin",
            "clever_boy",
            "cute_boy",
            "lovely_girl",
            "cartoon_pig",
            "bingjiao_didi",
            "junlang_nanyou",
            "chunzhen_xuedi",
            "lengdan_xiongzhang",
            "badao_shaoye",
            "tianxin_xiaoling",
            "qiaopi_mengmei",
            "wumei_yujie",
            "diadia_xuemei",
            "danya_xuejie",
            "Santa_Claus",
            "Grinch",
            "Rudolph",
            "Arnold",
            "Charming_Santa",
            "Charming_Lady",
            "Sweet_Girl",
            "Cute_Elf",
            "Attractive_Girl",
            "Serene_Woman",
        ] = "male-qn-jingying",
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: float = 600,
    ):
        """mininax TTS

        Args:
            api_key (str | None, optional):  API key. Defaults to None.
            group_id (str | None, optional):  Group ID. Defaults to None.
            base_url (str | None, optional):  Base URL. Defaults to None.
            model (Literal[ "speech-02-hd", "speech-02-turbo", "speech-01-hd", "speech-01-turbo", "speech-01-240228", "speech-01-turbo-240228" ], optional):  Model. Defaults to "speech-02-hd".

        """
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=sample_rate,
            num_channels=1,
        )
        if api_key is None:
            api_key = os.environ.get("MINIMAX_API_KEY")
        if group_id is None:
            group_id = os.environ.get("MINIMAX_GROUP_ID")
        if api_key is None or group_id is None:
            raise ValueError("MINIMAX_API_KEY and MINIMAX_GROUP_ID must be provided")
        self._opts = TTSOptions(
            api_key=api_key,
            group_id=group_id,
            model=model,
            language_boost=language_boost,
            sample_rate=sample_rate,
            bitrate=bitrate,
            audio_format=audio_format,
            num_channels=num_channels,
            speed=speed,
            volume=volume,
            pitch=pitch,
            voice_id=voice_id,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(
        self, text, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ):
        raise NotImplementedError("Minimax TTS does not support synthesize method")

    def stream(self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS):
        return SynthesizeStream(
            tts=self,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )


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

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
        )
        emitter = tts.SynthesizedAudioEmitter(
            event_ch=self._event_ch,
            request_id=request_id,
        )
        splitter = ChineseSentenceSplitter()
        is_last = False
        first_sentence_spend = None
        start_time = time.perf_counter()
        async for token in self._input_ch:
            if isinstance(token, self._FlushSentinel):
                is_last = True
                sentences = splitter.process_text(text="", is_last=True)
            else:
                is_last = False
                sentences = splitter.process_text(text=token, is_last=False)
            for i, sentence in enumerate(sentences):
                if first_sentence_spend is None:
                    first_sentence_spend = time.perf_counter() - start_time
                    logger.info(
                        "llm first sentence", extra={"spent": str(first_sentence_spend)}
                    )
                if len(sentence.strip()) > 0:
                    first_response_spend = None
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
                        headers=self._opts.get_http_header(),
                    ) as resp:
                        resp.raise_for_status()
                        resp.content._high_water = (
                            resp.content._high_water**2
                        )  # 可能会出现内存占用过高的情况，暂时没找到更好方法
                        async for data in resp.content:
                            if data[:5] == b"data:":
                                data = json.loads(data[5:])
                                if "data" in data and "extra_info" not in data:
                                    audio = data["data"]["audio"]
                                    if first_response_spend is None:
                                        first_response_spend = (
                                            time.perf_counter() - start_time
                                        )
                                        logger.info(
                                            "tts first response",
                                            extra={"spent": str(first_response_spend)},
                                        )
                                    for frame in audio_bstream.write(
                                        bytes.fromhex(audio)
                                    ):
                                        emitter.push(frame)

                    logger.info("tts end")
            if is_last:
                for frame in audio_bstream.flush():
                    emitter.push(frame)
                emitter.flush()


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
