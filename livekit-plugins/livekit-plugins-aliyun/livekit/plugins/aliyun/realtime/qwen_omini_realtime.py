from __future__ import annotations

import base64
import copy
import time
from typing import Any, Literal, cast

import aiohttp
import av
import numpy as np
from typing_extensions import TypedDict

from livekit import rtc
from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from ..log import logger
from ._base import (
    DEFAULT_BASE_URL,
    DEFAULT_TURN_DETECTION,
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    Modalities,
    RealtimeSession,
    Region,
    TurnDetection,
    _QwenRealtimeModel,
    _RealtimeFamily,
    _RealtimeOptions,
)

QwenOmniRealtimeModelName = Literal[
    "qwen3.8-omni-flash-realtime",
    "qwen3.5-omni-plus-realtime",
    "qwen3.5-omni-flash-realtime",
]
"""Qwen-Omni Realtime model identifiers."""

DEFAULT_OMNI_MODEL: QwenOmniRealtimeModelName = "qwen3.5-omni-flash-realtime"
DEFAULT_OMNI_VOICE: str = "Tina"
"""Default Qwen-Omni output voice for the 3.5 and 3.8 realtime series."""

class _OmniServerVadRequired(TypedDict):
    type: Literal["server_vad"]


class _OmniSemanticVadRequired(TypedDict):
    type: Literal["semantic_vad"]


class _OmniSemanticVadEcRequired(TypedDict):
    type: Literal["semantic_vad_ec"]


class _OmniServerVad(_OmniServerVadRequired, total=False):
    threshold: float
    silence_duration_ms: int
    idle_timeout_ms: int


class _OmniSemanticVad(_OmniSemanticVadRequired, total=False):
    threshold: float
    silence_duration_ms: int


class _OmniSemanticVadEc(_OmniSemanticVadEcRequired, total=False):
    threshold: float
    silence_duration_ms: int


QwenOmniTurnDetectionConfig = (
    _OmniServerVad | _OmniSemanticVad | _OmniSemanticVadEc | None
)
"""Qwen-Omni turn detection. ``None`` selects manual push-to-talk."""

_OMNI_TURN_TYPES = frozenset({"server_vad", "semantic_vad", "semantic_vad_ec"})
_OMNI_35_MODELS = frozenset(
    {"qwen3.5-omni-plus-realtime", "qwen3.5-omni-flash-realtime"}
)
_OMNI_38_MODEL = "qwen3.8-omni-flash-realtime"


def _check_threshold(options: dict[str, Any]) -> None:
    if "threshold" not in options:
        return
    value = options["threshold"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not -1.0 <= float(value) <= 1.0
    ):
        raise ValueError("threshold must be between -1.0 and 1.0")


def _check_silence_duration(options: dict[str, Any]) -> None:
    if "silence_duration_ms" not in options:
        return
    value = options["silence_duration_ms"]
    if isinstance(value, bool) or not isinstance(value, int) or not 200 <= value <= 6000:
        raise ValueError("silence_duration_ms must be an integer between 200 and 6000")


def _check_idle_timeout(model: str, kind: str, options: dict[str, Any]) -> None:
    if "idle_timeout_ms" not in options:
        return
    if kind != "server_vad" or model not in _OMNI_35_MODELS:
        raise ValueError(
            "idle_timeout_ms is only supported by qwen3.5-omni-plus-realtime "
            "and qwen3.5-omni-flash-realtime with server_vad"
        )
    value = options["idle_timeout_ms"]
    if isinstance(value, bool) or not isinstance(value, int) or not 5000 <= value <= 30000:
        raise ValueError("idle_timeout_ms must be an integer between 5000 and 30000")


def _resolve_omni_turn_detection(
    model: str,
    turn_detection: NotGivenOr[QwenOmniTurnDetectionConfig],
) -> QwenOmniTurnDetectionConfig:
    if not is_given(turn_detection):
        return copy.deepcopy(DEFAULT_TURN_DETECTION)
    if turn_detection is None:
        return None
    resolved = copy.deepcopy(turn_detection)
    kind = resolved.get("type")
    if kind in {"smart_turn", "smart_turn_v2"}:
        raise ValueError(
            "Qwen Omni Realtime does not support smart_turn or smart_turn_v2; "
            "use server_vad, semantic_vad, semantic_vad_ec, or None"
        )
    if kind not in _OMNI_TURN_TYPES:
        raise ValueError(
            "Qwen Omni Realtime turn detection must be server_vad, "
            f"semantic_vad, semantic_vad_ec, or None; got {kind!r}"
        )
    if kind == "semantic_vad_ec" and model != _OMNI_38_MODEL:
        raise ValueError(
            "semantic_vad_ec is only supported by qwen3.8-omni-flash-realtime"
        )
    _check_threshold(resolved)
    _check_silence_duration(resolved)
    _check_idle_timeout(model, str(kind), resolved)
    return cast(QwenOmniTurnDetectionConfig, resolved)


class QwenOmniTurnDetection:
    """Qwen-Omni turn-detection builders.

    ``server_vad`` ends a turn after silence. ``semantic_vad`` ignores
    backchannels and background speech. ``semantic_vad_ec`` interrupts more
    readily and is only accepted by ``qwen3.8-omni-flash-realtime``. Pass
    ``None`` to :class:`QwenOmniRealtime` for manual push-to-talk.
    """

    @staticmethod
    def server_vad(
        *,
        threshold: float = 0.5,
        silence_duration_ms: int = 800,
        idle_timeout_ms: int | None = None,
    ) -> _OmniServerVad:
        """Build a ``server_vad`` configuration.

        Args:
            threshold: Sensitivity in ``[-1.0, 1.0]``. Defaults to ``0.5``.
            silence_duration_ms: Trailing silence in milliseconds, from ``200``
                to ``6000``. Defaults to ``800``.
            idle_timeout_ms: Silence after playback, from ``5000`` to ``30000``
                milliseconds, after which Qwen3.5 speaks again. Qwen leaves
                this feature off unless a value is provided. Only
                ``qwen3.5-omni-plus-realtime`` and
                ``qwen3.5-omni-flash-realtime`` accept this field.
        """
        options: dict[str, Any] = {
            "type": "server_vad",
            "threshold": threshold,
            "silence_duration_ms": silence_duration_ms,
        }
        if idle_timeout_ms is not None:
            options["idle_timeout_ms"] = idle_timeout_ms
        _check_threshold(options)
        _check_silence_duration(options)
        if idle_timeout_ms is not None and (
            isinstance(idle_timeout_ms, bool)
            or not isinstance(idle_timeout_ms, int)
            or not 5000 <= idle_timeout_ms <= 30000
        ):
            raise ValueError(
                "idle_timeout_ms must be an integer between 5000 and 30000"
            )
        return cast(_OmniServerVad, options)

    @staticmethod
    def semantic_vad(
        *,
        threshold: float = 0.5,
        silence_duration_ms: int = 800,
    ) -> _OmniSemanticVad:
        """Build a ``semantic_vad`` configuration.

        Args:
            threshold: Sensitivity in ``[-1.0, 1.0]``. Defaults to ``0.5``.
            silence_duration_ms: Trailing silence in milliseconds, from ``200``
                to ``6000``. Defaults to ``800``.
        """
        return cast(
            _OmniSemanticVad,
            QwenOmniTurnDetection._semantic(
                "semantic_vad", threshold, silence_duration_ms
            ),
        )

    @staticmethod
    def semantic_vad_ec(
        *,
        threshold: float = 0.5,
        silence_duration_ms: int = 800,
    ) -> _OmniSemanticVadEc:
        """Build a ``semantic_vad_ec`` configuration for Qwen3.8.

        Args:
            threshold: Sensitivity in ``[-1.0, 1.0]``. Defaults to ``0.5``.
            silence_duration_ms: Trailing silence in milliseconds, from ``200``
                to ``6000``. Defaults to ``800``.
        """
        return cast(
            _OmniSemanticVadEc,
            QwenOmniTurnDetection._semantic(
                "semantic_vad_ec", threshold, silence_duration_ms
            ),
        )

    @staticmethod
    def _semantic(
        kind: Literal["semantic_vad", "semantic_vad_ec"],
        threshold: float,
        silence_duration_ms: int,
    ) -> dict[str, Any]:
        options: dict[str, Any] = {
            "type": kind,
            "threshold": threshold,
            "silence_duration_ms": silence_duration_ms,
        }
        _check_threshold(options)
        _check_silence_duration(options)
        return options


_OMNI_IMAGE_INTERVAL_S: float = 1.0
_OMNI_MAX_JPEG_BYTES: int = 190 * 1024
_OMNI_MAX_IMAGE_EDGE: int = 1280


def _omni_audio_config(voice: str) -> dict[str, Any]:
    """Build the nested audio block required by current Qwen-Omni sessions."""
    return {
        "input": {
            "format": {
                "type": "pcm",
                "sample_rate": INPUT_SAMPLE_RATE,
                "sample_format": "s16le",
                "channels": 1,
                "packing": "interleaved",
                "channel_layout": "mono",
            }
        },
        "output": {
            "voice": voice,
            "format": {"type": "pcm", "sample_rate": OUTPUT_SAMPLE_RATE},
        },
    }


def _even_edge(value: int) -> int:
    return max(2, value // 2 * 2)


def _encode_mjpeg(frame: av.VideoFrame, qscale: int) -> bytes:
    codec = av.CodecContext.create("mjpeg", "w")
    codec.width = frame.width
    codec.height = frame.height
    codec.pix_fmt = "yuvj420p"
    codec.options = {"qscale": str(qscale)}
    packets = list(codec.encode(frame)) + list(codec.encode(None))
    return b"".join(bytes(packet) for packet in packets)


def _encode_omni_jpeg(frame: rtc.VideoFrame) -> bytes:
    """Encode a LiveKit frame as a JPEG small enough for Qwen-Omni."""
    source = frame
    if frame.type != rtc.VideoBufferType.RGB24:
        source = frame.convert(rtc.VideoBufferType.RGB24)
    pixels = np.frombuffer(source.data, dtype=np.uint8)
    array = pixels.reshape(source.height, source.width, 3).copy()
    video = av.VideoFrame.from_ndarray(array, format="rgb24")
    scale = min(1.0, _OMNI_MAX_IMAGE_EDGE / max(video.width, video.height))
    width = _even_edge(int(video.width * scale))
    height = _even_edge(int(video.height * scale))
    yuv = video.reformat(format="yuvj420p", width=width, height=height)
    encoded = _encode_mjpeg(yuv, 8)
    if len(encoded) <= _OMNI_MAX_JPEG_BYTES:
        return encoded
    for qscale in (15, 31):
        encoded = _encode_mjpeg(yuv, qscale)
        if len(encoded) <= _OMNI_MAX_JPEG_BYTES:
            return encoded
    smaller = video.reformat(
        format="yuvj420p",
        width=_even_edge(width // 2),
        height=_even_edge(height // 2),
    )
    return _encode_mjpeg(smaller, 31)


class _QwenOmniFamily(_RealtimeFamily):
    """Qwen-Omni Realtime session shape: nested audio config and image input."""

    supported_models = frozenset(
        {
            "qwen3.8-omni-flash-realtime",
            "qwen3.5-omni-plus-realtime",
            "qwen3.5-omni-flash-realtime",
        }
    )

    def resolve_voice(self, model: str, voice: NotGivenOr[str]) -> str:
        del model
        resolved = DEFAULT_OMNI_VOICE if not is_given(voice) else voice
        if not isinstance(resolved, str) or not resolved.strip():
            raise ValueError("Qwen Omni voice must be a non-empty string")
        return resolved

    def resolve_turn_detection(
        self,
        model: str,
        turn_detection: NotGivenOr[TurnDetection],
    ) -> TurnDetection:
        return _resolve_omni_turn_detection(model, turn_detection)

    def session_fields(self, opts: _RealtimeOptions) -> dict[str, Any]:
        return {
            "modalities": opts.modalities,
            "turn_detection": copy.deepcopy(opts.turn_detection),
            "audio": _omni_audio_config(opts.voice),
        }

    def voice_changes(self, voice: str) -> dict[str, Any]:
        return {"audio": _omni_audio_config(voice)}

    def history_changes(self, max_history_turns: int) -> dict[str, Any]:
        del max_history_turns
        logger.warning("Qwen Omni Realtime ignores max_history_turns")
        return {}

    def push_video(self, session: RealtimeSession, frame: rtc.VideoFrame) -> None:
        if not session._audio_appended:
            logger.warning("Qwen Omni Realtime requires audio before the first image")
            return
        now = time.monotonic()
        if now - session._last_image_at < _OMNI_IMAGE_INTERVAL_S:
            return
        image = _encode_omni_jpeg(frame)
        session._last_image_at = now
        session.send_event(
            {
                "type": "input_image_buffer.append",
                "image": base64.b64encode(image).decode("ascii"),
            }
        )

    def check_endpoint(self, model: str, base_url: str) -> None:
        if model == "qwen3.8-omni-flash-realtime" and base_url == DEFAULT_BASE_URL:
            logger.warning(
                "qwen3.8-omni-flash-realtime requires a workspace endpoint; "
                "pass workspace_id or base_url"
            )


_OMNI_FAMILY = _QwenOmniFamily()


class QwenOmniRealtime(_QwenRealtimeModel):
    """Alibaba Cloud Qwen-Omni Realtime model.

    The model uses DashScope's Realtime WebSocket protocol and accepts both
    16 kHz mono PCM16 audio and camera frames. Generated audio is 24 kHz mono
    PCM16. ``qwen3.8-omni-flash-realtime`` requires a workspace endpoint.
    """

    def __init__(
        self,
        *,
        model: QwenOmniRealtimeModelName = DEFAULT_OMNI_MODEL,
        voice: NotGivenOr[str] = NOT_GIVEN,
        modalities: NotGivenOr[Modalities] = NOT_GIVEN,
        turn_detection: NotGivenOr[QwenOmniTurnDetectionConfig] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice | None] = NOT_GIVEN,
        api_key: str | None = None,
        base_url: NotGivenOr[str] = NOT_GIVEN,
        workspace_id: str | None = None,
        region: Region = "cn-beijing",
        max_history_turns: int = 50,
        data_inspection: bool = False,
        http_session: aiohttp.ClientSession | None = None,
        max_session_duration: NotGivenOr[float | None] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> None:
        """Create a Qwen-Omni Realtime model adapter.

        Args:
            model (QwenOmniRealtimeModelName): Qwen-Omni Realtime model.
                Defaults to ``"qwen3.5-omni-flash-realtime"``.
                ``"qwen3.8-omni-flash-realtime"`` requires a workspace endpoint
                via ``workspace_id`` or ``base_url``.
            voice (NotGivenOr[str]): Output voice. Defaults to ``"Tina"``.
                Qwen only applies the voice in the first ``session.update``
                event.
            modalities (NotGivenOr[Modalities]): Response modalities. Supported
                values are ``["text"]`` and a two-element list containing both
                ``"text"`` and ``"audio"``. Defaults to ``["text", "audio"]``.
            turn_detection (NotGivenOr[QwenOmniTurnDetectionConfig]): End-of-turn
                detection. Use :meth:`QwenOmniTurnDetection.server_vad`,
                :meth:`QwenOmniTurnDetection.semantic_vad`, or
                :meth:`QwenOmniTurnDetection.semantic_vad_ec`. Pass ``None``
                for manual push-to-talk. Defaults to ``server_vad``.
                ``semantic_vad_ec`` is only accepted by
                ``qwen3.8-omni-flash-realtime``. ``idle_timeout_ms`` is only
                accepted by the Qwen3.5 models with ``server_vad``.
            tool_choice (NotGivenOr[llm.ToolChoice | None]): LiveKit tool
                selection preference retained for API compatibility. Qwen
                currently chooses tools automatically; forced or named choices
                are logged and ignored.
            api_key (str | None): DashScope credential used as the WebSocket
                Bearer token. When omitted, ``DASHSCOPE_API_KEY`` is read from
                the environment. The constructor raises ``ValueError`` if
                neither source provides a key.
            base_url (NotGivenOr[str]): Complete DashScope Realtime WebSocket
                endpoint without a required ``model`` query parameter. The
                adapter adds or replaces that query parameter. This option takes
                precedence over ``workspace_id`` and
                ``DASHSCOPE_REALTIME_BASE_URL``.
            workspace_id (str | None): Model Studio workspace ID used to build
                the recommended workspace-specific endpoint. It is ignored when
                ``base_url`` is explicitly provided.
            region (Region): Region used with ``workspace_id``. Supported values
                are ``"cn-beijing"`` and ``"ap-southeast-1"``. Defaults to
                ``"cn-beijing"``.
            max_history_turns (int): Accepted for constructor compatibility and
                stored locally. Qwen-Omni ignores this value and it is not sent
                in ``session.update``.
            data_inspection (bool): Whether DashScope content inspection remains
                enabled. When ``False`` (default), the adapter sends
                ``x-dashscope-dataInspection: disable`` during the WebSocket
                handshake. Set to ``True`` to omit that opt-out header.
            http_session (aiohttp.ClientSession | None): Optional caller-owned
                HTTP session. Reusing a session shares connection pools. The
                adapter never closes a caller-owned session; when omitted, it
                uses LiveKit's HTTP context or creates and owns a new session.
            max_session_duration (NotGivenOr[float | None]): Seconds before the
                adapter gracefully recycles the WebSocket. Defaults to 20
                minutes. Pass ``None`` to disable proactive recycling.
            conn_options (APIConnectOptions): LiveKit connection timeout, retry
                count, and retry interval policy. Defaults to
                ``DEFAULT_API_CONNECT_OPTIONS``.

        Raises:
            ValueError: If credentials are missing, ``model`` or ``voice`` is
                unsupported, ``turn_detection`` uses Smart Turn, ``modalities``
                is invalid, or ``max_history_turns`` is outside ``[1, 50]``.
        """
        super().__init__(
            family=_OMNI_FAMILY,
            model=model,
            voice=voice,
            modalities=modalities,
            turn_detection=turn_detection,
            tool_choice=tool_choice,
            api_key=api_key,
            base_url=base_url,
            workspace_id=workspace_id,
            region=region,
            max_history_turns=max_history_turns,
            data_inspection=data_inspection,
            http_session=http_session,
            max_session_duration=max_session_duration,
            conn_options=conn_options,
        )
