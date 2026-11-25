from __future__ import annotations

import copy
from typing import Any, Literal, cast

import aiohttp
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
    DEFAULT_AUDIO_31_VOICE,
    DEFAULT_MODEL,
    DEFAULT_TURN_DETECTION,
    DEFAULT_VOICE,
    Modalities,
    RealtimeSession,
    Region,
    TurnDetection,
    Voice,
    _QwenRealtimeModel,
    _RealtimeFamily,
    _RealtimeOptions,
)

QwenAudioRealtimeModelName = Literal[
    "qwen-audio-3.1-realtime-plus",
    "qwen-audio-3.1-realtime-flash",
    "qwen-audio-3.0-realtime-plus",
    "qwen-audio-3.0-realtime-flash",
]
"""Qwen Audio Realtime model identifiers."""

_AUDIO_VOICES_30 = frozenset(
    {
        "longanqian",
        "longanlingxin",
        "longanlingxi",
        "longanxiaoxin",
        "longanlufeng",
    }
)


def _validate_audio_voice(model: str, voice: str) -> None:
    """Validate an Audio-series voice against its model generation.

    3.1 and 3.0 system voices are not interchangeable. The 3.1 system set
    grows over time, so any ``*_v3.1`` id or
    ``qwen-audio-3.1-tts-<locale>-<name>`` id is accepted.
    """
    if model.startswith("qwen-audio-3.1-"):
        if voice.endswith("_v3.1") or voice.startswith("qwen-audio-3.1-tts-"):
            return
    elif voice in _AUDIO_VOICES_30:
        return
    if voice.startswith(f"{model}-"):
        return
    raise ValueError(
        f"unsupported voice {voice!r} for model {model!r}; use a built-in "
        f"SystemVoice of the same model generation or a ClonedVoiceId "
        f"created with target_model={model!r}"
    )


class _AudioServerVadRequired(TypedDict):
    type: Literal["server_vad"]


class _AudioSmartTurnRequired(TypedDict):
    type: Literal["smart_turn", "smart_turn_v2"]


class _AudioServerVad(_AudioServerVadRequired, total=False):
    threshold: float
    silence_duration_ms: int


class _AudioSmartTurn(_AudioSmartTurnRequired, total=False):
    threshold: float
    voiceprint_audio_urls: list[str]


QwenAudioTurnDetectionConfig = _AudioServerVad | _AudioSmartTurn | None
"""Qwen Audio turn detection. ``None`` selects manual push-to-talk."""

_AUDIO_TURN_TYPES = frozenset({"server_vad", "smart_turn", "smart_turn_v2"})


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


def _check_voiceprints(options: dict[str, Any]) -> None:
    if "voiceprint_audio_urls" not in options:
        return
    urls = options["voiceprint_audio_urls"]
    if (
        not isinstance(urls, list)
        or len(urls) > 5
        or any(not isinstance(url, str) or not url.strip() for url in urls)
    ):
        raise ValueError(
            "voiceprint_audio_urls must contain at most 5 non-empty URL strings"
        )


def _resolve_audio_turn_detection(
    turn_detection: NotGivenOr[QwenAudioTurnDetectionConfig],
) -> QwenAudioTurnDetectionConfig:
    if not is_given(turn_detection):
        return copy.deepcopy(DEFAULT_TURN_DETECTION)
    if turn_detection is None:
        return None
    resolved = copy.deepcopy(turn_detection)
    kind = resolved.get("type")
    if kind not in _AUDIO_TURN_TYPES:
        raise ValueError(
            "Qwen Audio Realtime turn detection must be server_vad, "
            f"smart_turn, smart_turn_v2, or None; got {kind!r}"
        )
    _check_threshold(resolved)
    if kind == "server_vad":
        _check_silence_duration(resolved)
        if "voiceprint_audio_urls" in resolved:
            raise ValueError(
                "voiceprint_audio_urls is only valid for smart_turn and smart_turn_v2"
            )
    else:
        if "silence_duration_ms" in resolved:
            raise ValueError("silence_duration_ms is only valid for server_vad")
        _check_voiceprints(resolved)
    return cast(QwenAudioTurnDetectionConfig, resolved)


class QwenAudioTurnDetection:
    """Qwen Audio turn-detection builders.

    ``server_vad`` ends a turn after silence. ``smart_turn`` and
    ``smart_turn_v2`` end a turn from acoustics and semantics.
    ``smart_turn_v2`` is the recommended Smart Turn mode. Pass ``None`` to
    :class:`QwenAudioRealtime` for manual push-to-talk.
    """

    @staticmethod
    def server_vad(
        *,
        threshold: float = 0.5,
        silence_duration_ms: int = 800,
    ) -> _AudioServerVad:
        """Build a ``server_vad`` configuration.

        Args:
            threshold: Sensitivity in ``[-1.0, 1.0]``. Lower values are more
                sensitive. Defaults to ``0.5``.
            silence_duration_ms: Trailing silence in milliseconds, from ``200``
                to ``6000``. Defaults to ``800``.
        """
        options: dict[str, Any] = {
            "type": "server_vad",
            "threshold": threshold,
            "silence_duration_ms": silence_duration_ms,
        }
        return cast(_AudioServerVad, _resolve_audio_turn_detection(options))

    @staticmethod
    def smart_turn(
        *,
        threshold: float = 0.5,
        voiceprint_audio_urls: tuple[str, ...] = (),
    ) -> _AudioSmartTurn:
        """Build a ``smart_turn`` configuration.

        Args:
            threshold: Sensitivity in ``[-1.0, 1.0]``. Defaults to ``0.5``.
            voiceprint_audio_urls: Up to five public 16 kHz PCM or WAV URLs of
                the target speaker. Defaults to no voiceprints.
        """
        return QwenAudioTurnDetection._smart_turn(
            "smart_turn",
            threshold=threshold,
            voiceprint_audio_urls=voiceprint_audio_urls,
        )

    @staticmethod
    def smart_turn_v2(
        *,
        threshold: float = 0.5,
        voiceprint_audio_urls: tuple[str, ...] = (),
    ) -> _AudioSmartTurn:
        """Build a ``smart_turn_v2`` configuration.

        Args:
            threshold: Sensitivity in ``[-1.0, 1.0]``. Defaults to ``0.5``.
            voiceprint_audio_urls: Up to five public 16 kHz PCM or WAV URLs of
                the target speaker. Defaults to no voiceprints.
        """
        return QwenAudioTurnDetection._smart_turn(
            "smart_turn_v2",
            threshold=threshold,
            voiceprint_audio_urls=voiceprint_audio_urls,
        )

    @staticmethod
    def _smart_turn(
        kind: Literal["smart_turn", "smart_turn_v2"],
        *,
        threshold: float,
        voiceprint_audio_urls: tuple[str, ...],
    ) -> _AudioSmartTurn:
        options: dict[str, Any] = {"type": kind, "threshold": threshold}
        if voiceprint_audio_urls:
            options["voiceprint_audio_urls"] = list(voiceprint_audio_urls)
        return cast(_AudioSmartTurn, _resolve_audio_turn_detection(options))


class _QwenAudioFamily(_RealtimeFamily):
    """Qwen Audio Realtime session shape: flat voice and PCM format fields."""

    supported_models = frozenset(
        {
            "qwen-audio-3.1-realtime-plus",
            "qwen-audio-3.1-realtime-flash",
            "qwen-audio-3.0-realtime-plus",
            "qwen-audio-3.0-realtime-flash",
        }
    )

    def resolve_voice(self, model: str, voice: NotGivenOr[str]) -> str:
        if not is_given(voice):
            resolved = (
                DEFAULT_AUDIO_31_VOICE
                if model.startswith("qwen-audio-3.1-")
                else DEFAULT_VOICE
            )
        else:
            resolved = voice
        _validate_audio_voice(model, resolved)
        return resolved

    def resolve_turn_detection(
        self,
        model: str,
        turn_detection: NotGivenOr[TurnDetection],
    ) -> TurnDetection:
        del model
        return _resolve_audio_turn_detection(turn_detection)

    def session_fields(self, opts: _RealtimeOptions) -> dict[str, Any]:
        return {
            "modalities": opts.modalities,
            "voice": opts.voice,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "turn_detection": copy.deepcopy(opts.turn_detection),
            "max_history_turns": opts.max_history_turns,
        }

    def voice_changes(self, voice: str) -> dict[str, Any]:
        return {"voice": voice}

    def history_changes(self, max_history_turns: int) -> dict[str, Any]:
        return {"max_history_turns": max_history_turns}

    def push_video(self, session: RealtimeSession, frame: rtc.VideoFrame) -> None:
        del session, frame
        logger.warning("Qwen Audio Realtime does not support video input")


_AUDIO_FAMILY = _QwenAudioFamily()


class QwenAudioRealtime(_QwenRealtimeModel):
    """Alibaba Cloud Qwen Audio Realtime model.

    The model uses DashScope's OpenAI-like Realtime WebSocket protocol. Audio
    sent to Qwen is 16 kHz mono PCM16 and generated audio is 24 kHz mono PCM16.
    Qwen Audio does not accept camera frames.
    """

    def __init__(
        self,
        *,
        model: QwenAudioRealtimeModelName = DEFAULT_MODEL,
        voice: NotGivenOr[Voice | str] = NOT_GIVEN,
        modalities: NotGivenOr[Modalities] = NOT_GIVEN,
        turn_detection: NotGivenOr[QwenAudioTurnDetectionConfig] = NOT_GIVEN,
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
        """Create a Qwen Audio Realtime model adapter.

        Args:
            model (QwenAudioRealtimeModelName): Qwen Audio Realtime model.
                Defaults to ``"qwen-audio-3.0-realtime-plus"``.
            voice (NotGivenOr[Voice | str]): Built-in voice or a custom cloned
                voice ID. Audio 3.0 defaults to ``"longanqian"``. Audio 3.1
                defaults to ``"longanqian_v3.1"`` and accepts every ``*_v3.1``
                voice or a :class:`ClonedVoiceId` created for the same model.
                Qwen only applies the voice in the first ``session.update``
                event.
            modalities (NotGivenOr[Modalities]): Response modalities. Supported
                values are ``["text"]`` and a two-element list containing both
                ``"text"`` and ``"audio"``. Defaults to ``["text", "audio"]``.
            turn_detection (NotGivenOr[QwenAudioTurnDetectionConfig]): End-of-turn
                detection. Use :meth:`QwenAudioTurnDetection.server_vad`,
                :meth:`QwenAudioTurnDetection.smart_turn`, or
                :meth:`QwenAudioTurnDetection.smart_turn_v2`. Pass ``None`` for
                manual push-to-talk. Defaults to ``server_vad``.
            tool_choice (NotGivenOr[llm.ToolChoice | None]): LiveKit tool
                selection preference retained for API compatibility. Qwen Audio
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
            max_history_turns (int): Maximum number of question-answer turns
                Qwen includes in one inference request. Valid range: ``1`` to
                ``50``. Defaults to ``50``.
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
                unsupported, the cloned voice belongs to a different model,
                ``modalities`` is invalid, or ``max_history_turns`` is outside
                ``[1, 50]``.
        """
        super().__init__(
            family=_AUDIO_FAMILY,
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
