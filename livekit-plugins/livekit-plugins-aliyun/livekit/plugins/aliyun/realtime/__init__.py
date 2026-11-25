from __future__ import annotations

import warnings

import aiohttp

from livekit.agents import llm
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    APIConnectOptions,
    NotGivenOr,
)

from ..log import logger
from ._base import (
    DEFAULT_MODEL,
    ClonedVoiceId,
    Modalities,
    RealtimeModelName,
    RealtimeSession,
    Region,
    SemanticVadOptions,
    ServerVadOptions,
    SmartTurnOptions,
    SystemVoice,
    TurnDetection,
    Voice,
    _QwenRealtimeModel,
    _build_say_instruction as _build_say_instruction,
    _estimate_response_cost_cny as _estimate_response_cost_cny,
    _livekit_item_to_qwen_item as _livekit_item_to_qwen_item,
    _normalize_qwen_tool_schema as _normalize_qwen_tool_schema,
    _process_base_url as _process_base_url,
    _qwen_item_to_livekit_item as _qwen_item_to_livekit_item,
)
from .qwen_audio_realtime import (
    QwenAudioRealtime,
    QwenAudioTurnDetection,
    _AUDIO_FAMILY,
)
from .qwen_omini_realtime import (
    QwenOmniRealtime,
    QwenOmniTurnDetection,
    _OMNI_FAMILY,
)

_REMOVED_IN_FUTURE = (
    "aliyun.RealtimeModel is deprecated and will be removed in a future release; "
    "use aliyun.QwenAudioRealtime or aliyun.QwenOmniRealtime"
)


class RealtimeModel(_QwenRealtimeModel):
    """Deprecated adapter that accepts both Qwen Audio and Qwen-Omni models.

    This class remains so existing ``aliyun.RealtimeModel(...)`` calls keep
    working. Construction reports that the class will be removed. Pass a Qwen
    Audio model to :class:`QwenAudioRealtime`, or a Qwen-Omni model to
    :class:`QwenOmniRealtime`.
    """

    def __init__(
        self,
        *,
        model: RealtimeModelName = DEFAULT_MODEL,
        voice: NotGivenOr[Voice | str] = NOT_GIVEN,
        modalities: NotGivenOr[Modalities] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetection] = NOT_GIVEN,
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
        """Create a Qwen Audio or Qwen-Omni session through the deprecated API.

        Args:
            model (RealtimeModelName): Qwen Audio or Qwen-Omni Realtime model.
                Audio defaults to ``"qwen-audio-3.0-realtime-plus"``.
            voice (NotGivenOr[Voice | str]): Built-in voice, cloned voice ID, or
                an Omni voice such as ``"Tina"``.
            modalities (NotGivenOr[Modalities]): ``["text"]`` or
                ``["text", "audio"]``. Defaults to ``["text", "audio"]``.
            turn_detection (NotGivenOr[TurnDetection]): Audio accepts
                :class:`QwenAudioTurnDetection` modes. Omni accepts
                :class:`QwenOmniTurnDetection` modes. ``None`` selects manual
                push-to-talk.
            tool_choice: LiveKit tool preference. Qwen honors automatic choice.
            api_key (str | None): DashScope API key, or ``DASHSCOPE_API_KEY``.
            base_url (NotGivenOr[str]): Realtime WebSocket endpoint. Overrides
                ``workspace_id``.
            workspace_id (str | None): Model Studio workspace used to build the
                endpoint. Required for ``qwen3.8-omni-flash-realtime`` unless
                ``base_url`` is set.
            region (Region): Region paired with ``workspace_id``.
            max_history_turns (int): Audio history limit from 1 to 50. Omni
                ignores it.
            data_inspection (bool): Keep DashScope content inspection enabled.
            http_session: Optional caller-owned ``aiohttp.ClientSession``.
            max_session_duration (NotGivenOr[float | None]): Seconds before the
                WebSocket is recycled. Defaults to 20 minutes.
            conn_options (APIConnectOptions): LiveKit connect and retry policy.

        Raises:
            ValueError: If credentials, ``model``, ``voice``, ``modalities``, or
                ``max_history_turns`` are invalid for the selected model.
        """
        warnings.warn(_REMOVED_IN_FUTURE, DeprecationWarning, stacklevel=2)
        logger.warning(_REMOVED_IN_FUTURE)
        if model in _OMNI_FAMILY.supported_models:
            family = _OMNI_FAMILY
        elif model in _AUDIO_FAMILY.supported_models:
            family = _AUDIO_FAMILY
        else:
            choices = ", ".join(
                sorted(_AUDIO_FAMILY.supported_models | _OMNI_FAMILY.supported_models)
            )
            raise ValueError(
                f"unsupported Qwen Audio Realtime model {model!r}; use {choices}"
            )
        _QwenRealtimeModel.__init__(
            self,
            family=family,
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


__all__ = [
    "ClonedVoiceId",
    "Modalities",
    "QwenAudioRealtime",
    "QwenAudioTurnDetection",
    "QwenOmniRealtime",
    "QwenOmniTurnDetection",
    "RealtimeModel",
    "RealtimeModelName",
    "RealtimeSession",
    "Region",
    "SemanticVadOptions",
    "ServerVadOptions",
    "SmartTurnOptions",
    "SystemVoice",
    "TurnDetection",
    "Voice",
]
