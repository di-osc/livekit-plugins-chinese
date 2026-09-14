import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from aiohttp import WSMessage, WSMsgType
from livekit.agents import APIConnectOptions, APIStatusError

from livekit.plugins.aliyun.tts import TTS


_FOUR_SENTENCES = (
    "欢迎来到本次面试，我是今天的面试官。"
    "接下来我会先介绍流程，请你认真听。"
    "第一部分是自我介绍，请控制在两分钟以内。"
    "第二部分是项目经历，请重点讲你负责的模块。"
)
_PCM_CHUNK = b"\x00\x00" * 4800  # 200ms of 24kHz mono PCM


class _FakeWS:
    def __init__(self, *, fail_task: bool = False) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self._fail_task = fail_task
        self._incoming: asyncio.Queue[WSMessage] = asyncio.Queue()

    async def send_json(self, data: dict) -> None:
        self.sent.append(data)
        action = data["header"]["action"]
        if action == "run-task":
            event = "task-failed" if self._fail_task else "task-started"
            await self._incoming.put(_text_message({"header": {"event": event}}))
        elif action == "continue-task":
            await self._incoming.put(
                WSMessage(type=WSMsgType.BINARY, data=_PCM_CHUNK, extra=None)
            )
        elif action == "finish-task":
            await self._incoming.put(
                _text_message({"header": {"event": "task-finished"}})
            )

    async def receive(self) -> WSMessage:
        return await self._incoming.get()

    async def close(self) -> None:
        self.closed = True


def _text_message(payload: dict) -> WSMessage:
    return WSMessage(type=WSMsgType.TEXT, data=json.dumps(payload), extra=None)


def _install_fake_ws(
    plugin: TTS, fake_ws: _FakeWS, monkeypatch: pytest.MonkeyPatch
) -> None:
    @asynccontextmanager
    async def fake_connection(*, timeout: float):
        yield fake_ws

    monkeypatch.setattr(plugin._pool, "connection", fake_connection)


def test_tts_defaults_to_cross_region_cosyvoice_pair() -> None:
    tts = TTS(api_key="test")

    assert tts._opts.model == "cosyvoice-v3-flash"
    assert tts._opts.voice == "longanyang"


def test_task_params_share_given_task_id() -> None:
    opts = TTS(api_key="test")._opts
    task_id = "task-123"

    run_params = opts.get_run_task_params(task_id=task_id)
    continue_params = opts.get_continue_task_params(text="你好。", task_id=task_id)
    finish_params = opts.get_finish_task_params(task_id=task_id)

    assert run_params["header"]["action"] == "run-task"
    assert continue_params["header"]["action"] == "continue-task"
    assert finish_params["header"]["action"] == "finish-task"
    assert run_params["header"]["task_id"] == task_id
    assert continue_params["header"]["task_id"] == task_id
    assert finish_params["header"]["task_id"] == task_id
    assert continue_params["payload"]["input"]["text"] == "你好。"


@pytest.mark.asyncio
async def test_stream_emits_single_segment_for_multi_sentence_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = TTS(api_key="test")
    fake_ws = _FakeWS()
    _install_fake_ws(plugin, fake_ws, monkeypatch)

    stream = plugin.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text(_FOUR_SENTENCES)
    stream.end_input()

    events = [event async for event in stream]
    segment_ids = {event.segment_id for event in events}
    actions = [message["header"]["action"] for message in fake_ws.sent]
    continue_messages = [
        message
        for message in fake_ws.sent
        if message["header"]["action"] == "continue-task"
    ]
    task_ids = {message["header"]["task_id"] for message in fake_ws.sent}

    assert events
    assert len(segment_ids) == 1
    assert actions[0] == "run-task"
    assert actions[-1] == "finish-task"
    assert len(continue_messages) == 4
    assert len(task_ids) == 1

    await stream.aclose()


@pytest.mark.asyncio
async def test_stream_raises_on_task_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = TTS(api_key="test")
    fake_ws = _FakeWS(fail_task=True)
    _install_fake_ws(plugin, fake_ws, monkeypatch)

    stream = plugin.stream(conn_options=APIConnectOptions(max_retry=0))
    stream.push_text(_FOUR_SENTENCES)
    stream.end_input()

    with pytest.raises(APIStatusError, match="tts task failed"):
        async for _ in stream:
            pass

    await stream.aclose()
