from livekit.plugins.aliyun.tts import TTS


def test_tts_defaults_to_cross_region_cosyvoice_pair() -> None:
    tts = TTS(api_key="test")

    assert tts._opts.model == "cosyvoice-v3-flash"
    assert tts._opts.voice == "longanyang"
