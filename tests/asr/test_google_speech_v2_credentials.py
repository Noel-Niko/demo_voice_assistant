import sys
import types


class DummySpeechClient:
    def __init__(self, credentials=None):
        self.credentials = credentials


def test_initialize_uses_standard_env_when_present(monkeypatch):
    # Arrange: set env var and stub Google modules
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/fake.json")

    # Fake google package hierarchy and service_account module
    google_mod = types.ModuleType("google")
    oauth2_mod = types.ModuleType("google.oauth2")
    sa_mod = types.ModuleType("google.oauth2.service_account")

    class _Creds:
        @classmethod
        def from_service_account_file(cls, path):
            # Assert the path is what we passed via env
            assert path == "/tmp/fake.json"
            return "CREDS"

    sa_mod.Credentials = _Creds
    setattr(oauth2_mod, "service_account", sa_mod)
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_mod)
    monkeypatch.setitem(sys.modules, "google.oauth2.service_account", sa_mod)

    # Patch SpeechClient in our module
    import src.asr.google_speech_v2 as gsv2

    captured = {}

    class _Client:
        def __init__(self, credentials=None):
            captured["credentials"] = credentials

    monkeypatch.setattr(gsv2, "SpeechClient", _Client, raising=True)

    # Act
    provider = gsv2.GoogleSpeechV2Provider()
    provider.initialize(model="latest_long", project_id="proj")

    # Assert: credentials were passed to SpeechClient
    assert captured.get("credentials") == "CREDS"


def test_initialize_falls_back_to_adc_when_env_missing(monkeypatch):
    # Ensure env var not set
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    import src.asr.google_speech_v2 as gsv2

    called = {}

    class _Client:
        def __init__(self, credentials=None):
            called["credentials"] = credentials

    monkeypatch.setattr(gsv2, "SpeechClient", _Client, raising=True)

    # Act
    provider = gsv2.GoogleSpeechV2Provider()
    provider.initialize(model="latest_long", project_id="proj")

    # Assert: called without credentials (ADC path)
    assert called.get("credentials") is None
