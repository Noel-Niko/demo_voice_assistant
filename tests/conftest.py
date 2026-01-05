import os
import tempfile

import pytest


@pytest.fixture(scope="session", autouse=True)
def disable_external_tts_by_default():
    """Disable real TTS usage during tests to avoid network calls.

    Individual tests can override by setting TTS_ENABLED=true via monkeypatch.
    """
    os.environ.setdefault("TTS_ENABLED", "false")
    # Also nudge Google auth to avoid ADC probing in any stray initializations
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")


@pytest.fixture(scope="session", autouse=True)
def provide_dummy_openai_key():
    """Ensure routes /config can build OpenAI config without error."""
    os.environ.setdefault("OPENAI_API_KEY", "test-api-key")


@pytest.fixture(autouse=True)
def stub_google_service_account(monkeypatch):
    """Stub Google service account loading to avoid JSONDecodeError.

    Tests set GOOGLE_APPLICATION_CREDENTIALS to /dev/null to avoid ADC probing.
    The provider attempts to load that path as a service account file, which
    would raise JSONDecodeError. We replace the loader with a harmless stub and
    also stub ADC discovery just in case code paths use it.
    """
    # Stub explicit service account file loading
    try:
        import google.oauth2.service_account as sa  # type: ignore

        class _DummyCreds:
            service_account_email = "test@example.com"

        monkeypatch.setattr(
            sa.Credentials,
            "from_service_account_file",
            lambda path: _DummyCreds(),
            raising=True,
        )
    except Exception:
        # If the module isn't available in the environment, nothing to stub
        pass

    # Stub ADC discovery to avoid real probing if used
    try:
        import google.auth as gauth  # type: ignore

        class _AdcCreds:
            service_account_email = "adc@example.com"

        monkeypatch.setattr(gauth, "default", lambda: (_AdcCreds(), "test-project"), raising=True)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def isolate_performance_metrics(monkeypatch):
    """Isolate performance metrics persistence to a temp file for all tests.

    Avoids JSONDecodeError from any pre-existing repo-level metrics files.
    """
    # Create a dedicated temporary file path (do not create the file so loader returns defaults)
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp_path = tmp.name
    tmp.close()

    import src.llm.performance_metrics as pm

    # Patch the persistence class used by the module to point at our temp file
    orig_cls = pm.PerformanceMetricsPersistence
    monkeypatch.setattr(
        pm,
        "PerformanceMetricsPersistence",
        lambda metrics_file="performance_metrics.json": orig_cls(tmp_path),
        raising=True,
    )

    # Reset global singletons to ensure the patched persistence is used
    pm._global_metrics = None
    pm._metrics_persistence = None
