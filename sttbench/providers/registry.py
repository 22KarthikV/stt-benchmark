"""Provider lookup. A missing key means "skipped", never a silent score of 0."""

from collections.abc import Mapping

from .base import Provider
from .fake import FakeProvider

# name -> required environment variable (None: no key needed)
KEY_VARS: dict[str, str | None] = {
    "fake": None,
    "assemblyai": "ASSEMBLYAI_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
    "vosk": "VOSK_MODEL_PATH",
}


def _construct(name: str, env: Mapping[str, str]) -> Provider:
    if name == "fake":
        return FakeProvider()
    if name == "assemblyai":
        from .assemblyai import AssemblyAIProvider

        return AssemblyAIProvider.from_env(env)
    if name == "deepgram":
        from .deepgram import DeepgramProvider

        return DeepgramProvider.from_env(env)
    if name == "vosk":
        from .vosk_provider import VoskProvider

        return VoskProvider.from_env(env)
    raise ValueError(f"unknown provider: {name}")


def build_providers(
    names: list[str], env: Mapping[str, str]
) -> tuple[dict[str, Provider], dict[str, str]]:
    """Return (providers that can run, {name: reason} for those skipped)."""
    providers: dict[str, Provider] = {}
    skipped: dict[str, str] = {}
    for name in names:
        if name not in KEY_VARS:
            raise ValueError(f"unknown provider: {name} (known: {', '.join(KEY_VARS)})")
        var = KEY_VARS[name]
        if var and not env.get(var):
            what = "model path" if name == "vosk" else "key"
            skipped[name] = f"skipped: no {what} ({var})"
            continue
        providers[name] = _construct(name, env)
    return providers, skipped
