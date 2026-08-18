"""Custom provider profile.

Covers any endpoint registered as provider="custom" — a user-supplied
OpenAI-compatible base URL. Key quirk:
  - reasoning_config disabled → top-level reasoning_effort="none"
  - reasoning_config enabled + effort → top-level reasoning_effort
    (unset omits it so the endpoint's server default applies)
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CustomProfile(ProviderProfile):
    """User-configured OpenAI-compatible endpoint."""

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        **ctx: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        # Reasoning / thinking control for custom OpenAI-compatible endpoints.
        #
        #   - disabled  → top-level reasoning_effort="none"
        #   - enabled + effort set → TOP-LEVEL reasoning_effort string, the
        #     native OpenAI-compatible reasoning format.
        #   - enabled + no effort  → omit it, so the endpoint applies its own
        #     server-side default (do NOT force a level the user didn't pick).
        if reasoning_config and isinstance(reasoning_config, dict):
            _effort = (reasoning_config.get("effort") or "").strip().lower()
            _enabled = reasoning_config.get("enabled", True)
            if _effort == "none" or _enabled is False:
                top_level["reasoning_effort"] = "none"
            elif _effort:
                top_level["reasoning_effort"] = _effort

        return extra_body, top_level

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """base_url is user-configured; fetch only if set."""
        if not (base_url or self.base_url):
            return None
        return super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)


custom = CustomProfile(
    name="custom",
    aliases=(),
    env_vars=(),  # No fixed key — custom endpoint
    base_url="",  # User-configured
    # Only a floor used when the user hasn't set model.max_tokens — they can
    # override per-model — so we set it generously rather than lowballing it.
    default_max_tokens=65536,
)

register_provider(custom)
