"""Per-provider model-selection wizard flows for ``pilotage setup`` / ``pilotage model``.

Extracted from ``pilotage_cli/main.py`` as part of the god-file decomposition
campaign (``~/.pilotage/plans/god-file-decomposition.md``, Phase 2 — splitting
main.py handler/flow bodies out of the module). These 18 ``_model_flow_*``
functions are the interactive provider-setup branches dispatched by
``select_provider_and_model`` (which stays in main.py).

Behavior-neutral: each function is lifted verbatim. ``select_provider_and_model``
in main.py re-imports them (``from pilotage_cli.model_setup_flows import *``-style
explicit import) so existing call sites — and test monkeypatches that target
``pilotage_cli.main._model_flow_*`` — keep resolving against main.py's namespace.

main.py-internal helpers the flows call (``_prompt_api_key``, ``_save_custom_provider``,
the reasoning-effort/stepfun/qwen helpers, …) are
imported lazily inside the flows (``from pilotage_cli.main import ...`` resolves at
call time, when main.py is fully loaded) so this module never imports
``pilotage_cli.main`` at import time -> no import cycle.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import urllib.parse

from pilotage_cli.config import clear_model_endpoint_credentials
from pilotage_cli.providers import custom_provider_slug


# AWS cross-region inference profile prefixes. Any geo-prefixed profile only
# routes from endpoints in its own geography, so the Bedrock picker must not
# offer (e.g.) us.* profiles to an eu-central-2 endpoint — selecting one
# produces a config AWS rejects regardless of credentials.
# global.* routes from everywhere. Full set per the AWS cross-region
# inference docs.
BEDROCK_GEO_PREFIXES = (
    "us.", "eu.", "ap.", "apac.", "jp.", "ca.", "sa.", "me.", "af.",
)


def bedrock_region_geo_prefix(region_name: str) -> str:
    """Map an AWS region name to its inference-profile geo prefix ('' = unknown)."""
    r = (region_name or "").lower()
    for geo, region_prefixes in (
        ("us.", ("us-", "us_gov")),
        ("eu.", ("eu-",)),
        ("ap.", ("ap-",)),
        ("ca.", ("ca-",)),
        ("sa.", ("sa-",)),
        ("me.", ("me-",)),
        ("af.", ("af-",)),
    ):
        if r.startswith(region_prefixes):
            return geo
    return ""


def bedrock_model_routable_from_region(model_id: str, region_name: str) -> bool:
    """True when *model_id* can be invoked from *region_name*'s endpoint.

    Bare foundation-model ids and ``global.*`` profiles route from anywhere.
    Geo-prefixed inference profiles (``us.*``, ``eu.*``, ...) only route from
    endpoints in their own geography. Unknown region shapes hide nothing.
    """
    mid = (model_id or "").lower()
    matched_geo = next((p for p in BEDROCK_GEO_PREFIXES if mid.startswith(p)), None)
    if matched_geo is None or mid.startswith("global."):
        return True
    geo = bedrock_region_geo_prefix(region_name)
    if not geo:
        return True
    if geo == "ap.":
        # Asia-Pacific regions can carry ap./apac./jp. profile spellings.
        return matched_geo in ("ap.", "apac.", "jp.")
    return matched_geo == geo


def _existing_api_key_for_model_flow(provider_id: str, pconfig) -> tuple[str, str]:
    """Resolve an existing wizard credential without changing its storage."""
    from pilotage_cli.auth import _resolve_api_key_provider_secret

    return _resolve_api_key_provider_secret(provider_id, pconfig)


def _prune_replaced_custom_model_config_credentials(
    base_url: str,
    *,
    provider_name: str = "",
) -> None:
    """Drop stale ``model_config`` credentials from inactive custom pools.

    ``model_config`` means "the credential currently stored under
    ``model.api_key``". After an explicit custom-endpoint switch, any old
    custom pool still carrying that source points at the previous endpoint and
    can be selected before the freshly saved config is tried.
    """
    try:
        from agent.credential_pool import (
            CUSTOM_POOL_PREFIX,
            get_custom_provider_pool_key,
        )
        from pilotage_cli.auth import read_credential_pool, write_credential_pool

        active_pool_key = get_custom_provider_pool_key(
            base_url,
            provider_name=provider_name or None,
        )
        if not active_pool_key:
            return
        pools = read_credential_pool(None)
        if not isinstance(pools, dict):
            return
        for pool_key, entries in pools.items():
            if (
                not isinstance(pool_key, str)
                or not pool_key.startswith(CUSTOM_POOL_PREFIX)
                or pool_key == active_pool_key
                or not isinstance(entries, list)
            ):
                continue
            retained = []
            removed_ids = []
            changed = False
            for entry in entries:
                if isinstance(entry, dict) and entry.get("source") == "model_config":
                    changed = True
                    entry_id = entry.get("id")
                    if entry_id:
                        removed_ids.append(str(entry_id))
                    continue
                retained.append(entry)
            if changed:
                write_credential_pool(pool_key, retained, removed_ids=removed_ids)
    except Exception:
        return


def _prompt_auth_credentials_choice(title: str) -> str:
    """Prompt for reuse / reauthenticate / cancel with the standard radio UI.

    Returns one of ``"use"``, ``"reauth"``, ``"cancel"``. Falls back to a
    numbered prompt when curses is unavailable (piped stdin, non-TTY).
    """
    choices = [
        "Use existing credentials",
        "Reauthenticate (new OAuth login)",
        "Cancel",
    ]
    try:
        from pilotage_cli.setup import _curses_prompt_choice

        idx = _curses_prompt_choice(title, choices, 0)
        if idx >= 0:
            print()
            return ("use", "reauth", "cancel")[idx]
    except Exception:
        pass

    print(title)
    for i, label in enumerate(choices, 1):
        marker = "→" if i == 1 else " "
        print(f"  {marker} {i}. {label}")
    print()
    try:
        choice = input("  Choice [1/2/3]: ").strip()
    except (KeyboardInterrupt, EOFError):
        choice = "1"

    if choice == "2":
        return "reauth"
    if choice == "3":
        return "cancel"
    return "use"


def _model_flow_openai_codex(config, current_model=""):
    """OpenAI Codex provider: ensure logged in, then pick model."""
    from pilotage_cli.auth import (
        get_codex_auth_status,
        _prompt_model_selection,
        _save_model_choice,
        _update_config_for_provider,
        _login_openai_codex,
        PROVIDER_REGISTRY,
        DEFAULT_CODEX_BASE_URL,
    )
    from pilotage_cli.codex_models import get_codex_model_ids

    status = get_codex_auth_status()
    if status.get("logged_in"):
        print("  OpenAI Codex credentials: ✓")
        print()
        choice = _prompt_auth_credentials_choice("OpenAI Codex credentials:")

        if choice == "reauth":
            print("Starting a fresh OpenAI Codex login...")
            print()
            try:
                mock_args = argparse.Namespace()
                _login_openai_codex(
                    mock_args,
                    PROVIDER_REGISTRY["openai-codex"],
                    force_new_login=True,
                )
            except SystemExit:
                print("Login cancelled or failed.")
                return
            except Exception as exc:
                print(f"Login failed: {exc}")
                return
            status = get_codex_auth_status()
            if not status.get("logged_in"):
                print("Login failed.")
                return
        elif choice == "cancel":
            return
    else:
        print("Not logged into OpenAI Codex. Starting login...")
        print()
        try:
            mock_args = argparse.Namespace()
            _login_openai_codex(mock_args, PROVIDER_REGISTRY["openai-codex"])
        except SystemExit:
            print("Login cancelled or failed.")
            return
        except Exception as exc:
            print(f"Login failed: {exc}")
            return

    _codex_token = None
    # Prefer credential pool (where `pilotage auth` stores device_code tokens),
    # fall back to legacy provider state.
    try:
        _codex_status = get_codex_auth_status()
        if _codex_status.get("logged_in"):
            _codex_token = _codex_status.get("api_key")
    except Exception:
        pass
    if not _codex_token:
        try:
            from pilotage_cli.auth import resolve_codex_runtime_credentials

            _codex_creds = resolve_codex_runtime_credentials()
            _codex_token = _codex_creds.get("api_key")
        except Exception:
            pass

    codex_models = get_codex_model_ids(access_token=_codex_token)

    selected = _prompt_model_selection(
        codex_models,
        current_model=current_model,
        confirm_provider="openai-codex",
        confirm_base_url=DEFAULT_CODEX_BASE_URL,
        confirm_api_key=_codex_token or "",
    )
    if selected:
        _save_model_choice(selected)
        _update_config_for_provider("openai-codex", DEFAULT_CODEX_BASE_URL)
        print(f"Default model set to: {selected} (via OpenAI Codex)")
    else:
        print("No change.")



def _model_flow_custom(config):
    """Custom endpoint: collect URL, API key, and model name.

    Automatically saves the endpoint to ``custom_providers`` in config.yaml
    so it appears in the provider menu on subsequent runs.
    """
    from pilotage_cli.main import _auto_provider_name, _prompt_custom_api_mode_selection, _save_custom_provider
    from pilotage_cli.auth import _save_model_choice, deactivate_provider
    from pilotage_cli.config import (
        custom_endpoint_key_env,
        get_env_value,
        load_config,
        save_config,
        save_env_value,
    )
    from pilotage_cli.secret_prompt import masked_secret_prompt

    current_url = get_env_value("OPENAI_BASE_URL") or ""
    current_key = get_env_value("OPENAI_API_KEY") or ""

    print("Custom OpenAI-compatible endpoint configuration:")
    if current_url:
        print(f"  Current URL: {current_url}")
    if current_key:
        print(f"  Current key: {current_key[:8]}...")
    print()

    try:
        base_url = input(
            f"API base URL [{current_url or 'e.g. https://api.example.com/v1'}]: "
        ).strip()
        api_key = masked_secret_prompt(
            f"API key [{current_key[:8] + '...' if current_key else 'optional'}]: "
        ).strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    if not base_url and not current_url:
        print("No URL provided. Cancelled.")
        return

    # Validate URL format
    effective_url = base_url or current_url
    if not effective_url.startswith(("http://", "https://")):
        print(f"Invalid URL: {effective_url} (must start with http:// or https://)")
        return

    effective_key = api_key or current_key

    # Hint: most local model servers (Ollama, vLLM, llama.cpp) require /v1
    # in the base URL for OpenAI-compatible chat completions.  Prompt the
    # user if the URL looks like a local server without /v1.
    _url_lower = effective_url.rstrip("/").lower()
    _looks_local = any(
        h in _url_lower
        for h in ("localhost", "127.0.0.1", "0.0.0.0", ":11434", ":8080", ":5000")
    )
    if _looks_local and not _url_lower.endswith("/v1"):
        print()
        print("  Hint: Did you mean to add /v1 at the end?")
        print("  Most local model servers (Ollama, vLLM, llama.cpp) require it.")
        print(f"  e.g. {effective_url.rstrip('/')}/v1")
        try:
            _add_v1 = input("  Add /v1? [Y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            _add_v1 = "n"
        if _add_v1 in {"", "y", "yes"}:
            effective_url = effective_url.rstrip("/") + "/v1"
            if base_url:
                base_url = effective_url
            print(f"  Updated URL: {effective_url}")
        print()

    from pilotage_cli.models import probe_api_models

    probe = probe_api_models(effective_key, effective_url)
    if probe.get("used_fallback") and probe.get("resolved_base_url"):
        print(
            f"Warning: endpoint verification worked at {probe['resolved_base_url']}/models, "
            f"not the exact URL you entered. Saving the working base URL instead."
        )
        effective_url = probe["resolved_base_url"]
        if base_url:
            base_url = effective_url
    elif probe.get("models") is not None:
        print(
            f"Verified endpoint via {probe.get('probed_url')} "
            f"({len(probe.get('models') or [])} model(s) visible)"
        )
    else:
        print(
            f"Warning: could not verify this endpoint via {probe.get('probed_url')}. "
            f"Pilotage will still save it."
        )
        if probe.get("suggested_base_url"):
            suggested = probe["suggested_base_url"]
            if suggested.endswith("/v1"):
                print(
                    f"  If this server expects /v1 in the path, try base URL: {suggested}"
                )
            else:
                print(f"  If /v1 should not be in the base URL, try: {suggested}")

    # Prompt for API compatibility mode explicitly so codex-compatible custom
    # providers don't silently fall back to chat_completions.
    current_model_cfg = config.get("model")
    current_api_mode = ""
    if isinstance(current_model_cfg, dict):
        current_api_mode = str(current_model_cfg.get("api_mode") or "").strip()
    api_mode = _prompt_custom_api_mode_selection(
        effective_url,
        current_api_mode=current_api_mode,
    )
    if api_mode:
        print(f"  API mode: {api_mode}")
    else:
        print("  API mode: auto-detect")

    # Select model — use probe results when available, fall back to manual input
    model_name = ""
    detected_models = probe.get("models") or []
    try:
        if len(detected_models) == 1:
            print(f"  Detected model: {detected_models[0]}")
            confirm = input("  Use this model? [Y/n]: ").strip().lower()
            if confirm in {"", "y", "yes"}:
                model_name = detected_models[0]
            else:
                model_name = input("Model name (e.g. gpt-4, llama-3-70b): ").strip()
        elif len(detected_models) > 1:
            print("  Available models:")
            for i, m in enumerate(detected_models, 1):
                print(f"    {i}. {m}")
            pick = input(
                f"  Select model [1-{len(detected_models)}] or type name: "
            ).strip()
            if pick.isdigit() and 1 <= int(pick) <= len(detected_models):
                model_name = detected_models[int(pick) - 1]
            elif pick:
                model_name = pick
        else:
            model_name = input("Model name (e.g. gpt-4, llama-3-70b): ").strip()

        context_length_str = input(
            "Context length in tokens [leave blank for auto-detect]: "
        ).strip()

        # Prompt for a display name — shown in the provider menu on future runs
        default_name = _auto_provider_name(effective_url)
        display_name = input(f"Display name [{default_name}]: ").strip() or default_name
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    context_length = None
    if context_length_str:
        try:
            context_length = int(
                context_length_str.replace(",", "")
                .replace("k", "000")
                .replace("K", "000")
            )
            if context_length <= 0:
                context_length = None
        except ValueError:
            print(f"Invalid context length: {context_length_str} — will auto-detect.")
            context_length = None

    # The key goes to.env and config.yaml only references it. Keyed
    # on host:port so two servers on one machine keep separate credentials.
    custom_key_env = ""
    if effective_key:
        _parsed = urllib.parse.urlparse(effective_url)
        _identity = _parsed.hostname or ""
        if _parsed.port:
            _identity = f"{_identity}_{_parsed.port}"
        custom_key_env = custom_endpoint_key_env(_identity)
        save_env_value(custom_key_env, effective_key)
        print(f"  API key saved to .env as {custom_key_env}")

    if model_name:
        _save_model_choice(model_name)

        # Update config and deactivate any OAuth provider
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = "custom"
        model["base_url"] = effective_url
        if custom_key_env:
            model["api_key"] = f"${{{custom_key_env}}}"
        if api_mode:
            model["api_mode"] = api_mode
        else:
            model.pop("api_mode", None)
        save_config(cfg)
        deactivate_provider()

        # Sync the caller's config dict so the setup wizard's final
        # save_config(config) preserves our model settings.  Without
        # this, the wizard overwrites model.provider/base_url with
        # the stale values from its own config dict.
        config["model"] = dict(model)

        print(f"Default model set to: {model_name} (via {effective_url})")
    else:
        if base_url or api_key:
            deactivate_provider()
        # Even without a model name, persist the custom endpoint on the
        # caller's config dict so the setup wizard doesn't lose it.
        _caller_model = config.get("model")
        if not isinstance(_caller_model, dict):
            _caller_model = {"default": _caller_model} if _caller_model else {}
        _caller_model["provider"] = "custom"
        _caller_model["base_url"] = effective_url
        if custom_key_env:
            _caller_model["api_key"] = f"${{{custom_key_env}}}"
        if api_mode:
            _caller_model["api_mode"] = api_mode
        else:
            _caller_model.pop("api_mode", None)
        config["model"] = _caller_model
        print("Endpoint saved. Use `/model` in chat or `pilotage model` to set a model.")

    # Auto-save to custom_providers so it appears in the menu next time
    _save_custom_provider(
        effective_url,
        effective_key,
        model_name or "",
        context_length=context_length,
        name=display_name,
        api_mode=api_mode,
        key_env=custom_key_env,
    )
    _prune_replaced_custom_model_config_credentials(
        effective_url,
        provider_name=display_name,
    )


def _model_flow_named_custom(config, provider_info):
    """Handle a named custom provider from config.yaml custom_providers list.

    Always probes the endpoint's /models API to let the user pick a model.
    If a model was previously saved, it is pre-selected in the menu.
    Falls back to the saved model if probing fails.
    """
    from pilotage_cli.main import _custom_provider_api_key_config_value, _custom_provider_base_url_config_value, _save_custom_provider
    from pilotage_cli.auth import _save_model_choice, deactivate_provider
    from pilotage_cli.config import load_config, save_config
    from pilotage_cli.models import fetch_api_models

    name = provider_info["name"]
    base_url = provider_info["base_url"]
    api_mode = provider_info.get("api_mode", "")
    api_key = provider_info.get("api_key", "")
    key_env = provider_info.get("key_env", "")
    saved_model = provider_info.get("model", "")
    provider_key = (provider_info.get("provider_key") or "").strip()

    # Resolve key from env var if api_key not set directly
    if not api_key and key_env:
        api_key = os.environ.get(key_env, "")
    config_api_key = _custom_provider_api_key_config_value(provider_info, api_key)

    # Honor ``discover_models: false`` (default True) — when discovery is
    # disabled, use the configured ``models:`` list verbatim and skip the
    # live /models probe. This lets operators restrict the picker to the
    # subset their plan actually serves instead of the endpoint's full
    # catalog (: Baidu Qianfan returns 100+ models for a 2-3 model
    # plan). Same semantics as the slash-command picker (model_switch.py
    # sections 3 & 4): default discovers, false keeps the explicit list.
    discover = provider_info.get("discover_models", True)
    if isinstance(discover, str):
        discover = discover.lower() not in {"false", "no", "0"}
    configured_models: list[str] = []
    cfg_models = provider_info.get("models", {})
    if isinstance(cfg_models, dict):
        configured_models = [str(m) for m in cfg_models if str(m).strip()]
    elif isinstance(cfg_models, list):
        configured_models = [
            str(m) for m in cfg_models if isinstance(m, str) and m.strip()
        ]

    print(f"  Provider: {name}")
    print(f"  URL:      {base_url}")
    if saved_model:
        print(f"  Current:  {saved_model}")
    print()

    if not discover and configured_models:
        # Discovery disabled with an explicit list — use it verbatim, no probe.
        print(f"Using configured models (discover_models: false): {len(configured_models)}")
        models = configured_models
    else:
        print("Fetching available models...")
        fetch_kwargs = {"timeout": 8.0}
        if api_mode:
            fetch_kwargs["api_mode"] = api_mode
        live_models = fetch_api_models(api_key, base_url, **fetch_kwargs)
        # If the probe came back empty but the operator configured an explicit
        # list, fall back to it rather than forcing manual entry.
        models = live_models or configured_models
        # Persist the live catalog back to the custom_providers entry so that
        # no-probe surfaces (dashboard, desktop, ACP) show the full model list
        # instead of collapsing to the single ``model:`` default. Mirrors the
        # picker path in model_switch.py::_save_discovered_models_to_config; a
        # failed save is non-fatal.
        if live_models:
            try:
                from pilotage_cli.model_switch import (
                    _save_discovered_models_to_config,
                )

                _save_discovered_models_to_config(base_url, live_models)
            except Exception:
                pass

    if models:
        default_idx = 0
        if saved_model and saved_model in models:
            default_idx = models.index(saved_model)

        print(f"Found {len(models)} model(s):\n")
        try:
            from pilotage_cli.curses_ui import curses_radiolist

            menu_items = [
                f"{m} (current)" if m == saved_model else m for m in models
            ] + ["Cancel"]
            idx = curses_radiolist(
                f"Select model from {name}:",
                menu_items,
                selected=default_idx,
                cancel_returns=-1,
                searchable=True,
            )
            print()
            if idx < 0 or idx >= len(models):
                print("Cancelled.")
                return
            model_name = models[idx]
        except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
            for i, m in enumerate(models, 1):
                suffix = " (current)" if m == saved_model else ""
                print(f"  {i}. {m}{suffix}")
            print(f"  {len(models) + 1}. Cancel")
            print()
            try:
                val = input(f"Choice [1-{len(models) + 1}]: ").strip()
                if not val:
                    print("Cancelled.")
                    return
                idx = int(val) - 1
                if idx < 0 or idx >= len(models):
                    print("Cancelled.")
                    return
                model_name = models[idx]
            except (ValueError, KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                return
    elif saved_model:
        print("Could not fetch models from endpoint.")
        try:
            model_name = input(f"Model name [{saved_model}]: ").strip() or saved_model
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
    else:
        print("Could not fetch models from endpoint. Enter model name manually.")
        try:
            model_name = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return
        if not model_name:
            print("No model specified. Cancelled.")
            return

    # Activate and save the model to the custom_providers entry
    _save_model_choice(model_name)

    cfg = load_config()
    model = cfg.get("model")
    if not isinstance(model, dict):
        model = {"default": model} if model else {}
        cfg["model"] = model
    if provider_key:
        model["provider"] = custom_provider_slug(name, provider_key)
        model.pop("base_url", None)
        model.pop("api_key", None)
    else:
        model["provider"] = "custom"
        model["base_url"] = _custom_provider_base_url_config_value(
            provider_info, base_url
        )
        if config_api_key:
            model["api_key"] = config_api_key
    # Apply api_mode from custom_providers entry, or clear stale value
    custom_api_mode = provider_info.get("api_mode", "")
    if custom_api_mode:
        model["api_mode"] = custom_api_mode
    else:
        model.pop("api_mode", None)  # let runtime auto-detect from URL
    save_config(cfg)
    deactivate_provider()

    # Persist the selected model back to whichever schema owns this endpoint.
    if provider_key:
        cfg = load_config()
        providers_cfg = cfg.get("providers")
        if isinstance(providers_cfg, dict):
            provider_entry = providers_cfg.get(provider_key)
            if isinstance(provider_entry, dict):
                provider_entry["default_model"] = model_name
                # Only persist an inline api_key when the user originally had
                # one (either a literal secret or a ``${VAR}`` template). When
                # the entry relies on ``key_env``, do not synthesize a
                # ``${key_env}`` api_key — the runtime already resolves the
                # key from ``key_env`` directly, and writing the resolved
                # secret (or even a synthesized template) would silently
                # downgrade credential hygiene on entries that intentionally
                # keep plaintext out of ``config.yaml``. See.
                original_api_key_ref = str(
                    provider_info.get("api_key_ref", "") or ""
                ).strip()
                original_api_key = str(provider_info.get("api_key", "") or "").strip()
                had_inline_api_key = bool(original_api_key_ref or original_api_key)
                if (
                    had_inline_api_key
                    and config_api_key
                    and not str(provider_entry.get("api_key", "") or "").strip()
                ):
                    provider_entry["api_key"] = config_api_key
                if key_env and not str(provider_entry.get("key_env", "") or "").strip():
                    provider_entry["key_env"] = key_env
                cfg["providers"] = providers_cfg
                save_config(cfg)
    else:
        # Save model name to the custom_providers entry for next time
        _save_custom_provider(base_url, config_api_key, model_name, api_mode=api_mode)

    print(f"\n✅ Model set to: {model_name}")
    print(f"   Provider: {name} ({base_url})")

def _model_flow_kimi(config, current_model=""):
    """Kimi / Moonshot model selection with automatic endpoint routing.

    - sk-kimi-* keys   → api.kimi.com/coding/v1  (Kimi Coding Plan)
    - Other keys        → api.moonshot.ai/v1      (legacy Moonshot)

    No manual base URL prompt — endpoint is determined by key prefix.
    """
    from pilotage_cli.main import _prompt_api_key
    from pilotage_cli.auth import (
        PROVIDER_REGISTRY,
        KIMI_CODE_BASE_URL,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from pilotage_cli.config import (
        get_env_value,
        save_env_value,
        load_config,
        save_config,
    )
    from pilotage_cli.models import _PROVIDER_MODELS

    provider_id = "kimi-coding"
    pconfig = PROVIDER_REGISTRY[provider_id]
    base_url_env = pconfig.base_url_env_var or ""

    # Step 1: Check / prompt for API key
    existing_key, existing_source = _existing_api_key_for_model_flow(provider_id, pconfig)

    existing_key, abort = _prompt_api_key(
        pconfig,
        existing_key,
        provider_id=provider_id,
        existing_source=existing_source,
    )
    if abort:
        return

    # Step 2: Auto-detect endpoint from key prefix
    is_coding_plan = existing_key.startswith("sk-kimi-")
    if is_coding_plan:
        effective_base = KIMI_CODE_BASE_URL
        print(f"  Detected Kimi Coding Plan key → {effective_base}")
    else:
        effective_base = pconfig.inference_base_url
        print(f"  Using Moonshot endpoint → {effective_base}")
    # Clear any manual base URL override so auto-detection works at runtime
    if base_url_env and get_env_value(base_url_env):
        save_env_value(base_url_env, "")
    print()

    # Step 3: Model selection — show appropriate models for the endpoint
    model_list = _PROVIDER_MODELS.get("kimi-coding" if is_coding_plan else "moonshot", [])

    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            confirm_provider=provider_id,
            confirm_base_url=effective_base,
            confirm_api_key=existing_key,
        )
    else:
        try:
            selected = input("Enter model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        _save_model_choice(selected)

        # Update config with provider and base URL
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = provider_id
        model["base_url"] = effective_base
        model.pop("api_mode", None)  # let runtime auto-detect from URL
        clear_model_endpoint_credentials(model, clear_api_mode=False)
        save_config(cfg)
        deactivate_provider()

        endpoint_label = "Kimi Coding" if is_coding_plan else "Moonshot"
        print(f"Default model set to: {selected} (via {endpoint_label})")
    else:
        print("No change.")

def _model_flow_stepfun(config, current_model=""):
    """StepFun Step Plan flow with region-specific endpoints."""
    from pilotage_cli.main import _infer_stepfun_region, _prompt_api_key, _prompt_provider_choice, _stepfun_base_url_for_region
    from pilotage_cli.auth import (
        PROVIDER_REGISTRY,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from pilotage_cli.config import (
        get_env_value,
        save_env_value,
        load_config,
        save_config,
    )
    from pilotage_cli.models import _PROVIDER_MODELS, fetch_api_models

    provider_id = "stepfun"
    pconfig = PROVIDER_REGISTRY[provider_id]
    base_url_env = pconfig.base_url_env_var or ""

    existing_key, existing_source = _existing_api_key_for_model_flow(provider_id, pconfig)

    existing_key, abort = _prompt_api_key(
        pconfig,
        existing_key,
        provider_id=provider_id,
        existing_source=existing_source,
    )
    if abort:
        return

    current_base = ""
    if base_url_env:
        current_base = get_env_value(base_url_env) or os.getenv(base_url_env, "")
    if not current_base:
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            current_base = str(model_cfg.get("base_url") or "").strip()
    current_region = _infer_stepfun_region(current_base or pconfig.inference_base_url)

    region_choices = [
        (
            "international",
            f"International ({_stepfun_base_url_for_region('international')})",
        ),
        ("china", f"China ({_stepfun_base_url_for_region('china')})"),
    ]
    ordered_regions = []
    for region_key, label in region_choices:
        if region_key == current_region:
            ordered_regions.insert(0, (region_key, f"{label}  ← currently active"))
        else:
            ordered_regions.append((region_key, label))
    ordered_regions.append(("cancel", "Cancel"))

    region_idx = _prompt_provider_choice([label for _, label in ordered_regions])
    if region_idx is None or ordered_regions[region_idx][0] == "cancel":
        print("No change.")
        return

    selected_region = ordered_regions[region_idx][0]
    effective_base = _stepfun_base_url_for_region(selected_region)
    if base_url_env:
        save_env_value(base_url_env, effective_base)

    live_models = fetch_api_models(existing_key, effective_base)
    if live_models:
        model_list = live_models
        print(f"  Found {len(model_list)} model(s) from {pconfig.name} API")
    else:
        model_list = _PROVIDER_MODELS.get(provider_id, [])
        if model_list:
            print(
                f"  Could not auto-detect models from {pconfig.name} API — "
                "showing Step Plan fallback catalog."
            )

    if model_list:
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            confirm_provider=provider_id,
            confirm_base_url=effective_base,
            confirm_api_key=existing_key,
        )
    else:
        try:
            selected = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        _save_model_choice(selected)

        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = provider_id
        model["base_url"] = effective_base
        model.pop("api_mode", None)
        clear_model_endpoint_credentials(model, clear_api_mode=False)
        save_config(cfg)
        deactivate_provider()

        config["model"] = dict(model)
        print(f"Default model set to: {selected} (via {pconfig.name})")
    else:
        print("No change.")

def _select_zai_endpoint(current_base: str) -> str:
    """Present a picker for Z.AI endpoint selection during setup.

    Offers the four official Z.AI endpoints (Global, China, Coding Plan
    Global, Coding Plan China) plus a custom-proxy option.  The list is
    sourced from ``ZAI_ENDPOINTS`` in ``pilotage_cli.auth`` so it stays in
    sync with the probe list.

    Returns the selected base URL.  Falls back to *current_base* on cancel
    or error.
    """
    from pilotage_cli.main import _prompt_provider_choice
    from pilotage_cli.auth import ZAI_ENDPOINTS

    # Build label + URL pairs from the shared endpoint list.
    options = [(label, url) for _, url, _, label in ZAI_ENDPOINTS]
    normalized_current = (current_base or "").strip().rstrip("/")

    # Default to the currently-active option if it matches one of the
    # known endpoints; otherwise default to the first (Global).
    default_idx = 0
    for idx, (_, url) in enumerate(options):
        if normalized_current == url.rstrip("/"):
            default_idx = idx
            break
    else:
        if normalized_current:
            # A custom URL is active — offer "Custom proxy" as the default.
            default_idx = len(options)

    choices = [f"{label} ({url})" for label, url in options]
    choices.append("Custom proxy URL")

    selected = _prompt_provider_choice(
        choices,
        default=default_idx,
        title="Select Z.AI / GLM endpoint:",
    )
    if selected is None:
        return current_base

    if selected == len(options):
        # Custom proxy URL
        try:
            override = input(f"Custom base URL [{current_base}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return current_base
        if not override:
            return current_base
        if not override.startswith(("http://", "https://")):
            print("  Invalid URL — must start with http:// or https://. Keeping current value.")
            return current_base
        return override.rstrip("/")

    return options[selected][1].rstrip("/")


def _model_flow_api_key_provider(config, provider_id, current_model=""):
    """Generic flow for API-key providers (z.ai, MiniMax, OpenCode, etc.)."""
    from pilotage_cli.main import _prompt_api_key
    from pilotage_cli.auth import (
        PROVIDER_REGISTRY,
        _prompt_model_selection,
        _save_model_choice,
        deactivate_provider,
    )
    from pilotage_cli.config import (
        get_env_value,
        save_env_value,
        load_config,
        save_config,
    )
    from pilotage_cli.models import (
        _PROVIDER_MODELS,
        fetch_api_models,
    )

    pconfig = PROVIDER_REGISTRY[provider_id]
    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""
    base_url_env = pconfig.base_url_env_var or ""

    # Check / prompt for API key
    existing_key, existing_source = _existing_api_key_for_model_flow(provider_id, pconfig)

    existing_key, abort = _prompt_api_key(
        pconfig,
        existing_key,
        provider_id=provider_id,
        existing_source=existing_source,
    )
    if abort:
        return

    # Optional base URL override.
    # Precedence: env var → config.yaml model.base_url → registry default.
    # Reading config.yaml prevents silently overwriting a saved remote URL
    # (e.g. a remote LM Studio endpoint) with localhost when the user just
    # presses Enter at the prompt below.
    current_base = ""
    if base_url_env:
        current_base = get_env_value(base_url_env) or os.getenv(base_url_env, "")
    if not current_base:
        try:
            _m = load_config().get("model") or {}
            if str(_m.get("provider") or "").strip().lower() == provider_id:
                current_base = str(_m.get("base_url") or "").strip()
        except Exception:
            pass
    effective_base = current_base or pconfig.inference_base_url

    if provider_id == "zai":
        # Z.AI has four official endpoints (Global, China, Coding Plan
        # Global, Coding Plan China) with separate billing paths.  Present
        # a picker instead of a plain text input so users can explicitly
        # choose the endpoint that matches their key type.
        chosen_base = _select_zai_endpoint(effective_base)
        if chosen_base and chosen_base != effective_base and base_url_env:
            save_env_value(base_url_env, chosen_base)
        effective_base = chosen_base
    else:
        try:
            override = input(f"Base URL [{effective_base}]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            override = ""
        if override and base_url_env:
            if not override.startswith(("http://", "https://")):
                print(
                    "  Invalid URL — must start with http:// or https://. Keeping current value."
                )
            else:
                save_env_value(base_url_env, override)
                effective_base = override

    # Model selection — resolution order:
    #   1. models.dev registry (cached, filtered for agentic/tool-capable models)
    #   2. Curated static fallback list (offline insurance)
    #   3. Live /models endpoint probe (small providers without models.dev data)
    #
    if provider_id == "novita":
        from pilotage_cli.models import fetch_api_models

        api_key_for_probe = existing_key or (get_env_value(key_env) if key_env else "")
        curated = _PROVIDER_MODELS.get(provider_id, [])
        live_models = fetch_api_models(api_key_for_probe, effective_base)
        if live_models:
            model_list = live_models
            print(f"  Found {len(model_list)} model(s) from {pconfig.name} API")
        else:
            mdev_models: list = []
            try:
                from agent.models_dev import list_agentic_models

                mdev_models = list_agentic_models(provider_id)
            except Exception:
                pass
            if mdev_models:
                seen = {m.lower() for m in mdev_models}
                model_list = list(mdev_models)
                for m in curated:
                    if m.lower() not in seen:
                        model_list.append(m)
                        seen.add(m.lower())
                print(f"  Found {len(model_list)} model(s) from models.dev registry")
            else:
                model_list = curated
                if model_list:
                    print(
                        f'  Showing {len(model_list)} curated models — use "Enter custom model name" for others.'
                    )
    else:
        curated = _PROVIDER_MODELS.get(provider_id, [])

        # Try models.dev first — returns tool-capable models, filtered for noise
        mdev_models: list = []
        try:
            from agent.models_dev import list_agentic_models

            mdev_models = list_agentic_models(provider_id)
        except Exception:
            pass

        if mdev_models:
            # Merge models.dev with curated list so newly added models
            # (not yet in models.dev) still appear in the picker.
            if curated:
                seen = {m.lower() for m in mdev_models}
                merged = list(mdev_models)
                for m in curated:
                    if m.lower() not in seen:
                        merged.append(m)
                        seen.add(m.lower())
                model_list = merged
            else:
                model_list = mdev_models
            print(f"  Found {len(model_list)} model(s) from models.dev registry")
        elif curated and len(curated) >= 8:
            # Curated list is substantial — use it directly, skip live probe
            model_list = curated
            print(
                f'  Showing {len(model_list)} curated models — use "Enter custom model name" for others.'
            )
        else:
            api_key_for_probe = existing_key or (
                get_env_value(key_env) if key_env else ""
            )
            live_models = fetch_api_models(api_key_for_probe, effective_base)
            if live_models and len(live_models) >= len(curated):
                model_list = live_models
                print(f"  Found {len(model_list)} model(s) from {pconfig.name} API")
            else:
                model_list = curated
                if model_list:
                    print(
                        f'  Showing {len(model_list)} curated models — use "Enter custom model name" for others.'
                    )
            # else: no defaults either, will fall through to raw input

    if model_list:
        # Per-model pricing, when the provider supports it (fireworks via the
        # models.dev disk cache, novita/deepinfra via their cached /models
        # endpoints). get_pricing_for_provider() is memoized in-process and
        # returns {} for providers without pricing — never a blocking fetch
        # beyond the catalog lookup that already happened above.
        pricing: dict = {}
        try:
            from pilotage_cli.models import get_pricing_for_provider

            pricing = get_pricing_for_provider(provider_id) or {}
        except Exception:
            pricing = {}
        selected = _prompt_model_selection(
            model_list,
            current_model=current_model,
            pricing=pricing,
            confirm_provider=provider_id,
            confirm_base_url=effective_base,
            confirm_api_key=existing_key,
        )
    else:
        try:
            selected = input("Model name: ").strip()
        except (KeyboardInterrupt, EOFError):
            selected = None

    if selected:
        _save_model_choice(selected)

        # Update config with provider, base URL, and provider-specific API mode
        cfg = load_config()
        model = cfg.get("model")
        if not isinstance(model, dict):
            model = {"default": model} if model else {}
            cfg["model"] = model
        model["provider"] = provider_id
        model["base_url"] = effective_base
        clear_model_endpoint_credentials(model, clear_api_mode=False)
        model.pop("api_mode", None)
        save_config(cfg)
        deactivate_provider()

        print(f"Default model set to: {selected} (via {pconfig.name})")
    else:
        print("No change.")
