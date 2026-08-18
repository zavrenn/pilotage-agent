"""OpenAI Chat Completions transport.

Handles the default api_mode ('chat_completions') used by OpenAI and other
OpenAI-compatible endpoints.

Messages and tools are already in OpenAI format — convert_messages and
convert_tools are near-identity.  The complexity lives in build_kwargs
which has provider-specific conditionals for max_tokens defaults,
reasoning configuration, temperature handling, and extra_body assembly.
"""

import json
from typing import Any, Dict

from agent.prompt_builder import DEVELOPER_ROLE_MODELS
from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage


def _static_prompt_instructions(messages: list[dict[str, Any]]) -> str:
    """Return the stable system/developer prefix used for cache routing.

    Chat Completions carries instructions in its message list rather than a
    separate ``instructions`` field.  Only a leading system/developer message
    is static by contract; later messages are conversation state and must not
    split a warm prefix bucket on every turn.
    """
    if not messages or not isinstance(messages[0], dict):
        return ""
    first = messages[0]
    if first.get("role") not in {"system", "developer"}:
        return ""
    content = first.get("content")
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(content or "")


def _add_prompt_cache_key(
    api_kwargs: dict[str, Any],
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    supports_prompt_cache_key: bool,
    session_id: str | None = None,
    cache_scope_id: str | None = None,
) -> None:
    """Add a content-addressed key only for an explicitly capable endpoint.

    ``cache_scope_id``, when provided, is the rotation-stable logical scope
    (compression-lineage root — agent/prompt_cache_scope.py) and takes
    precedence over the physical ``session_id`` so the key survives
    context-compression session rotation.
    """
    if not supports_prompt_cache_key:
        return

    # An explicit caller body field is authoritative too.  Do not add a
    # duplicate top-level field whose SDK merge precedence could overwrite it.
    extra_body = api_kwargs.get("extra_body")
    if "prompt_cache_key" in api_kwargs or (
        isinstance(extra_body, dict) and "prompt_cache_key" in extra_body
    ):
        return

    # Reuse the Responses transport's single authoritative hash algorithm and
    # session-scope normalization so equivalent static prefixes route to the
    # same cache bucket across modes, without concentrating unrelated
    # sessions into one shared bucket.
    from agent.transports.codex import _cache_scope_from_session_id, _content_cache_key

    cache_key = _content_cache_key(
        _static_prompt_instructions(messages),
        tools,
        _cache_scope_from_session_id(cache_scope_id or session_id),
    )
    if cache_key:
        api_kwargs["prompt_cache_key"] = cache_key


def _reasoning_config_for_model(model: str, reasoning_config: dict | None) -> dict | None:
    """Return the model's wire-compatible reasoning config."""
    if not isinstance(reasoning_config, dict):
        return reasoning_config
    if (
        "gpt-5.6" in (model or "").lower()
        and str(reasoning_config.get("effort") or "").strip().lower() == "ultra"
    ):
        normalized = dict(reasoning_config)
        normalized["effort"] = "max"
        return normalized
    return reasoning_config


def _is_openai_api_base_url(base_url: Any) -> bool:
    """True only for api.openai.com itself (exact host).

    OpenAI documents ``prompt_cache_key`` as a first-class body field and
    GPT-5.6+ docs recommend it for reliable cache routing, so the flag is
    implied for the real endpoint. Deliberately NOT a substring match:
    Azure OpenAI and strict OpenAI-compat endpoints may reject unknown
    fields and must stay opt-in via ``supports_prompt_cache_key``.
    """
    try:
        from urllib.parse import urlparse

        host = (urlparse(str(base_url or "").strip()).hostname or "").lower()
    except Exception:
        return False
    return host == "api.openai.com"


class ChatCompletionsTransport(ProviderTransport):
    """Transport for api_mode='chat_completions'.

    The default path for OpenAI-compatible providers.
    """

    @property
    def api_mode(self) -> str:
        return "chat_completions"

    def convert_messages(
        self, messages: list[dict[str, Any]], **kwargs
    ) -> list[dict[str, Any]]:
        """Messages are already in OpenAI format — strip internal fields
        that strict chat-completions providers reject with HTTP 400/422
        (or, in the case of some OpenAI-compatible gateways, 5xx):

        - Codex Responses API fields: ``codex_reasoning_items`` /
          ``codex_message_items`` on the message, ``call_id`` /
          ``response_item_id`` on ``tool_calls`` entries.
        - ``extra_content`` on ``tool_calls`` — opaque per-tool-call
          metadata some thinking models attach. Strict providers reject any
          payload containing it with
          ``Extra inputs are not permitted, field: 'messages[N].tool_calls[M].extra_content'``.
        - ``tool_name`` on tool-result messages — written by
          ``make_tool_result_message()`` for the SQLite FTS index, but not
          part of the Chat Completions schema. Strict providers reject any
          payload containing it with
          ``Extra inputs are not permitted, field: 'messages[N].tool_name'``.
          Permissive providers silently ignore the field.
        - Pilotage-internal scaffolding markers — any top-level message key
          starting with ``_`` (e.g. ``_empty_recovery_synthetic``,
          ``_empty_terminal_sentinel``, ``_thinking_prefill``). These are
          bookkeeping flags the agent loop attaches to messages so the
          persistence layer can later strip its own scaffolding; they must
          never reach the wire. Permissive providers silently drop unknown
          message keys, but strict gateways reject with
          ``Extra inputs are not permitted, field: 'messages[N]._empty_recovery_synthetic'``,
          which then poisons every subsequent request in the session.
        """
        needs_sanitize = False
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if (
                "codex_reasoning_items" in msg
                or "codex_message_items" in msg
                or "tool_name" in msg
                or "effect_disposition" in msg
                or "timestamp" in msg # — strict providers reject this
                or "api_content" in msg  # persist-what-you-send sidecar
            ):
                needs_sanitize = True
                break
            if any(isinstance(k, str) and k.startswith("_") for k in msg):
                needs_sanitize = True
                break
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                # Defense-in-depth: a strict OpenAI-compatible provider
                # rejects an assistant message carrying
                # ``tool_calls: []`` (empty array) with
                # HTTP 400 "Empty tool_calls is not supported in message."
                # The pre-API sanitizer in agent_runtime_helpers drops these,
                # but only on the conversation_loop path — other routes can
                # reach the wire without it. For every request that
                # serializes through this transport (conversation loop and
                # any caller using it), this is the last boundary, so
                # normalize here. Requests built by fully separate payload
                # paths (e.g. some auxiliary clients) never pass through
                # this layer and are out of scope for it. follow-up)
                if (
                    msg.get("role") == "assistant"
                    and "tool_calls" in msg
                    and not tool_calls
                ):
                    needs_sanitize = True
                    break
                for tc in tool_calls:
                    if isinstance(tc, dict) and (
                        "call_id" in tc
                        or "response_item_id" in tc
                        or "extra_content" in tc
                    ):
                        needs_sanitize = True
                        break
                if needs_sanitize:
                    break
            elif (
                isinstance(tool_calls, type(None))
                and msg.get("role") == "assistant"
                and "tool_calls" in msg
            ):
                # Explicit ``tool_calls: null`` is equally invalid on strict
                # providers — treat it like the empty-array case.
                needs_sanitize = True
                break

        if not needs_sanitize:
            return messages

        sanitized = list(messages)
        for msg_idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue

            copied_msg: dict[str, Any] | None = None

            def mutable_msg() -> dict[str, Any]:
                nonlocal copied_msg
                if copied_msg is None:
                    copied_msg = dict(msg)
                    sanitized[msg_idx] = copied_msg
                return copied_msg

            if (
                "codex_reasoning_items" in msg
                or "codex_message_items" in msg
                or "tool_name" in msg
                or "effect_disposition" in msg
                or "timestamp" in msg # — leak into strict providers
                or "api_content" in msg  # persist-what-you-send sidecar
            ):
                out_msg = mutable_msg()
                out_msg.pop("codex_reasoning_items", None)
                out_msg.pop("codex_message_items", None)
                out_msg.pop("tool_name", None)
                out_msg.pop("effect_disposition", None)
                out_msg.pop("timestamp", None) # — leak into strict providers
                out_msg.pop("api_content", None)  # persist-what-you-send sidecar


            # Drop all Pilotage-internal scaffolding markers (``_``-prefixed).
            # OpenAI's message schema has no ``_``-prefixed fields, so this
            # is safe and future-proofs against new markers being added.
            internal_keys = [k for k in msg if isinstance(k, str) and k.startswith("_")]
            if internal_keys:
                out_msg = mutable_msg()
                for key in internal_keys:
                    out_msg.pop(key, None)

            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                # Strip empty/invalid tool_calls arrays at the transport
                # layer (see detection above). Strict OpenAI-compatible
                # providers reject ``tool_calls: []`` with HTTP 400; dropping
                # the key keeps the message schema-valid. Matches the
                # pre-API sanitizer's behaviour so all routes agree.
                if (
                    msg.get("role") == "assistant"
                    and "tool_calls" in msg
                    and not tool_calls
                ):
                    out_msg = mutable_msg()
                    out_msg.pop("tool_calls", None)
                    continue
                copied_tool_calls: list[Any] | None = None
                for tc_idx, tc in enumerate(tool_calls):
                    if isinstance(tc, dict):
                        should_copy_tc = (
                            "call_id" in tc
                            or "response_item_id" in tc
                            or "extra_content" in tc
                        )
                        if should_copy_tc:
                            if copied_tool_calls is None:
                                copied_tool_calls = list(tool_calls)
                            copied_tc = dict(tc)
                            copied_tc.pop("call_id", None)
                            copied_tc.pop("response_item_id", None)
                            copied_tc.pop("extra_content", None)
                            copied_tool_calls[tc_idx] = copied_tc
                if copied_tool_calls is not None:
                    mutable_msg()["tool_calls"] = copied_tool_calls
            elif (
                isinstance(tool_calls, type(None))
                and msg.get("role") == "assistant"
                and "tool_calls" in msg
            ):
                # Explicit ``tool_calls: null`` is invalid on strict
                # providers — drop the key entirely.
                mutable_msg().pop("tool_calls", None)
        return sanitized

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Tools are already in OpenAI format — identity."""
        return tools

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params,
    ) -> dict[str, Any]:
        """Build chat.completions.create() kwargs.

        params (all optional):
            timeout: float — API call timeout
            max_tokens: int | None — user-configured max tokens
            ephemeral_max_output_tokens: int | None — one-shot override
            max_tokens_param_fn: callable — returns {max_tokens: N} or {max_completion_tokens: N}
            reasoning_config: dict | None
            request_overrides: dict | None
            session_id: str | None
            model_lower: str — lowercase model name for pattern matching
            # Provider profile path (all per-provider quirks live in providers/)
            provider_profile: ProviderProfile | None — when present, delegates to
                _build_kwargs_from_profile(); all flag params below are bypassed.
            # Legacy-path flags — only used when provider_profile is None
            # (i.e. custom / unregistered providers). Known providers all go
            # through provider_profile.
            is_custom_provider: bool
            # Reasoning
            supports_reasoning: bool
            extra_body_additions: dict | None
            supports_prompt_cache_key: bool — explicit endpoint capability for
                the top-level Chat Completions request field; defaults off.
        """
        # Codex sanitization: drop reasoning_items / call_id / response_item_id.
        sanitized = self.convert_messages(messages, model=model)

        # ── Provider profile: single-path when present ──────────────────
        _profile = params.get("provider_profile")
        if _profile:
            return self._build_kwargs_from_profile(
                _profile, model, sanitized, tools, params
            )

        # ── Legacy fallback (unregistered / unknown provider) ───────────
        # Reached only when get_provider_profile() returned None.
        # Known providers always go through the profile path above.

        # Developer role swap for GPT-5/Codex models
        model_lower = params.get("model_lower", (model or "").lower())
        if (
            sanitized
            and isinstance(sanitized[0], dict)
            and sanitized[0].get("role") == "system"
            and any(p in model_lower for p in DEVELOPER_ROLE_MODELS)
        ):
            sanitized = list(sanitized)
            sanitized[0] = {**sanitized[0], "role": "developer"}

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": sanitized,
        }

        timeout = params.get("timeout")
        if timeout is not None:
            api_kwargs["timeout"] = timeout

        # Tools
        if tools:
            api_kwargs["tools"] = tools

        # max_tokens resolution — priority: ephemeral > user > provider default
        max_tokens_fn = params.get("max_tokens_param_fn")
        ephemeral = params.get("ephemeral_max_output_tokens")
        max_tokens = params.get("max_tokens")
        reasoning_config = _reasoning_config_for_model(model, params.get("reasoning_config"))

        if ephemeral is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(ephemeral))
        elif max_tokens is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(max_tokens))
        # extra_body assembly
        extra_body: dict[str, Any] = {}

        provider_name = str(params.get("provider_name") or "").strip().lower()
        base_url = params.get("base_url")

        if params.get("supports_reasoning", False):
            _effort = "medium"
            if reasoning_config and isinstance(reasoning_config, dict):
                _effort = reasoning_config.get("effort", "medium") or "medium"
            extra_body["reasoning"] = {"enabled": True, "effort": _effort}

        # Merge any pre-built extra_body additions
        additions = params.get("extra_body_additions")
        if additions:
            extra_body.update(additions)

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        # Request overrides last (service_tier etc.)
        overrides = params.get("request_overrides")
        if overrides:
            api_kwargs.update(overrides)

        _add_prompt_cache_key(
            api_kwargs,
            messages=sanitized,
            tools=api_kwargs.get("tools"),
            supports_prompt_cache_key=bool(params.get("supports_prompt_cache_key"))
            or _is_openai_api_base_url(params.get("base_url")),
            session_id=params.get("session_id"),
            cache_scope_id=params.get("cache_scope_id"),
        )

        return api_kwargs

    def _build_kwargs_from_profile(self, profile, model, sanitized, tools, params):
        """Build API kwargs using a ProviderProfile — single path, no legacy flags.

        This method replaces the entire flag-based kwargs assembly when a
        provider_profile is passed. Every quirk comes from the profile object.
        """
        from providers.base import OMIT_TEMPERATURE

        # Message preprocessing
        sanitized = profile.prepare_messages(sanitized)

        # Developer role swap — model-name-based, applies to all providers
        _model_lower = (model or "").lower()
        if (
            sanitized
            and isinstance(sanitized[0], dict)
            and sanitized[0].get("role") == "system"
            and any(p in _model_lower for p in DEVELOPER_ROLE_MODELS)
        ):
            sanitized = list(sanitized)
            sanitized[0] = {**sanitized[0], "role": "developer"}

        api_kwargs: dict[str, Any] = {
            "model": model,
            "messages": sanitized,
        }

        # Temperature
        if profile.fixed_temperature is OMIT_TEMPERATURE:
            pass  # Don't include temperature at all
        elif profile.fixed_temperature is not None:
            api_kwargs["temperature"] = profile.fixed_temperature
        else:
            # Use caller's temperature if provided
            temp = params.get("temperature")
            if temp is not None:
                api_kwargs["temperature"] = temp

        # Timeout
        timeout = params.get("timeout")
        if timeout is not None:
            api_kwargs["timeout"] = timeout

        # Tools
        if tools:
            api_kwargs["tools"] = tools

        # max_tokens resolution — priority: ephemeral > user > profile default
        max_tokens_fn = params.get("max_tokens_param_fn")
        ephemeral = params.get("ephemeral_max_output_tokens")
        user_max = params.get("max_tokens")
        # Per-model default cap — profiles override get_max_tokens() when
        # they front several backends with different completion-token limits
        # (e.g. opencode-go: mimo-v2.5-pro = 131072).
        profile_max = profile.get_max_tokens(model)

        if ephemeral is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(ephemeral))
        elif user_max is not None and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(user_max))
        elif profile_max and max_tokens_fn:
            api_kwargs.update(max_tokens_fn(profile_max))

        # Provider-specific api_kwargs extras (reasoning_effort, metadata, etc.)
        reasoning_config = _reasoning_config_for_model(model, params.get("reasoning_config"))
        extra_body_from_profile, top_level_from_profile = (
            profile.build_api_kwargs_extras(
                reasoning_config=reasoning_config,
                supports_reasoning=params.get("supports_reasoning", False),
                model=model,
                base_url=params.get("base_url"),
                session_id=params.get("session_id"),
            )
        )
        api_kwargs.update(top_level_from_profile)

        # extra_body assembly
        extra_body: dict[str, Any] = {}

        # Profile's extra_body (tags, provider prefs, vl_high_resolution, etc.)
        profile_body = profile.build_extra_body(
            session_id=params.get("session_id"),
            model=model,
            base_url=params.get("base_url"),
            reasoning_config=reasoning_config,
        )
        if profile_body:
            extra_body.update(profile_body)

        # Profile's reasoning/thinking extra_body entries
        if extra_body_from_profile:
            extra_body.update(extra_body_from_profile)

        # Merge any pre-built extra_body additions from the caller
        additions = params.get("extra_body_additions")
        if additions:
            extra_body.update(additions)

        # Request overrides (user config)
        overrides = params.get("request_overrides")
        if overrides:
            for k, v in overrides.items():
                if k == "extra_body" and isinstance(v, dict):
                    extra_body.update(v)
                else:
                    api_kwargs[k] = v

        if extra_body:
            api_kwargs["extra_body"] = extra_body

        _add_prompt_cache_key(
            api_kwargs,
            messages=sanitized,
            tools=api_kwargs.get("tools"),
            supports_prompt_cache_key=bool(getattr(profile, "supports_prompt_cache_key", False)),
            session_id=params.get("session_id"),
            cache_scope_id=params.get("cache_scope_id"),
        )

        return api_kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize OpenAI ChatCompletion to NormalizedResponse.

        For chat_completions, this is near-identity — the response is already
        in OpenAI format.  extra_content on tool_calls is preserved via
        ToolCall.provider_data.  reasoning_details and reasoning_content are
        also preserved for downstream replay.
        """
        choice = response.choices[0]
        msg = choice.message
        # Some endpoints return an integer finish_reason instead of a string
        _fr = choice.finish_reason
        if isinstance(_fr, int):
            _fr = str(_fr)
        finish_reason = _fr or "stop"

        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                # Preserve provider-specific extras on the tool call.
                # Some thinking models attach extra_content — without
                # replay on the next turn the API rejects the request with
                # 400.
                tc_provider_data: dict[str, Any] = {}
                extra = getattr(tc, "extra_content", None)
                if extra is None and hasattr(tc, "model_extra"):
                    extra = (tc.model_extra if isinstance(tc.model_extra, dict) else {}).get("extra_content")
                if extra is not None:
                    if hasattr(extra, "model_dump"):
                        try:
                            extra = extra.model_dump(warnings=False)
                        except TypeError:
                            try:
                                extra = extra.model_dump()
                            except Exception:
                                pass
                        except Exception:
                            pass
                    tc_provider_data["extra_content"] = extra
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                        provider_data=tc_provider_data or None,
                    )
                )

        usage = None
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            usage = Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )

        # Preserve reasoning fields separately.  Some providers use
        # ``reasoning_content``; others use ``reasoning``.  Downstream code
        # (_extract_reasoning, thinking-prefill retry) reads both distinctly,
        # so keep them apart in provider_data rather than merging.
        reasoning = getattr(msg, "reasoning", None)
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content is None and hasattr(msg, "model_extra"):
            model_extra = getattr(msg, "model_extra", None) or {}
            if isinstance(model_extra, dict) and "reasoning_content" in model_extra:
                reasoning_content = model_extra["reasoning_content"]

        provider_data: Dict[str, Any] = {}
        if reasoning_content is not None:
            provider_data["reasoning_content"] = reasoning_content
        rd = getattr(msg, "reasoning_details", None)
        if rd:
            provider_data["reasoning_details"] = rd

        # OpenAI structured-refusal field. When a model declines, the SDK
        # populates ``message.refusal`` with the explanation and leaves
        # ``content`` empty. Some proxies surface a refusal this way — or
        # via ``finish_reason="content_filter"``. Without capturing it the
        # refusal looks
        # like an empty response, so the agent loop retries a deterministic
        # refusal three times and gives up with "no content after retries".
        # Promote it to content + a ``content_filter`` finish reason so the
        # loop's refusal handler surfaces it clearly and stops. ``refusal`` is
        # ``None`` for normal responses, so this is a no-op in the common case.
        content = msg.content
        refusal = getattr(msg, "refusal", None)
        if refusal is None and hasattr(msg, "model_extra"):
            _msg_extra = getattr(msg, "model_extra", None) or {}
            if isinstance(_msg_extra, dict):
                refusal = _msg_extra.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            # Record the refusal explanation regardless — it's useful provider
            # metadata even when the model also returned a usable payload.
            provider_data["refusal"] = refusal
            _has_text = isinstance(content, str) and content.strip()
            _has_tool_calls = bool(tool_calls)
            # Only promote to a terminal ``content_filter`` when the refusal is
            # the *sole* payload — no visible text and no tool calls. A response
            # that carries real content (or tool calls) alongside a refusal note
            # is a normal, usable turn: surfacing it as a failed safety refusal
            # would discard the model's actual work. In the empty-payload case,
            # adopt the refusal as content so the loop has something to show.
            if not _has_text and not _has_tool_calls:
                content = refusal
                if finish_reason in (None, "stop"):
                    finish_reason = "content_filter"

        return NormalizedResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            reasoning=reasoning,
            usage=usage,
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Check that response has valid choices."""
        if response is None:
            return False
        if not hasattr(response, "choices") or response.choices is None:
            return False
        if not response.choices:
            return False
        return True

    def extract_cache_stats(self, response: Any) -> dict[str, int] | None:
        """Extract cache stats from prompt_tokens_details, or a top-level
        prompt_cache_hit_tokens field."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0 if details else 0
        written = getattr(details, "cache_write_tokens", 0) or 0 if details else 0
        if not cached:
            # Alternative shape: top-level prompt_cache_hit_tokens /
            # prompt_cache_miss_tokens.
            cached = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
        if cached or written:
            return {"cached_tokens": cached, "creation_tokens": written}
        return None


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("chat_completions", ChatCompletionsTransport)
