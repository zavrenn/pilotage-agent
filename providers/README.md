# providers/

Registry and ABC for the inference providers Pilotage knows about.

Each provider is declared once as a `ProviderProfile`. Every other layer —
auth resolution, transport kwargs, model listing, runtime routing — reads from
these profiles instead of maintaining its own parallel data.

```
providers/
├── base.py         ProviderProfile dataclass + OMIT_TEMPERATURE sentinel
├── __init__.py     Registry: register_provider(), get_provider_profile(), list_providers()
└── README.md       This file
```

The profiles themselves live as plugin directories, each holding an
`__init__.py` that calls `register_provider(profile)` plus a `plugin.yaml`
manifest. Two are bundled:

- `plugins/model-providers/openai-codex/` — the live path. OAuth to
  `https://chatgpt.com/backend-api/codex`, `api_mode="codex_responses"`.
- `plugins/model-providers/custom/` — any user-supplied OpenAI-compatible
  base URL.

`$PILOTAGE_HOME/plugins/model-providers/<name>/` is scanned too and overrides
a bundled plugin of the same name (last writer wins in `register_provider`).
Discovery is lazy: `_discover_providers()` runs on the first
`get_provider_profile()` or `list_providers()` call.

---

## How it wires in

- `pilotage_cli/auth.py` extends `PROVIDER_REGISTRY` with every `api_key`
  profile it sees.
- `pilotage_cli/models.py` extends `CANONICAL_PROVIDERS` and calls
  `profile.fetch_models()` inside `provider_model_ids()`.
- `pilotage_cli/doctor.py` adds a `/models` health check per `api_key` profile.
- `pilotage_cli/config.py` injects every `env_var` into `OPTIONAL_ENV_VARS`
  so the setup wizard knows about it.
- `agent/model_metadata.py` maps hostname → provider via
  `profile.get_hostname()`.
- `agent/auxiliary_client.py` reads `profile.default_aux_model`.
- `agent/transports/chat_completions.py::_build_kwargs_from_profile()` invokes
  `prepare_messages()`, `build_extra_body()` and `build_api_kwargs_extras()`
  on every call.
- `run_agent.py` passes `provider_profile=<ProviderProfile>` so the transport
  takes the profile path.

---

## Overridable hooks

| Hook | Purpose |
|------|---------|
| `get_hostname()` | URL-based detection — default derives from `base_url`. |
| `prepare_messages(msgs)` | Message preprocessing before the request is built. |
| `build_extra_body(**ctx)` | Provider-specific `extra_body` payload. |
| `build_api_kwargs_extras(**ctx)` | `(extra_body_additions, top_level_kwargs)`. |
| `fetch_models(*, api_key)` | Live catalog fetch — default hits `{models_url or base_url}/models` with Bearer auth. |

Full field reference: the dataclass in `providers/base.py`.
