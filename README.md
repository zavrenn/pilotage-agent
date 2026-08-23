<p align="center">
  <img src="assets/logo-circle.png" alt="Pilotage Agent logo" width="180">
</p>

# Pilotage Agent

A lightweight runtime for focused, long-lived AI agents.

Pilotage Agent is a deliberately small alternative to broad, general-purpose
agent frameworks. It prioritizes clarity, control, and reliability over support
for every model, channel, environment, and use case.

> Build the smallest system that can be trusted with a company.

## Focus

The project is intended for personal and organizational agents that maintain
context, retain useful knowledge, use tools and skills, run scheduled work, and
communicate through familiar messaging channels.

New capabilities will be added from concrete use cases, not to pursue framework
completeness.

## Principles

- Focused rather than universal.
- Explicit behavior rather than hidden complexity.
- Strong isolation and clear trust boundaries.
- Reliability before feature count.
- Evidence before expansion.

## Status

The Genesis core currently provides ChatGPT subscription authentication,
allowlisted WhatsApp and Telegram messaging, persistent conversations with
native context compaction, full-text conversation recall, curated memory,
profile-local tools and skills, isolated profiles, management commands, DDGS
web search, native image analysis, OpenAI voice-message transcription,
Codex-backed image generation and editing, allowlisted messaging groups, and
durable cron jobs. Memory, skill, and cron changes use configurable per-profile
approval gates over both messaging channels. This Genesis slice is not
production-complete, and there is no stable public API.

## Install on Ubuntu

```bash
scripts/install.sh
# Edit ~/.pilotage-agent/.env and ~/.pilotage-agent/config.yaml
./.venv/bin/pilotage login
./.venv/bin/pilotage run
```

Voice-message transcription requires VOICE_TOOLS_OPENAI_KEY in the profile
.env; ChatGPT login does not authorize the OpenAI audio API.

An enabled WhatsApp channel prints its pairing QR on the first run. Telegram
uses TELEGRAM_BOT_TOKEN and numeric IDs in TELEGRAM_ALLOWED_USERS from the
profile .env.
Webhook delivery also requires TELEGRAM_WEBHOOK_URL and a long random
TELEGRAM_WEBHOOK_SECRET; leaving the URL blank uses long polling.
After testing the enabled channels, install the resident user service:

```bash
bash scripts/install-service.sh
```

Useful operator commands:

```bash
./.venv/bin/pilotage status
./.venv/bin/pilotage profile create work
# Edit ~/.pilotage-agent/profiles/work/.env and config.yaml
./.venv/bin/pilotage --profile work run
bash scripts/install-service.sh --profile work
./.venv/bin/pilotage cron list --all
```

Each named profile owns its `SOUL.md` identity, configuration, WhatsApp session,
Telegram credentials, conversations, memory, skills, workspace, cron jobs, and
an automatically assigned bridge port. Only one live runtime may own a profile.
A named profile may fall back only to the default profile's ChatGPT
authentication. An optional `AGENTS.md` in
the working directory supplies workspace instructions to each new conversation.

## Origin

Pilotage Agent builds on [Hermes Agent](https://hermes-agent.nousresearch.com) by
Nous Research, used under the MIT License. It selectively reuses proven Hermes
mechanisms and code while keeping a smaller Pilotage-owned runtime shape.

The original copyright notice is retained in [LICENSE](LICENSE), as the MIT
License requires.
