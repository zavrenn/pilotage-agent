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
allowlisted WhatsApp messaging, persistent conversations with native context
compaction, curated memory, profile-local tools and skills, isolated profiles,
management commands, DDGS web search, and durable cron jobs. WhatsApp group mode
remains closed until its mention policy is implemented. This Genesis slice is
not production-complete: Telegram, voice transcription, and the approval
workflow are not implemented yet. There is no stable public API.

## Install on Ubuntu

```bash
scripts/install.sh
# Edit ~/.pilotage-agent/.env and ~/.pilotage-agent/config.yaml
./.venv/bin/pilotage login
./.venv/bin/pilotage run
```

The first run prints the WhatsApp pairing QR. After pairing and testing, install
the resident user service:

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
conversations, memory, skills, workspace, cron jobs, and an automatically assigned
bridge port. Only one live runtime may own a profile. A named profile may fall back
only to the default profile's ChatGPT authentication. An optional `AGENTS.md` in
the working directory supplies workspace instructions to each new conversation.

## Origin

Pilotage Agent builds on [Hermes Agent](https://hermes-agent.nousresearch.com) by
Nous Research, used under the MIT License. It selectively reuses proven Hermes
mechanisms and code while keeping a smaller Pilotage-owned runtime shape.

The original copyright notice is retained in [LICENSE](LICENSE), as the MIT
License requires.
