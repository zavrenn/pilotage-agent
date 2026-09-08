<p align="center">
  <img src="assets/logo-circle.png" alt="Pilotage Agent logo" width="180">
</p>

# Pilotage Agent

A lightweight runtime for focused, long-lived AI agents.

> [!IMPORTANT]
> Pilotage Agent is a personal, opinionated runtime built for a specific,
> controlled deployment. It is shared publicly for reuse and study, not as a
> general-purpose agent framework or supported product. Its capabilities and
> integrations are deliberately limited to concrete needs. Ubuntu Server 24.04
> on amd64 is the documented deployment target; other compatible environments
> may work, but are not currently validated.

Pilotage Agent prioritizes clarity, control, and reliability over breadth.

> Build the smallest system that can be trusted with a company.

## Focus

The project is intended for personal and organizational agents that maintain
context, retain useful knowledge, use tools and skills, run scheduled work, and
communicate through familiar messaging channels.

New capabilities are added only when required by the maintainer's deployment,
not to pursue framework completeness.

## Principles

- Focused rather than universal.
- Explicit behavior rather than hidden complexity.
- Strong isolation and clear trust boundaries.
- Reliability before feature count.
- Evidence before expansion.

## Status

Pilotage Agent is an early V1 focused on ChatGPT subscription authentication,
GPT-5.6, WhatsApp and Telegram, persistent context and memory, selected tools
and skills, and scheduled work.

It is still evolving and has no stable public API or compatibility guarantee.

## Install on Ubuntu

Clone the repository as the unprivileged service user:

```bash
git clone https://github.com/zavrenn/pilotage-agent.git
cd pilotage-agent
```

Run the system dependency installer from that checkout as root:

```bash
bash scripts/install-system-dependencies.sh
```

Then return to the unprivileged service user:

```bash
bash scripts/install.sh
# Review ~/.pilotage-agent/config.yaml and add required integration credentials to .env
./.venv/bin/pilotage login
./.venv/bin/pilotage whatsapp  # optional: configure, pair, and enable WhatsApp
./.venv/bin/pilotage telegram  # optional: configure, verify, and enable Telegram
./.venv/bin/pilotage run
```

Voice-message transcription requires `VOICE_TOOLS_OPENAI_KEY` in the profile
`.env`; ChatGPT login does not authorize the OpenAI audio API.

Full-page web extraction requires `FIRECRAWL_API_KEY` in the profile `.env`, or
`FIRECRAWL_API_URL` for a self-hosted Firecrawl instance. DDGS search needs no
key. Without either Firecrawl setting, the agent offers search only and Doctor
does not require extraction credentials. Restart after configuring extraction.

`pilotage whatsapp` saves the WhatsApp allowlist and home destination in the
effective profile environment file, then pairs and enables the channel.
`pilotage telegram` securely collects and verifies the bot token, allowed user
IDs, and home destination, then enables Telegram. Run either or both setup
commands; a fresh install enables neither channel.
An allowed WhatsApp number or Telegram user ID may use the agent in a DM or any
group; `require_mention` can require that person to address the agent directly
inside groups.
Webhook delivery also requires `TELEGRAM_WEBHOOK_URL` and a long random
`TELEGRAM_WEBHOOK_SECRET`; leaving the URL blank uses long polling.
After testing the enabled channels, install the resident user service:

```bash
bash scripts/install-service.sh
./.venv/bin/pilotage doctor
```

The deployment is ready only when Doctor reports every required check as ready.

Useful operator commands:

```bash
pilotage status
pilotage update --check
pilotage update
pilotage restart
pilotage logs -f --level WARNING
pilotage logs --since 1h -n 100
pilotage profile create work
# Edit ~/.pilotage-agent/profiles/work/.env and config.yaml
pilotage --profile work run
bash scripts/install-service.sh --profile work
pilotage --profile work service status
pilotage --profile work service stop
pilotage --profile work service start
pilotage --profile work restart
pilotage --profile work logs -f
pilotage cron list --all
```

The installer links `pilotage` into `~/.local/bin`; add that directory to your
shell's `PATH` if needed, or keep using `./.venv/bin/pilotage`.
`update --check` fetches Git metadata without changing code or restarting anything.
`update` follows the current branch's configured upstream and refuses local edits
or commits ahead of upstream. Stop other profiles sharing the installation first.
It stops the selected service, fast-forwards the checkout, runs the locked installer
and an import check, then starts the service again only if it was running before.
An installation failure leaves it stopped; fix the error, rerun `update`, then
`restart`. These commands use the invoking user's existing permissions and never
elevate privileges; separating runtime ownership from the agent account remains
a deployment responsibility.

`logs` reads the selected user service's journal. `-n` limits the journal entries
inspected before severity filtering; `--level WARNING` includes warnings and more
severe messages. Stream messages without a severity remain visible, including
tracebacks and startup errors. `--since` accepts
relative times such as `30m` or a timestamp such as `"2026-09-08 20:39:00"`.

Each named profile owns its `SOUL.md` identity, configuration, WhatsApp session,
Telegram credentials, conversations, memory, skills, workspace, cron jobs, and
an automatically assigned bridge port. Only one live runtime may own a profile.
A profile's `display.language` selects English, French, or Arabic for static
runtime messages; the agent's own language and register remain in `SOUL.md`.
The top-level `timezone` is shared by cron and daily conversation resets.
A named profile may fall back only to the default profile's ChatGPT
authentication. An optional `AGENTS.md` in
the working directory supplies workspace instructions to each new conversation.

Capabilities are controlled by `tools.enabled` and `tools.disabled`, including
channel overrides; disabled groups cannot be enabled by a client response.
`cron.enabled: false` stops scheduled execution and blocks scheduling requests.
Chat-created jobs keep their originating channel's execution settings even when
their delivery destination changes. Disabling that channel also blocks job
activation and execution until it is enabled again. Jobs without a chat origin
use the common profile settings, matching the operator CLI.
Enabled requests run without a client approval step. Legacy `approvals.*`
settings remain accepted for existing profiles but no longer affect execution;
they are not feature switches. Memory and skill changes still require the
foreground execution boundary, configuration access, validation and the existing
rollback journal. Messaging replies contain plain client messages; operator
diagnostics remain available through the CLI and logs.

`gateway.media_delivery_allow_dirs` is the complete native-file delivery allowlist
when configured: only those directories are allowed, and `[]` disables file
delivery. When omitted, the profile workspace remains the default. Existing
configurations needing that workspace as well must now list it explicitly.
This controls file paths, not disclosure of copied content or text.

Recovery suppresses recognized obsolete technical notices from earlier versions,
retaining their delivery records and any evidence of already accepted messages.
An old attachment warning appended to a business reply is replaced with a plain
localized notice; accepted chunks are never resent.
Other pending replies keep their normal delivery recovery.

DOCX/XLSX reading caps both the input file and the total expanded XML read at
50 MiB and supports stored or DEFLATE-compressed XML. Excessive expansion and
unsupported XML compression are rejected with a plain message.

## Verify changes

After installing the locked environments:

```bash
./.venv/bin/python -m unittest discover -s tests
npm --prefix bridge test
```

## Origin

Pilotage Agent builds on [Hermes Agent](https://hermes-agent.nousresearch.com) by
Nous Research, used under the MIT License. It selectively reuses proven Hermes
mechanisms and code while keeping a smaller Pilotage-owned runtime shape.

The original copyright notice is retained in [LICENSE](LICENSE), as the MIT
License requires.
