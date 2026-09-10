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
GPT-6 Astra, WhatsApp and Telegram, persistent context and memory, selected tools
and skills, and scheduled work.

It is still evolving and has no stable public API or compatibility guarantee.

## Install on Ubuntu

For a fresh protected LXC deployment, follow
[Pilotage Deploy](https://github.com/zavrenn/pilotage-deploy). The original
single-account installation remains available for development and existing
deployments; `pilotage update` does not silently migrate them.

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
./.venv/bin/pilotage model     # confirm Astra access and choose its default effort
./.venv/bin/pilotage whatsapp  # optional: configure, pair, and enable WhatsApp
./.venv/bin/pilotage telegram  # optional: configure, verify, and enable Telegram
./.venv/bin/pilotage run
```

Voice-message transcription requires `VOICE_TOOLS_OPENAI_KEY` in the agent
`.env`; ChatGPT login does not authorize the OpenAI audio API.

Full-page web extraction requires `FIRECRAWL_API_KEY` in the agent `.env`, or
`FIRECRAWL_API_URL` for a self-hosted Firecrawl instance. DDGS search needs no
key. Without either Firecrawl setting, the agent offers search only and Doctor
does not require extraction credentials. Restart after configuring extraction.

`pilotage whatsapp` saves the WhatsApp allowlist and home destination in the
effective agent environment file, then pairs and enables the channel.
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
pilotage model
pilotage update --check
pilotage update
pilotage restart
pilotage logs -f --level WARNING
pilotage logs --since 1h -n 100
pilotage service status
pilotage service stop
pilotage service start
pilotage cron list --all
```

The installer links `pilotage` into `~/.local/bin`; add that directory to your
shell's `PATH` if needed, or keep using `./.venv/bin/pilotage`.

`pilotage model` is an interactive menu. It checks Astra availability using this
agent's ChatGPT login, then saves the chosen default effort (`high` initially).
It accepts no model names or effort flags. Restart with `pilotage restart` to
apply saved defaults.
Protected installations run this command as the operator; credentials are checked
by the unprivileged runtime account. No model fallback is selected if Astra is
unavailable.

In WhatsApp or Telegram, `/effort` reads the session's effort without resetting
the conversation or updating its activity. It checks the same account options as
`pilotage model`; `/effort high` selects an available level from `low`, `medium`,
`high`, `xhigh`, or `max`. Unavailable levels are rejected. If the account check
fails, the current effort can still be read, and changes are refused.
The choice applies to the next request, survives restarts, and is cleared by
`/new` or an automatic session reset. An active or recovered turn keeps its
starting effort. Other conversations and scheduled jobs keep their own configured
effort. Existing channel-specific effort overrides still take precedence over the
agent default.

`update --check` fetches Git metadata without changing code or restarting anything.
`update` follows the current branch's configured upstream and refuses local edits
or commits ahead of upstream.
It stops the service, fast-forwards the checkout, runs the locked installer
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

Each container runs one agent with its own `SOUL.md` identity, configuration,
ChatGPT authentication, channel sessions, conversations, memory, skills,
workspace, and cron jobs. These remain under `~/.pilotage-agent`.
There is no profile manager, state selection, or authentication fallback.
Only one live runtime may own the agent state directory.
`display.language` selects English, French, or Arabic for static
runtime messages; the agent's own language and register remain in `SOUL.md`.
The top-level `timezone` is shared by cron and daily conversation resets.
An optional `AGENTS.md` in
the working directory supplies workspace instructions to each new conversation.

The system service is `pilotage-agent.service` and starts with `pilotage run`.
Chats and participants have separate conversation state within the agent;
the container provides the filesystem security boundary.

Capabilities are controlled by `tools.enabled` and `tools.disabled`, including
channel overrides; disabled groups cannot be enabled by a client response.
`cron.enabled: false` stops scheduled execution and blocks scheduling requests.
Chat-created jobs keep their originating channel's execution settings even when
their delivery destination changes. Disabling that channel also blocks job
activation and execution until it is enabled again. Jobs without a chat origin
use the common agent settings, matching the operator CLI.
Enabled requests run without a client approval step. Legacy `approvals.*`
settings remain accepted for existing installations but no longer affect execution;
they are not feature switches. Memory and skill changes still require the
foreground execution boundary, configuration access, validation and the existing
rollback journal. Messaging replies contain plain client messages; operator
diagnostics remain available through the CLI and logs.

`gateway.media_delivery_allow_dirs` is the complete native-file delivery allowlist
when configured: only those directories are allowed, and `[]` disables file
delivery. When omitted, the agent workspace remains the default. Existing
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

## Protected LXC deployment

The `operator` account has sudo and owns a clean runtime checkout at
`/opt/pilotage-agent`. The `agent` account has no sudo. Its workspace, memory,
skills, cron data, channel sessions and authentication refresh remain writable.
Runtime code, `config.yaml`, `.env`, `SOUL.md`, and the system service are
protected from agent edits, including replacement of their containing folders.
Operator Git credentials stay in the private operator home.
Protected deployments reject alternate `PILOTAGE_CONFIG` and `PILOTAGE_ENV_FILE`
paths. Keep these files directly in the agent state directory.

The [Pilotage Deploy bootstrap](https://github.com/zavrenn/pilotage-deploy)
prepares this layout automatically for new containers.
For an existing container, first prepare a **fresh trusted checkout** under the
operator account; never run a root installer from the old agent-owned checkout.
As root (skip account creation if `operator` already exists):

```bash
getent group operator >/dev/null || addgroup operator
adduser --ingroup operator operator
install -d -o operator -g operator -m 0755 /opt/pilotage-agent
runuser -l operator -c 'git clone https://github.com/zavrenn/pilotage-agent.git /opt/pilotage-agent'
cd /opt/pilotage-agent
bash scripts/setup-accounts.sh operator
```

As `operator`, install the dependencies:

```bash
cd /opt/pilotage-agent
bash scripts/install.sh --dependencies-only
```

During a planned cutover, stop the old services and agent login sessions, and
relocate/revoke any deployment Git credentials left under `/home/agent`.
Then run as root from the new checkout:

```bash
bash scripts/install-protection.sh operator
./.venv/bin/python -I -B scripts/verify-protection.py
```

The installer preserves existing agent data, disables
the legacy user-service autostart, and installs a **stopped** system service.
It refuses running agent processes, unexpected privileges and linked protected
files. Provision a separate container for each additional agent.

Use `pilotage login`, `whatsapp`, `telegram`, `restart`, `doctor`, `update`, and
`logs -f --level WARNING` from the operator account. State operations run as
`agent`; administration uses the operator's normal sudo authentication.
Log in again after account setup to pick up the new group membership.
Run `restart` before `doctor`, which checks the live deployment.
Every `pilotage update` restores runtime read/execute permissions and verifies
the updated import as `agent` before restarting.

Install the agent asset repository once into the live `/home/agent` checkout.
After a fresh bootstrap, run as the operator (replace the URL with your agent):

```bash
sudo /opt/pilotage-agent/.venv/bin/python -I -B /opt/pilotage-agent/scripts/install-agent-checkout.py https://github.com/zavrenn/demo-agent.git
cd /home/agent
git status
```

The installer preserves runtime state and `.env`, replaces bootstrap config and
identity, and refuses existing asset conflicts or an existing checkout. Git runs
as the operator, with its authentication in the operator home. `/home/agent`
and `.git` belong to the operator; `.git` is private. Skills have inherited ACLs
so both accounts can edit them without granting agent access to Git or settings.
The only retained asset checkout is `/home/agent`; there is no copy step.

Review agent edits using `git status` and `git diff` as the operator. Stop the
service, review and commit edits you want to keep, then use `git pull --no-rebase`
to merge upstream updates while keeping those commits. If there are conflicts,
keep the service stopped, resolve them and finish the merge before restarting.
Use `git merge --abort` to cancel an unfinished merge. Do not run Git as root or
agent, or make the entire home agent-writable.
The checkout installer requires the current fresh bootstrap and the `acl`
system package; it does not migrate an existing deployed agent.

This prevents runtime/configuration modification. It does not conceal the
instructions or runtime credentials that the executing agent must read.
The folder protection uses Linux's [sticky-directory ownership checks](https://man7.org/linux/man-pages/man2/unlink.2.html).

## Verify changes

Run tests in a separate development checkout, never in the protected live
installation. After installing the locked environments there:

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
