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
context, retain useful knowledge, use tools, and communicate through familiar
messaging channels. Its planned initial scope is ChatGPT, WhatsApp, and Telegram.

New capabilities will be added from concrete use cases, not to pursue framework
completeness.

## Principles

- Focused rather than universal.
- Explicit behavior rather than hidden complexity.
- Strong isolation and clear trust boundaries.
- Reliability before feature count.
- Evidence before expansion.

## Status

Pilotage Agent is in early development. A runtime core is now in place, but there
is no stable public API yet and interfaces change without notice.

## Origin

Pilotage Agent builds on [Hermes Agent](https://hermes-agent.nousresearch.com) by
Nous Research, used under the MIT License. Upstream supplies the gateway, session
handling, messaging adapters, and agent loop. Pilotage removes what it does not
need and reshapes the rest around the principles above.

The original copyright notice is retained in [LICENSE](LICENSE), as the MIT
License requires.
