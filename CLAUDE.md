# Pilotage Agent

A lightweight runtime for focused, long-lived agents, owned end to end. It runs
headless, talks to its users over messaging channels, and is deliberately small:
one model service, few moving parts, full control over memory, messaging and
runtime behavior.

It is being built up from the upstream Hermes Agent source (Nous Research, MIT)
step by step. It is not a fork and not a trimmed copy — the fork was abandoned
once the fixes we needed became architectural rather than incremental. Building
the runtime is the work.

@AGENTS.md
@CONTEXT.local.md

If those two files were not loaded with this one, read them now before doing
anything else. `AGENTS.md` is binding. `CONTEXT.local.md` is private evidence:
the target agent, the confirmed boundaries, and the dead ends that are not worth
re-exploring.
