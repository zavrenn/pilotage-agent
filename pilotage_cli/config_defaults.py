"""Default configuration data for Pilotage Agent.

Pure-data leaf module: DEFAULT_CONFIG and OPTIONAL_ENV_VARS, extracted
verbatim from pilotage_cli/config.py. Must not import from pilotage_cli.config.
"""

DEFAULT_CONFIG = {
    "model": "",
    "providers": {},
    "fallback_providers": [],
    "credential_pool_strategies": {},
    "toolsets": ["pilotage-cli"],
    # SQLite journal mode used by every Pilotage database opener. WAL is the
    # normal default; set DELETE for weak-fsync/shared filesystems where WAL is
    # not crash-safe (for example macOS virtiofs, NFS, or SMB).
    "database": {
        "journal_mode": "wal",
        # Optional WAL sizing pragmas, applied when set to integers.
        # None = SQLite defaults (autocheckpoint 1000 pages, no size limit).
        "wal_autocheckpoint": None,
        "journal_size_limit": None,
    },
    # Soft file-descriptor limit for long-running Pilotage server processes.
    # Clamped to the OS hard limit; 0/false/null disables the adjustment.
    "runtime": {
        "nofile_soft_limit": 4096,
    },
    # Global active chat session cap across CLI, TUI/dashboard, and messaging.
    # None/0 = unbounded.
    "max_concurrent_sessions": None,
    # Soft LRU cap on in-memory TUI/desktop/dashboard sessions. When more than
    # this many are live, the gateway evicts the least-recently-active DETACHED
    # sessions (no live client) so accumulated agents don't pile up under memory
    # pressure. Reopening one re-resumes it from disk. 0/null disables.
    "max_live_sessions": 16,
    "agent": {
        "max_turns": 500,
        # Inactivity timeout for gateway agent execution (seconds).
        # The agent can run indefinitely as long as it's actively calling
        # tools or receiving API responses.  Only fires when the agent has
        # been completely idle for this duration.  0 = unlimited.
        "gateway_timeout": 1800,
        # Maximum time an alias routing key waits for the active turn holding
        # the same resolved session lease. On expiry the inbound message is
        # rejected with a resend notice rather than run without serialization.
        # Non-positive values fall back to 1800 seconds.
        "gateway_turn_lease_timeout": 1800,
        # Per-session AIAgent cache in the gateway. Each cached agent keeps a
        # warm prompt prefix AND the session's full transcript, so the cache
        # trades memory for cost: too small and every turn re-pays an uncached
        # prompt, too large and tool-heavy transcripts fill the heap.
        "agent_cache": {
            # LRU entry cap.
            "max_size": 128,
            # Evict an agent that has been idle this long (seconds).
            "idle_ttl_secs": 3600,
            # Anonymous-RSS budget (MB) above which the gateway starts shedding
            # least-recently-used transcripts, which reload from the persisted
            # session on the next turn. "auto" derives the budget from the
            # cgroup memory limit the gateway runs under (or total RAM when
            # uncapped); a number sets it explicitly; 0/off disables the pass
            # and lets memory grow to whatever the two bounds above allow.
            "memory_high_mb": "auto",
            # Upper bound on how many sessions one pressure pass sheds, so a
            # burst of teardowns cannot stall the gateway.
            "max_evictions_per_pass": 16,
            # Most-recently-used sessions the pressure pass never touches —
            # they are the ones actively paying for a warm prompt cache.
            "protect_recent": 8,
        },
        # Force-interrupt budget once gateway stop()/drain has begun
        # (seconds). Applies to SIGTERM/external stop and to the final
        # phase of in-band restart after any after-turn wait. 0 = interrupt
        # immediately (the default).
        #
        # Keep this short and under systemd TimeoutStopSec — a long value
        # here invites SIGKILL-mid-cleanup. For in-band restart
        # (/restart, SIGUSR1), prefer restart_after_turn_timeout below so
        # active turns finish *before* stop begins.
        "restart_drain_timeout": 0,
        # Cron-only floor under the stop()/drain wait (seconds). A chat turn
        # interrupted by a restart is announced to the user and resumed on
        # their next message; an interrupted cron run is written to jobs.json
        # as a permanent failure that nobody is waiting on, so it must not
        # inherit restart_drain_timeout's 0. Clamped at runtime to
        # the shutdown-watchdog leash minus teardown headroom, so raising it
        # past ~50s has no effect unless TimeoutStopSec is raised too.
        # 0 = opt out (cron drains on restart_drain_timeout, legacy).
        "cron_drain_timeout": 30,
        # In-band restart wait for active turns to finish before stop()
        # (seconds). /restart and SIGUSR1 refuse new work, then wait up to
        # this cap for in-flight agents/cron/api runs to complete naturally
        # so the requesting turn is not amputated by restart_drain_timeout.
        # 0 = legacy behaviour (enter stop()/drain immediately). Default
        # 30 min is a safety valve for wedged agents, not a target latency —
        # an interactive `pilotage gateway restart` must never block for hours
        # on a turn that wedged. Long unattended turns can raise
        # this in config.yaml.
        "restart_after_turn_timeout": 1800,
        # Upper bound (seconds) a submitted prompt waits for the deferred
        # agent build (MCP discovery, model metadata, skills scan) before
        # failing with a visible error. The gateway's wait is
        # patient — the prompt is delivered the moment the build completes
        # and a progress notice is emitted past 30s — so this cap only fires
        # on a genuinely hung build. Raise it for deployments with many slow
        # or unreachable MCP servers.
        "build_wait_timeout": 600,
        # Max app-level retry attempts for API errors (connection drops,
        # provider timeouts, 5xx, etc.) before the agent surfaces the
        # failure.  The OpenAI SDK already does its own low-level retries
        # (max_retries=2 default) for transient network errors; this is
        # the Pilotage-level retry loop that wraps the whole call.  Lower
        # this to 1 if you use fallback providers and want fast failover
        # on flaky primaries; raise it if you prefer to tolerate longer
        # provider hiccups on a single provider.
        "api_max_retries": 3,
        "service_tier": "",
        # Tool-use enforcement: injects system prompt guidance that tells the
        # model to actually call tools instead of describing intended actions.
        # Values: "auto" (default — applies to gpt/codex models), true/false
        # (force on/off for all models), or a list of model-name substrings
        # to match (e.g. ["gpt", "codex"]).
        "tool_use_enforcement": "auto",
        # Intent-ack continuation: when the model opens a turn by narrating an
        # action it will take ("I'll go check the logs...") but emits no tool
        # call, intercept the turn-end, inject a "continue now, execute the
        # tools" nudge, and loop instead of ending the turn (capped at 2 nudges
        # per turn). This is the corrective sibling of tool_use_enforcement (the
        # preventive prompt-side guard). Values: "auto" (default — fires only on
        # the codex_responses api_mode, the historical behavior), true (all
        # api_modes — fixes the weak-model "stops after stating intent" case),
        # false (never), or a list of model-name substrings to match.
        "intent_ack_continuation": "auto",
        # Universal "finish the job" guidance — short prompt block applied to
        # all models that targets two cross-family failure modes: (1) stopping
        # after a stub instead of finishing the artifact, (2) fabricating
        # plausible-looking output when a real path is blocked.  Costs ~80
        # tokens in the cached system prompt.  Set False to disable globally.
        "task_completion_guidance": True,
        # Universal parallel-tool-call guidance — short prompt block applied to
        # all models that tells the model to batch independent tool calls
        # (reads, searches, web fetches, read-only commands) into one turn
        # instead of one call per turn.  The runtime already runs independent
        # calls concurrently, so this just steers the model to produce the
        # batch — cutting round-trips and the resent-context cost that
        # compounds over a long conversation.  Costs ~70 tokens in the cached
        # system prompt.  Set False to disable globally.
        "parallel_tool_call_guidance": True,
        # Local-environment toolchain probe — surfaces Python/pip/uv/PEP-668
        # state in the system prompt when something non-default is detected
        # (e.g. python3 has no pip module, pip→python version mismatch, PEP
        # 668 enforcement without uv).  Costs zero tokens when the env is
        # clean (probe emits nothing).  Skipped for remote terminal backends
        # (docker/modal/ssh — they have their own probe).  Set False to
        # disable entirely.
        "environment_probe": True,
        # Embedder-supplied environment description appended to the system
        # prompt's environment-hints block. Lets a host that wraps Pilotage
        # (sandbox runner, managed platform) explain the runtime environment
        # — proxy, credential handling, mount layout — without editing the
        # identity slot (SOUL.md). Empty by default. The PILOTAGE_ENVIRONMENT_HINT
        # env var overrides this (build-time/container mechanism).
        "environment_hint": "",
        # Coding posture — on interactive coding surfaces (CLI, TUI, desktop
        # app, ACP) in a code workspace, Pilotage adds a coding operating brief
        # + a live git/workspace snapshot to the system prompt. See
        # agent/coding_context.py.
        #   "auto" (default) — prompt-only posture when the surface is
        #                      interactive AND cwd is a code workspace.
        #                      Toolsets are never touched; messaging platforms
        #                      unaffected.
        #   "focus"          — auto + collapse the toolset to the lean coding
        #                      set (+ enabled MCP servers) + demote non-coding
        #                      skill categories to names-only in the prompt's
        #                      skill index. Explicit opt-in.
        #   "on"             — force the prompt posture everywhere.
        #   "off"            — disable entirely.
        "coding_context": "auto",
        # Standing operator instructions for the coding posture. A string (or
        # list of strings) appended to the coding brief as an extra stable
        # system block — pin project-wide workflow rules here instead of editing
        # the shipped brief, e.g. "For UI work, don't run tsc/lint until I
        # approve. Clean the diff before you commit and push." Cache-safe:
        # takes effect next session. Empty by default.
        "coding_instructions": "",
        # When verify-on-stop finds edited code without fresh verification
        # evidence, append guidance for creative UI work (avoid broad
        # tsc/lint/test before visual approval) and clean-diff expectations.
        # Set false to keep the evidence nudge terse.
        "verify_guidance": True,
        # Upper bound on consecutive `pre_verify` "continue" nudges in a single
        # turn, so a user/plugin hook can never trap the loop.
        "max_verify_nudges": 3,
        # Verification closure: after the agent edits files in a code workspace,
        # do not accept a final answer until fresh verification evidence exists
        # or the agent explains why it cannot run checks. The loop is bounded
        # and uses the passive verification ledger. Default is False (opt-in):
        # the v31/v32 config migrations already switch existing installs off
        # because the verification narrative proved more noise than signal,
        # and the docs tell users to treat off as the effective default — a
        # fresh install must not be the one population that still gets the
        # nudges. Set true to force on everywhere, or "auto" for the legacy
        # surface-aware behavior (on for interactive coding surfaces — CLI,
        # TUI, desktop — and programmatic callers, off for conversational
        # messaging surfaces). Doc/markdown/skill-only edits never fire it.
        "verify_on_stop": False,
        # Staged inactivity warning: send a warning to the user at this
        # threshold before escalating to a full timeout.  The warning fires
        # once per run and does not interrupt the agent.  0 = disable warning.
        "gateway_timeout_warning": 900,
        # Maximum time (seconds) the gateway will block an agent waiting for
        # a clarify-tool response from the user.  Hit this and the agent
        # unblocks with "[user did not respond within Xm]" so it can adapt
        # rather than pinning the running-agent guard forever.  CLI clarify
        # blocks indefinitely (input() is synchronous) and ignores this.
        # Default 3600 (1h): real users step away (meetings, AFK) and the
        # old 600s default evicted the entry mid-think, so a later button
        # tap landed on a dead entry. Tradeoff: a higher value
        # holds the gateway's running-agent guard longer for a genuinely
        # abandoned prompt — lower it if a single session must free up the
        # guard sooner.
        "clarify_timeout": 3600,
        # Periodic "still working" notification interval (seconds).
        # Sends a status message every N seconds so the user knows the
        # agent hasn't died during long tasks.  0 = disable notifications.
        # Lower values mean faster feedback on slow tasks but more chat
        # noise; 180s is a compromise that catches spinning weak-model runs
        # (60+ tool iterations with tiny output) before users assume the
        # bot is dead and /restart.
        "gateway_notify_interval": 180,
        # Session stall watchdog (seconds). Scope: this is a
        # RECOVERY notifier for an in-process AIAgent that has an
        # adapter-queued follow-up (pending inbound / queued event) while its
        # activity clock is stale — NOT a general gateway/session stall
        # detector. It does not observe startup restoration, build sentinels,
        # turn leases, debounce state, or work owned by another process; the
        # scan cadence is per AIAgent instance, not globally coordinated per
        # durable session. Notify-only: warns the user to try /new. Distinct
        # from gateway_timeout (which kills the turn) and
        # gateway_notify_interval ("still working" heartbeats). 0 = disable.
        "session_stall_timeout": 300,
        # Long-lived reconnect-loop escalation (seconds). A platform that has
        # been continuously failing/reconnecting for this long gets
        # needs_attention flagged in gateway runtime status (visible in
        # `pilotage status` / fleet monitoring). Retries never stop — this is a
        # signal, not a circuit breaker. 0 = disable.
        "reconnect_attention_after": 7200,
        # Freshness window for the gateway auto-continue note (seconds).
        # After a gateway crash/restart/SIGTERM mid-run, the next user
        # message gets a "[System note: your previous turn was
        # interrupted — process the unfinished tool result(s) first]"
        # prepended so the model picks up where it left off.  That's the
        # right behaviour while the interruption is fresh, but stale
        # markers (transcript last touched hours or days ago) can revive
        # an unrelated old task when the user's next message starts new
        # work.  This window is the max age of the last persisted
        # transcript row for which we still inject the continue note.
        # Default 3600s comfortably covers a long turn (gateway_timeout
        # default is 1800s) plus runtime slack.  Set to 0 to disable the
        # gate and restore pre-fix behaviour (always inject).
        "gateway_auto_continue_freshness": 3600,
        # Max seconds the gateway waits for boot auto-resume turns to finish
        # before it releases the startup-restore inbound gate.  While startup
        # restore is in progress the gateway QUEUES every inbound message
        # instead of replying, so no channel gets an answer until this gate
        # opens.  Without a bound, one pathologically long resumed turn holds
        # the gate shut and every channel's inbound piles up unanswered for as
        # long as that turn runs.  On timeout the gate releases and the slow
        # resume turn keeps running in the background; duplicate-agent
        # protection is unaffected because the resume slot is claimed
        # synchronously before the gate runs.  Set to 0 to disable the bound
        # (historical "wait forever" behaviour).
        "gateway_startup_restore_drain_timeout": 30,
        # Stale-stream ceiling for local endpoints in
        # seconds. When the base stale timeout is at its default (180s) and a
        # local endpoint is detected, this finite ceiling replaces the former
        # infinite disable so a wedged local server eventually trips the
        # detector instead of hanging forever. The env var
        # ``PILOTAGE_LOCAL_STREAM_STALE_TIMEOUT`` overrides for escape-hatch use.
        "local_stream_stale_timeout": 900,
        # How user-attached images are presented to the main model on each turn.
        #   "auto"   — attach natively when the active model reports
        #              supports_vision=True AND the user hasn't explicitly
        #              configured auxiliary.vision.provider.  Otherwise fall
        #              back to text (vision_analyze pre-analysis).
        #   "native" — always attach natively; non-vision models will either
        #              error at the provider or get a last-chance text fallback
        #              (see run_agent._prepare_messages_for_api).
        #   "text"   — always pre-analyze with vision_analyze and prepend the
        #              description as text; the main model never sees pixels.
        # Affects gateway platforms, the TUI, and CLI /attach.  vision_analyze
        # remains available as a tool regardless of this setting — the routing
        # only controls how inbound user images are presented.
        "image_input_mode": "auto",
        "disabled_toolsets": [],

        # Per-model reasoning effort overrides (spelling-tolerant).
        # Dict mapping model names (any reasonable spelling) to effort levels.
        # Takes precedence over agent.reasoning_effort when the current model
        # matches a key in this dict.
        # Edit directly in config.yaml (no CLI support due to dots in keys).
        "reasoning_overrides": {},
    },

    "terminal": {
        "backend": "local",
        # Remote-backend graceful degradation: when a connection-class
        # infrastructure failure occurs (SSH host unreachable, Docker daemon
        # down), "warn" (default) returns a structured degraded tool result
        # with a reason + retry hint so the model can act on it; "fail"
        # preserves the historical error + traceback behavior.
        "degraded_mode": "warn",
        "cwd": ".",  # Use current directory
        # Terminal font family for the desktop app's embedded xterm.js terminal.
        # When set (e.g. "'CaskaydiaCoveNerdFont', 'JetBrains Mono', monospace"),
        # the desktop terminal uses this as the CSS font-family value, with the
        # built-in default ("'JetBrains Mono', 'Cascadia Code', 'SF Mono', Menlo,
        # Consolas, monospace") as fallback when the field is empty or unset.
        # This lets users install a Nerd Font (or any custom font) and configure
        # it here without patching the built desktop app.
        "font_family": "",
        "timeout": 180,
        # Bounded grace period (seconds) between SIGTERM and an escalated
        # SIGKILL when terminating a host process tree (browser daemons, etc.).
        # A daemon that stalls in its SIGTERM handler is force-killed after this
        # window so it can't leak indefinitely. 0 disables escalation (SIGTERM
        # only — the historical behavior). Floored internally at 0.
        "daemon_term_grace_seconds": 2.0,
        # Environment variables to pass through to sandboxed execution
        # (terminal and execute_code).  Skill-declared required_environment_variables
        # are passed through automatically; this list is for non-skill use cases.
        "env_passthrough": [],
        # HOME handling for host tool subprocesses:
        #   auto    — host keeps the real OS-user HOME; containers use
        #             PILOTAGE_HOME/home for persistent state (default)
        #   real    — force the real OS-user HOME
        #   profile — force PILOTAGE_HOME/home when it exists (old strict
        #             per-profile CLI config isolation)
        "home_mode": "auto",
        # Extra files to source in the login shell when building the
        # per-session environment snapshot.  Use this when tools like nvm,
        # pyenv, asdf, or custom PATH entries are registered by files that
        # a bash login shell would skip — most commonly ``~/.bashrc``
        # (bash doesn't source bashrc in non-interactive login mode) or
        # zsh-specific files like ``~/.zshrc`` / ``~/.zprofile``.
        # Paths support ``~`` / ``${VAR}``. Missing files are silently
        # skipped. When empty, Pilotage auto-sources ``~/.profile``,
        # ``~/.bash_profile``, and ``~/.bashrc`` (in that order) if the
        # snapshot shell is bash (this is the ``auto_source_bashrc``
        # behaviour — disable with that key if you want strict login-only
        # semantics).
        "shell_init_files": [],
        # When true (default), Pilotage sources the user's shell rc files
        # (``~/.profile``, ``~/.bash_profile``, ``~/.bashrc``) in the
        # login shell used to build the environment snapshot. This
        # captures PATH additions, shell functions, and aliases — which a
        # plain ``bash -l -c`` would otherwise miss because bash skips
        # bashrc in non-interactive login mode, and because a default
        # Debian/Ubuntu ``~/.bashrc`` short-circuits on non-interactive
        # sources. ``~/.profile`` and ``~/.bash_profile`` are tried first
        # because ``n`` / ``nvm`` / ``asdf`` installers typically write
        # their PATH exports there without an interactivity guard. Turn
        # this off if your rc files misbehave when sourced
        # non-interactively (e.g. one that hard-exits on TTY checks).
        "auto_source_bashrc": True,
        "docker_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "docker_forward_env": [],
        # Explicit environment variables to set inside Docker containers.
        # Unlike docker_forward_env (which reads values from the host process),
        # docker_env lets you specify exact key-value pairs — useful when Pilotage
        # runs as a systemd service without access to the user's shell environment.
        # Example: {"SSH_AUTH_SOCK": "/run/user/1000/ssh-agent.sock"}
        "docker_env": {},
        "singularity_image": "docker://nikolaik/python-nodejs:python3.11-nodejs20",
        "modal_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        "daytona_image": "nikolaik/python-nodejs:python3.11-nodejs20",
        # Vercel Sandbox runtime (vercel_sandbox backend only).
        # Supported: node24, node22, python3.13.
        "vercel_runtime": "node24",
        # Container resource limits (docker, singularity, modal, daytona, vercel_sandbox — ignored for local/ssh)
        "container_cpu": 1,
        "container_memory": 5120,       # MB (default 5GB)
        "container_disk": 51200,        # MB (default 50GB)
        "container_persistent": True,   # Persist filesystem across sessions
        # Docker volume mounts — share host directories with the container.
        # Each entry is "host_path:container_path" (standard Docker -v syntax).
        # Example:
        # ["/home/user/projects:/workspace/projects",
        #  "/home/user/.pilotage/cache/documents:/output"]
        # For gateway MEDIA delivery, write inside Docker to /output/... and emit
        # the host-visible path in MEDIA:, not the container path.
        "docker_volumes": [],
        # Explicit opt-in: mount the host cwd into /workspace for Docker sessions.
        # Default off because passing host directories into a sandbox weakens isolation.
        "docker_mount_cwd_to_workspace": False,
        # Opt-in egress lockdown for Docker terminal sessions. When false,
        # Docker runs with --network=none so commands cannot reach the network.
        "docker_network": True,
        "docker_extra_args": [],        # Extra flags passed verbatim to docker run
        # /dev/shm size for the Docker sandbox. Docker's 64 MB default silently
        # breaks Chromium/Playwright and PyTorch DataLoader workers; tmpfs is
        # lazily allocated so the higher ceiling costs nothing until used.
        # Set to "" (or "0") to omit the flag and use Docker's default.
        "docker_shm_size": "1g",
        # Explicit opt-in: run the Docker container as the host user's uid:gid
        # (via `--user`).  When enabled, files written into bind-mounted dirs
        # (docker_volumes, the persistent workspace, or the auto-mounted cwd)
        # are owned by your host user instead of root, which avoids needing
        # `sudo chown` after container runs. Default off to preserve behavior
        # for images whose entrypoints expect to start as root (e.g. the
        # bundled Pilotage image, which drops to the `pilotage` user via
        # s6-setuidgid inside each supervised service).
        # When on, SETUID/SETGID caps are omitted from the container since
        # no privilege drop is needed.
        "docker_run_as_host_user": False,
        # Persistent shell — keep a long-lived bash shell across execute() calls
        # so cwd/env vars/shell variables survive between commands.
        # Enabled by default for non-local backends (SSH); local is always opt-in
        # via TERMINAL_LOCAL_PERSISTENT env var.
        "persistent_shell": True,
    },

    "web": {
        "backend": "",           # shared fallback — applies to both search and extract
        "search_backend": "",    # per-capability override for web_search (e.g. "searxng")
        "extract_backend": "",   # per-capability override for web_extract (e.g. "native")
        "extract_char_limit": 15000,  # per-page char budget for web_extract; larger pages truncate + store full text in cache/web
    },

    # Filesystem checkpoints — automatic snapshots before destructive file ops.
    # When enabled, the agent takes a snapshot of the working directory once
    # per conversation turn (on first write_file/patch call).  Use /rollback
    # to restore.
    #
    # Defaults changed in v2 (single shared shadow store, real pruning):
    #   - enabled: True -> False   (opt-in; most users never use /rollback)
    #   - max_snapshots: 50 -> 20  (now actually enforced via ref rewrite)
    #   - auto_prune:   False -> True (orphans/stale pruned automatically)
    # Opt in via ``pilotage chat --checkpoints`` or set enabled=True here.
    "checkpoints": {
        "enabled": False,
        # Max checkpoints to keep per working directory.  Pre-v2 this only
        # limited the `/rollback` listing; v2 actually rewrites the ref and
        # garbage-collects older commits.
        "max_snapshots": 20,
        # Hard ceiling on total ``~/.pilotage/checkpoints/`` size (MB).  When
        # exceeded, the oldest checkpoint per project is dropped in a
        # round-robin pass until total size falls under the cap.
        # 0 disables the size cap.
        "max_total_size_mb": 500,
        # Skip any single file larger than this when staging a checkpoint.
        # Prevents accidental snapshotting of datasets, model weights, and
        # other large generated assets.  0 disables the filter.
        "max_file_size_mb": 10,
        # Auto-maintenance: pilotage sweeps the checkpoint base at startup
        # (at most once per ``min_interval_hours``) and:
        #   * deletes project entries whose last_touch is older than
        #     ``retention_days``
        #   * GCs the single shared store to reclaim unreachable objects
        #   * enforces ``max_total_size_mb`` across remaining projects
        #   * deletes ``legacy-*`` archives older than ``retention_days``
        #
        # NOTE: this automatic sweep never deletes "orphan" entries (workdir
        # no longer found on disk). A missing workdir at startup is
        # ambiguous — it can mean the project was deleted, or that an
        # external volume / network share / VPN is simply not mounted yet —
        # and this sweep runs unattended, so it must never guess. Orphan
        # cleanup is only available via the explicit
        # ``pilotage checkpoints prune`` command (add ``--keep-orphans`` to
        # skip it), where a human is looking at the output.
        "auto_prune": True,
        "retention_days": 7,
        "min_interval_hours": 24,
    },

    # Hard cap (chars) for a single automatic context file such as SOUL.md,
    # AGENTS.md, CLAUDE.md, .pilotage.md, or .cursorrules before Pilotage applies
    # head/tail truncation. ``null`` (the default) lets the cap scale with the
    # model's context window (floor 20K, ceiling 500K) so large-context models
    # rarely truncate a project doc. Set a positive integer to pin a fixed cap
    # and override the dynamic behavior. Separate from read_file tool limits.
    "context_file_max_chars": None,

    # Maximum characters returned by a single read_file call.  Reads that
    # exceed this are rejected with guidance to use offset+limit.
    # 100K chars ≈ 25–35K tokens across typical tokenisers.
    "file_read_max_chars": 100_000,

    # Tool-output truncation thresholds. When terminal output or a
    # single read_file page exceeds these limits, Pilotage truncates the
    # payload sent to the model (keeping head + tail for terminal,
    # enforcing pagination for read_file). Tuning these trades context
    # footprint against how much raw output the model can see in one
    # shot. Ported from anomalyco/opencode.
    #
    # - max_bytes:       terminal_tool output cap, in chars
    #                    (default 50_000 ≈ 12-15K tokens).
    # - max_lines:       read_file pagination cap — the maximum `limit`
    #                    a single read_file call can request before
    #                    being clamped (default 2000).
    # - max_line_length: per-line cap applied when read_file emits a
    #                    line-numbered view (default 2000 chars).
    "tool_output": {
        "max_bytes": 50_000,
        "max_lines": 2000,
        "max_line_length": 2000,
    },

    # Tool loop guardrails nudge models when they repeat failed or
    # non-progressing tool calls. Soft warnings are always-on by default;
    # hard stops are opt-in so interactive CLI/TUI sessions keep flowing.
    "tool_loop_guardrails": {
        "warnings_enabled": True,
        "hard_stop_enabled": False,
        "warn_after": {
            "exact_failure": 2,
            "same_tool_failure": 3,
            "idempotent_no_progress": 2,
        },
        "hard_stop_after": {
            "exact_failure": 5,
            "same_tool_failure": 8,
            "idempotent_no_progress": 5,
        },
        # Per-turn runaway-loop caps (inspired by Claude Code v2.1.212,
        # Week 29, July 2026). Hard ceilings on how many times a runaway-prone
        # tool may be called within a SINGLE agent loop (turn); the counters
        # reset at the start of every turn, so a legitimate multi-turn session
        # is never starved. They are always-on and fire regardless of the
        # warn/hard-stop thresholds above. A single turn issuing dozens of web
        # searches or spawning dozens of subagents is already pathological, so
        # the defaults are low. Set either to 0 to disable that cap (unlimited).
        "loop_caps": {
            "max_web_searches": 50,   # max web_search calls per turn (0 = unlimited)
            "max_subagents": 50,      # max subagents spawned per turn (0 = unlimited)
        },
    },

    "compression": {
        "enabled": True,
        "progress_notices": False, # opt-in: when True, routine compression
                                      # progress statuses (compacting/preflight/pre-API/
                                      # idle/retry) are delivered to chat gateway
                                      # platforms instead of being suppressed by the
                                      # gateway noise filter. Default False keeps
                                      # routine compression silent-by-design on chat
                                      # surfaces (server-side logging only). Failure
                                      # notices and manual /compress feedback are
                                      # always visible regardless of this setting.
        "threshold": 0.50,            # compress when context usage exceeds this ratio.
                                      # Models with context windows below 512K are
                                      # floored at 0.75 (raise-only) so compaction
                                      # doesn't fire with half the window still free;
                                      # set this above 0.75 to override the floor.
        "threshold_tokens": None,     # absolute token cap — when set, compression
                                      # triggers at the lower of the ratio-based
                                      # threshold and this token count. Clamped to
                                      # the model's context length at apply-time.
        "target_ratio": 0.20,         # fraction of threshold to preserve as recent tail
        "protect_last_n": 20,         # minimum recent messages to keep uncompressed
        "min_tail_user_messages": 1,  # REAL (actionable) user messages guaranteed to
                                      # survive in the uncompressed tail. 1 = existing
                                      # single last-user anchor (default, behavior-
                                      # preserving); raise to e.g. 3 to keep the last
                                      # 3 real user turns verbatim when bulky tool
                                      # outputs fill the tail token budget.
        "max_attempts": 3,            # compression retry rounds before a turn gives up
                                      # with "max compression attempts reached". Raise
                                      # (e.g. 6) for tool-schema-heavy sessions where 3
                                      # rounds cannot clear the request estimate.
                                      # Validated >= 1, hard-capped at 10.
        "proactive_prune_tokens": 0,  # opt-in trigger (tokens) for the deterministic,
                                      # no-LLM tool-result prune, run independently of
                                      # `threshold` above. On large-window models
                                      # `threshold` (≈50% of the window) rarely fires,
                                      # so old tool output otherwise rides in history
                                      # and is re-sent every turn; a low value like
                                      # 48000 reclaims it early. 0 = off. Recent tail
                                      # protected by `protect_last_n`. Built-in
                                      # compressor only (other engines inherit a no-op).
                                      # NOTE: each committed prune rewrites already-sent
                                      # history, breaking the provider prompt-cache
                                      # prefix — the min_reclaim gate below keeps those
                                      # breaks episodic rather than per-turn.
        "proactive_prune_min_result_chars": 8000,  # the prune's summarize pass only
                                      # touches tool results larger than this (chars);
                                      # clamped to >= 200 so a generated summary can't
                                      # itself be re-summarized.
        "proactive_prune_min_reclaim_tokens": 4096,  # a proactive prune only commits
                                      # when it reclaims at least this many tokens
                                      # (measured on the pruned output), then waits
                                      # for a full trigger-sized token runway to
                                      # regrow before rearming. Keeps prompt-cache
                                      # breaks episodic. 0 = no minimum-savings gate.
        "micro_compact": False,       # opt-in: after each completed turn, fold the
                                      # oldest un-absorbed exchange into a rolling
                                      # summary, amortizing compression cost instead
                                      # of paying it in one batch stall. Default False
                                      # because a pass rewrites already-sent history
                                      # and so breaks the provider prompt-cache prefix
                                      # EVERY turn — the per-turn cache break that
                                      # `proactive_prune_min_reclaim_tokens` above
                                      # exists to avoid. Enable only when you have
                                      # measured that the amortized stall is worth
                                      # more to you than the cached-prefix discount.
                                      # See docs/micro-compaction.md.
        "micro_compact_every_n_turns": 1,  # cadence: run a pass every Nth completed
                                      # turn. Since each pass costs one prompt-cache
                                      # break, this is the dial for how often that
                                      # cost is paid — 1 reclaims most aggressively
                                      # at one break per turn, 5 trades reclaim rate
                                      # for a fifth of the breaks. Clamped to >= 1.
                                      # Ignored unless `micro_compact` is true.
        "micro_compact_defrag_threshold_tokens": 2000,  # once the rolling summary
                                      # exceeds this many tokens, the next pass
                                      # re-summarizes the summary itself instead of
                                      # letting it grow without bound.
        "hygiene_hard_message_limit": 5000,  # gateway session-hygiene force-compress threshold by message count
        "hygiene_timeout_seconds": 30,  # max seconds gateway waits for pre-agent hygiene compression
                                      # WITHOUT forward progress. The summary call streams, so
                                      # this is an inactivity budget: a slow model still
                                      # producing tokens keeps extending the wait; only a
                                      # silent/hung call is cut off.
        "hygiene_total_ceiling_seconds": 600,  # absolute cap on the hygiene compression wait even
                                      # while tokens are still moving — bounds a degenerate
                                      # trickle stream. Clamped to >= hygiene_timeout_seconds.
        "hygiene_failure_cooldown_seconds": 300,  # skip repeated failed hygiene attempts for this session
        "context_timeout_seconds": 120,  # inactivity budget for in-agent compress_context
                                      # (conversation loop, /compress, preflight, etc.).
                                      # Same progress-aware semantics as hygiene_timeout_seconds:
                                      # streamed summary tokens extend the wait; only a silent
                                      # worker is cut off. 0 = disable the owned wrapper
                                      # (callers that already pass commit_fence, e.g. gateway
                                      # hygiene, never use this path).
        "context_total_ceiling_seconds": 600,  # absolute cap on the *pre-commit*
                                      # in-agent compress_context wait (summary /
                                      # stream phase) even while tokens are still
                                      # moving. Clamped to >= context_timeout_seconds
                                      # when the idle budget is > 0. Guarantee:
                                      # the summary phase is bounded by this
                                      # ceiling; an already-started SessionDB
                                      # commit is never abandoned mid-flight —
                                      # if the commit itself runs past the
                                      # ceiling it is logged (WARNING, then
                                      # ERROR) and surfaced to the user via the
                                      # warning channel while the host keeps
                                      # waiting in bounded increments for the
                                      # commit to finish.
        "protect_first_n": 3,         # non-system head messages always preserved
                                      # verbatim, in ADDITION to the system prompt
                                      # (which is always implicitly protected). Set to
                                      # 0 for long-running rolling-compaction sessions
                                      # where you want nothing pinned except the
                                      # system prompt + rolling summary + recent tail.
        "abort_on_summary_failure": False,  # When True, auto-compression that fails
                                      # to generate a summary (aux LLM errored / returned
                                      # non-JSON / timed out) aborts entirely instead of
                                      # dropping the middle window with a static
                                      # "summary unavailable" placeholder.  Messages are
                                      # preserved unchanged and the session "freezes" at
                                      # its current size until the user runs /compress
                                      # (which bypasses the failure cooldown) or /new.
                                      # Default False matches historical behavior; set to
                                      # True if you'd rather pause than silently lose
                                      # context turns when your aux model is flaky.
        "codex_gpt55_autoraise": True,  # Historical key name kept for compatibility.
                                      # When True, gpt-5.4 / gpt-5.5 / gpt-5.6 on the
                                      # ChatGPT Codex OAuth route raise their compaction
                                      # trigger to 85% (vs the global `threshold` above).
                                      # Codex hard-caps these families at a 272K window, so
                                      # the default 50% would compact at ~136K and waste half
                                      # the usable context. Set to False to opt back down to
                                      # the global threshold (e.g. 0.50) for those Codex
                                      # sessions. Only this exact route is affected —
                                      # gpt-5.4 / 5.5 / 5.6 on OpenAI's direct API
                                      # keep the global threshold regardless.
        "codex_gpt55_autoraise_notice": True,  # Display the one-time Codex gpt-5.4/5.5/5.6
                                      # autoraise banner. Set False to keep the
                                      # 85% threshold autoraise but suppress the
                                      # user-facing notice in CLI/gateway output.
        "codex_app_server_auto": "native",  # Codex app-server (codex CLI runtime) thread
                                      # compaction mode. The codex agent owns the real
                                      # thread context, so Pilotage' summarizer cannot
                                      # shrink it. native = codex decides when
                                      # to compact its own thread (default); pilotage =
                                      # Pilotage' compression threshold triggers
                                      # thread/compact/start; off = never auto-trigger
                                      # (codex may still compact natively).
        "codex_responses_native": False,  # Opt in to OpenAI's server-side compaction
                                      # on the Responses API. Engages ONLY for
                                      # gpt-5.6-family models on api.openai.com or
                                      # the ChatGPT Codex backend; every other
                                      # route/model is unaffected. Pilotage' local
                                      # compression stays armed as the fallback.
        "codex_responses_compact_threshold": 200000,  # Server-side compaction trigger
                                      # (input tokens). Clamped below the local
                                      # compression threshold at request time so
                                      # the server compacts before Pilotage does.
        "in_place": True,             # When True, compaction rewrites the message
                                      # list and rebuilds the system prompt WITHOUT
                                      # rotating the session id — the conversation
                                      # keeps one durable id for its whole life
                                      # (no parent_session_id chain, no `name #N`
                                      # renumbering). Eliminates the session-rotation
                                      # bug cluster /goal loss, lost
                                      # response, orphans, search gaps,
                                      # null cwd) — see. Non-destructive:
                                      # the live context is compacted (lossy for what
                                      # the model reloads), but the pre-compaction
                                      # turns are soft-archived under the same id
                                      # (active=0, compacted=1) — still searchable via
                                      # session_search and recoverable, not deleted.
                                      # Default True since 2107b86024; set False to
                                      # restore the legacy rotating-compaction path.
        "model_thresholds": {},       # Per-model threshold overrides. Keys are
                                      # substring-matched against the model name
                                      # (longest match wins); values replace the
                                      # global `threshold` for that model, e.g.
                                      #   model_thresholds:
                                      #     "gpt-5.6": 0.40
                                      #     "gpt-5.4": 0.35
                                      # The small-context floor (0.75 for <512K
                                      # models) still applies on top of overrides
                                      # (raise-only: an override above the floor
                                      # wins; one below it is raised to the floor).
        "idle_compact_after_seconds": 0,  # Opt-in idle compaction (0 = disabled).
                                      # When > 0, a session that resumes after at
                                      # least this many seconds of inactivity
                                      # compacts its accumulated history up front,
                                      # before the first reply — so a long-lived
                                      # thread resumed hours later doesn't re-read
                                      # its full stale context on every turn.
                                      # Time-based; complements (does not replace)
                                      # the size-based `threshold` above. Skipped
                                      # when the context is already at/below the
                                      # post-compression target (threshold ×
                                      # target_ratio) and it honors the same
                                      # failure-cooldown / anti-thrash / per-session
                                      # lock guards as every automatic compaction.
                                      # Example: 1800 = compact after 30 min idle.
    },

    # Auxiliary model config — provider:model for each side task.
    # Format: provider is the provider name, model is the model slug.
    # "auto" for provider = auto-detect best available provider.
    # Empty model = use provider's default auxiliary model.
    #
    # extra_body: forwarded verbatim as request body fields on every aux call
    # for that task. Use this to set provider-specific knobs (independent of
    # main-agent settings).
    "auxiliary": {
        # Same-provider retries for a transient transport blip (connection
        # reset / timeout / 5xx / 408) on ANY auxiliary call before falling
        # back. Default 2 (→ 3 total attempts), clamped [0,6]. Matters most for
        # pinned calls like MoA reference advisors, where provider fallback is
        # not a meaningful recovery, so an unretried blip silently loses the
        # call.
        "transient_retries": 2,
        # Endpoints that reject NON-streaming chat requests outright.
        # Auxiliary calls to a matching endpoint are sent with stream=True
        # and aggregated client-side. Entries are case-insensitive
        # substrings matched against the endpoint URL.
        "stream_only_base_urls": [],
        "vision": {
            "provider": "auto",    # auto | codex | custom
            "model": "",           # e.g. "gpt-4o"
            "base_url": "",        # direct OpenAI-compatible endpoint (takes precedence over provider)
            "api_key": "",         # API key for base_url (falls back to OPENAI_API_KEY)
            "timeout": 120,        # seconds — LLM API call timeout; vision payloads need generous timeout
            "extra_body": {},      # OpenAI-compatible provider-specific request fields
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
            "download_timeout": 30,  # seconds — image HTTP download timeout; increase for slow connections
        },
        "web_extract": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 360,        # seconds (6min) — per-attempt LLM summarization timeout; increase for slow local models
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "compression": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 120,        # seconds — compression summarises large contexts; increase for local models
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Note: session_search no longer uses an auxiliary LLM ( —
        # single-shape tool returns DB content directly). The old
        # ``auxiliary.session_search.*`` block was removed here. Existing
        # values in user config.yaml files are harmless leftovers and ignored.
        "approval": {
            "provider": "auto",
            "model": "",           # fast/cheap model recommended (e.g. gpt-4o-mini)
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        "title_generation": {
            "enabled": True,
            "provider": "auto",
            "model": "",
            "prefer_fast_model": False,  # opt in to provider fast tier; auto otherwise uses the main model
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
            "language": "",
        },
        "memory_query_rewrite": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 8,
            "extra_body": {},
        },
        "tts_audio_tags": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 30,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Profile describer — auto-generates a 1-2 sentence description
        # of what a profile is good at. Invoked by
        # ``pilotage profile describe <name> --auto`` and the dashboard's
        # auto-generate button. Short, cheap call.
        "profile_describer": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Goal judge — evaluates whether a /goal run's latest response
        # satisfies the goal/contract, and drafts goal contracts. Short
        # structured-JSON calls; a fast cheap model is fine.
        "goal_judge": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Monitor — urgency/importance classifier used by the important-mail
        # monitor catalog automation (cron/scripts/classify_items.py). Scores
        # candidate items 0-10 against the user's criteria so only above-
        # threshold items get delivered. "auto" = main chat model; override to
        # a cheap fast model since per-item scoring is high-volume and a small
        # model is fine.
        "monitor": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 60,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
        # Background review — the post-turn self-improvement fork that decides
        # whether to save a memory / patch a skill. "auto" (default) = run on
        # the main chat model, replaying the full conversation, which is already
        # warm in the prompt cache (cheap cache reads) — unchanged, optimal.
        # Set provider/model to a cheaper model to run the review there for
        # ~3-5x lower cost. A different model can't reuse the main prompt cache anyway, so
        # the fork automatically replays a compact digest instead of the full
        # transcript when routed (minimises the cold-write). Same model = full
        # replay; different model = digest. Quality holds (memory capture
        # identical, skill near-identical in benchmarks).
        "background_review": {
            "provider": "auto",
            "model": "",
            "base_url": "",
            "api_key": "",
            "timeout": 120,
            "extra_body": {},
            "reasoning_effort": "",  # per-task thinking level: none|minimal|low|medium|high|xhigh|max|ultra (empty = provider default)
        },
    },
    
    "display": {
        "compact": False,
        "personality": "",
        "resume_display": "full",
        # Recap tuning for /resume and startup resume. The defaults match the
        # historical hardcoded values; expose them as config so power users can
        # widen or tighten the snapshot to taste.
        "resume_exchanges": 10,            # max user+assistant pairs to show
        "resume_max_user_chars": 300,      # truncate user message text
        "resume_max_assistant_chars": 200, # truncate non-last assistant text
        "resume_max_assistant_lines": 3,   # truncate non-last assistant lines
        # When True (default), assistant entries that are *only* tool calls
        # (no visible text) are skipped in the recap. This prevents the recap
        # from being dominated by `[2 tool calls: terminal, read_file]` lines
        # when an exchange was tool-heavy. Set False to restore the legacy
        # behavior of showing tool-call summaries inline.
        "resume_skip_tool_only": True,
        "busy_input_mode": "interrupt",  # interrupt | queue | steer
        # When busy_input_mode="steer", suppress only the visible
        # "Steered into current run" confirmation bubble by setting this false.
        # The mid-turn steering itself still happens.
        "busy_steer_ack_enabled": True,
        # Classic CLI multiline fallbacks beyond Alt+Enter.
        # Default true matches Claude Code / Codex / OpenCode: Ctrl+J inserts
        # a newline, a trailing backslash followed by Enter continues the draft,
        # and supported terminals are asked to report Shift+Enter distinctly.
        # Set false to restore the legacy c-j submit fallback on unusual POSIX
        # PTYs whose plain Enter arrives as LF instead of CR.
        "cli_multiline_shortcuts": True,
        # Which interface bare `pilotage` (and `pilotage chat`) launches by default:
        #   "cli" — the classic prompt_toolkit REPL (default, preserves prior behavior)
        #   "tui" — the modern Ink TUI (same as passing `--tui`)
        # Explicit flags always win over this setting: `--cli` forces the classic
        # REPL and `--tui` (or PILOTAGE_TUI=1) forces the TUI regardless of config.
        "interface": "cli",
        # When true, `pilotage --tui` auto-resumes the most recent human-
        # facing session on launch instead of forging a fresh one.
        # Mirrors `pilotage -c` muscle memory.  Default off so existing
        # users aren't surprised.  PILOTAGE_TUI_RESUME=<id> always wins.
        "tui_auto_resume_recent": False,
        # When true (default), `pilotage --tui` drops a one-time hint
        # ("subagents working · /agents to watch live") the first time a turn
        # starts delegating, nudging the user toward the live spawn-tree
        # dashboard. Set false to suppress the hint.
        "tui_agents_nudge": True,
        "bell_on_complete": False,
        # Stream the model's reasoning/thinking live before the response.
        # Default ON: on thinking models the reasoning phase can run tens of
        # seconds, and with this off the user stares at a spinner the whole
        # time even though tokens are streaming. Set false for quiet output.
        "show_reasoning": True,
        # When reasoning display is on, the post-response "Reasoning" recap box
        # collapses long thinking to the first 10 lines. Set true to print the
        # complete thinking text uncollapsed (live streaming is always full).
        "reasoning_full": False,
        # Background self-improvement review notifications surfaced in chat.
        #   "off"     — no chat notification (the review still runs and writes)
        #   "on"      — generic "💾 Memory updated" line (default)
        #   "verbose" — include a compact content preview of what changed
        # Per-platform overrides via display.platforms.<platform>.memory_notifications.
        "memory_notifications": "on",
        # Gateway notifications when a terminal(background=true) process
        # finishes:
        #   "concise" — one-line status message; failures append a short
        #               output tail (default)
        #   "all"     — running-output updates + final raw-output message
        #   "result"  — final raw-output message only
        #   "error"   — final raw-output message only on non-zero exit
        #   "off"     — no watcher messages at all
        "background_process_notifications": "concise",
        "streaming": False,
        "timestamps": False,      # Show message timestamps (CLI labels, TUI rows, desktop transcript)
        "timestamp_format": "%H:%M",  # strftime format for timestamps (e.g. "%b-%d %H:%M")
        "final_response_markdown": "strip",  # render | strip | raw
        # Preserve recent classic CLI output across Ctrl+L, /redraw, and
        # terminal resize full-screen clears. Disable if a terminal emulator
        # behaves badly with replayed scrollback.
        "persistent_output": True,
        "persistent_output_max_lines": 200,
        # Clear terminal scrollback as well as the visible viewport when the
        # classic CLI performs a full redraw/resize recovery. Disabled by
        # default because some users prefer preserving terminal history;
        # enable when a terminal/tmux stack stamps stale prompt chrome into
        # scrollback during fullscreen/restore window transitions.
        "cli_rebuild_scrollback_on_redraw": False,
        # Print a one-line summary of resolved modal prompts (approval /
        # clarify) into scrollback so the question and decision survive the
        # panel repaint. Set false to keep scrollback untouched.
        "persist_prompts": True,
        "inline_diffs": True,     # Show inline diff previews for write actions (write_file, patch, skill_manage)
        # File-mutation verifier footer.  When true (default), the agent
        # appends a one-line advisory to its final response whenever a
        # write_file / patch call failed during the turn and was never
        # superseded by a successful write to the same path.  This catches
        # the "batch of parallel patches, half fail, model claims success"
        # class of over-claim that otherwise forces users to run
        # `git status` to verify edits landed.  Set false to suppress.
        "file_mutation_verifier": True,
        # Nous credits status-bar notices (usage bands, grant-spent, depleted /
        # restored).  When false, no credits notices are emitted — balance data
        # is still captured and /usage keeps working.  Off switch for sub +
        # top-up users who find the gauge noisy.
        "credits_notices": True,
        # Turn-completion explainer.  When true (default), the agent appends a
        # one-line explanation to its final response whenever a turn ends
        # abnormally with no usable reply — empty content after retries, a
        # partial/truncated stream, a still-pending tool result, or an
        # iteration/budget limit.  Replaces the bare "(empty)" sentinel so the
        # failure isn't silent from the UI's perspective.  Set false to suppress.
        "turn_completion_explainer": True,
        "show_cost": False,       # Show $ cost in the status bar (off by default)
        # Show a color-coded battery read-out as the first status-bar element in
        # the CLI/TUI (off by default). No-op on machines without a battery.
        "battery": False,
        # Focus view (/focus): display-only reduced-output mode. When true the
        # CLI/TUI pins tool_progress to "off" (reusing the existing suppression
        # path), reports a per-turn hidden-line count with a recovery hint, and
        # pins a "focus" segment in the status bar. focus_saved_tool_progress
        # holds the mode /focus off restores. Never affects what is sent to the
        # model — see pilotage_cli/focus_view.py.
        "focus_view": False,
        "focus_saved_tool_progress": "all",
        "skin": "default",
        # UI language for static user-facing messages (approval prompts, a
        # handful of gateway slash-command replies).  Does NOT affect agent
        # responses, log lines, tool outputs, or slash-command descriptions.
        # Supported: en, zh, ja, de, es, fr, tr, uk.  Unknown values fall back to en.
        "language": "en",
        # TUI busy indicator style: kaomoji (default), emoji, unicode (braille
        # spinner), or ascii.  Live-swappable via `/indicator <style>`.
        "tui_status_indicator": "kaomoji",
        # Seconds between prompt_toolkit redraws in the classic CLI when idle.
        # Default 1.0 keeps the wall-clock status-bar read-outs (idle-since-
        # last-turn) ticking and keeps the bottom chrome alive during idle —
        # without it prompt_toolkit stops repainting the status bar after a
        # turn and it can go stale/disappear.
        # Set 0 to disable the background refresh if it fights terminal
        # auto-scroll in non-fullscreen mode on some emulators.
        "cli_refresh_interval": 1.0,
        "user_message_preview": {  # CLI: how many submitted user-message lines to echo back in scrollback
            "first_lines": 2,
            "last_lines": 2,
        },
        "interim_assistant_messages": True,  # Gateway: send natural mid-turn assistant status messages. Desktop: keep mid-turn narration between tool calls instead of collapsing to the final message.
        # Codex Responses models narrate progress in a dedicated commentary
        # channel. When true (default), completed commentary messages are
        # delivered as visible mid-turn updates via the interim message path.
        # When false, commentary falls back to the reasoning channel and is
        # only visible when show_reasoning is enabled.
        "show_commentary": True,
        "tool_progress_command": False,  # Enable /verbose command in messaging gateway
        # NOTE: display.tool_progress_overrides is deprecated and no longer
        # seeded here — use display.platforms. A user-set value is still
        # honored at runtime (gateway display_config back-compat read) and
        # folded into display.platforms by the v15→16 migration.
        "tool_preview_length": 0,  # Max chars for tool call previews (0 = no limit, show full paths/commands)
        # Human-phrased tool status labels for built-in tools: "Searching the
        # web for ...", "Reading <file>", "Browsing <url>" instead of the raw
        # tool name. Applies to CLI spinner + gateway/desktop tool-progress.
        # Custom/plugin/MCP tools always fall back to the raw preview.
        "friendly_tool_labels": True,
        # CLI-only post-turn accounting line printed after each interactive turn:
        # "⋯ 12.4s · edited 2 files +18 -3 · read 4 files · ran 3 commands".
        # Observed from the tool-progress feed the CLI already receives; never
        # printed in quiet/non-interactive paths or in gateway/messaging
        # surfaces (those have their own runtime footer).
        "turn_summary": True,
        # CLI-only: append cumulative turn output tokens to the live spinner
        # timer ("⚡ Reading file  ( 2.3s · ↓ 1.2k tok)"). Updates as each API
        # call in the turn reports usage.
        "spinner_token_flow": True,
        # How gateway tool-progress is grouped on platforms that support message
        # editing: "accumulate" (default) edits one bubble in place; "separate"
        # sends one message per tool (the pre-v0.9 behavior, noisier). Only
        # applies where tool_progress is already enabled. Per-platform override
        # via display.platforms.<platform>.tool_progress_grouping.
        "tool_progress_grouping": "accumulate",
        # Optional custom phrases for generic long-running status messages.
        # Built-in defaults live in gateway/assets/status_phrases.yaml. Users
        # can set `path`/`paths` to PILOTAGE_HOME-relative YAML files/directories
        # (or rely on conventional status_phrases.yaml / status_phrases/*.yaml).
        # Keys: status, generic. Use
        # mode: "append" (default) to add phrases, or "replace" to fully
        # replace configured surfaces. Per-platform overrides live under
        # display.platforms.<platform>.status_phrases.
        "status_phrases": {},
        # How a reasoning/thinking summary renders when show_reasoning is on.
        # "code" (default) = 💭 fenced code block; "blockquote" = "> " lines;
        # "subtext" = "-# " lines. Override per-platform via
        # display.platforms.<platform>.reasoning_style.
        "reasoning_style": "code",
        # Auto-delete system-notice replies (e.g. "✨ New session started!",
        # "♻ Restarting gateway…", "⚡ Stopped…") after N seconds on platforms
        # that support message deletion (currently Telegram; other platforms
        # ignore and leave the message in place).  Only affects slash-command
        # replies wrapped with gateway.platforms.base.EphemeralReply — agent
        # responses and content messages are never touched.  Default 0
        # (disabled) preserves prior behavior.
        "ephemeral_system_ttl": 0,
        # Per-platform display/streaming overrides. Each key is a gateway
        # platform ("telegram", "whatsapp", …) mapping to a dict of
        # display settings that override the global value for that platform
        # only. A setting left unset here falls through to the global default.
        #
        # Shipped defaults encode the streaming experience that works best
        # per platform: Telegram has native animated draft streaming
        # (sendMessageDraft), which is smooth, so streaming is on by default
        # there. These are gap-fillers: a user who explicitly sets, e.g.,
        # display.platforms.telegram.streaming: false keeps their value
        # (config deep-merge has user values win over defaults). The global
        # streaming.enabled master switch still gates everything — these
        # per-platform flags only take effect once streaming is enabled.
        "platforms": {
            "telegram": {"streaming": True},
        },
        # Gateway runtime-metadata footer appended to the FINAL message of a turn
        # (disabled by default to keep replies minimal). When enabled, renders
        # e.g. `model · 68% · ~/projects/pilotage`. Per-platform overrides go under
        # display.platforms.<platform>.runtime_footer.
        "runtime_footer": {
            "enabled": False,
            "fields": ["model", "context_pct", "cwd"],  # Order shown; drop any to hide
        },
        "copy_shortcut": "auto",  # "auto" (platform default) | "ctrl_c" | "ctrl_shift_c" | "disabled"
        # Petdex animated mascot (https://github.com/crafter-station/petdex).
        # A purely cosmetic sprite that reacts to agent activity across the
        # CLI, TUI, and desktop app. Manage with `pilotage pets`. Disabled until
        # a pet is installed + selected (no effect on prompt caching — this is
        # a display concern only).
        "pet": {
            "enabled": False,
            # Active pet slug; resolved against installed pets in
            # get_pilotage_home()/pets/. Empty → first installed pet.
            "slug": "",
            # Terminal render protocol for CLI/TUI:
            #   auto  — detect kitty/iTerm2/sixel, else unicode half-blocks
            #   kitty | iterm | sixel | unicode | off
            "render_mode": "auto",
            # Master size scalar (relative to native 192×208 frames). One knob
            # shrinks every surface: the desktop canvas scales its pixels by it
            # and the CLI/TUI derive their terminal column width from it. The
            # half-block fallback clamps to a legibility floor (it can't shrink
            # as far as true-pixel kitty/GUI without turning to mush).
            "scale": 0.33,
            # Hard override for terminal column width. 0 = auto (derive from
            # scale); set a positive int only to pin the half-block/kitty width
            # independently of scale.
            "unicode_cols": 0,
        },
    },

    # Privacy settings
    "privacy": {
        "redact_pii": False,  # When True, hash user IDs and strip phone numbers from LLM context
    },

    # Text-to-speech configuration. OpenAI is the only built-in backend;
    # `max_text_length` overrides the 4096-character per-request cap.
    "tts": {
        "provider": "openai",
        "openai": {
            "model": "gpt-4o-mini-tts",
            "voice": "alloy",
            # Voices: alloy, ash, ballad, cedar, coral, echo, fable, marin,
            # nova, onyx, sage, shimmer, verse (gpt-4o-mini-tts; the tts-1
            # era stopped at alloy/echo/fable/onyx/nova/shimmer)
        },
    },

    "stt": {
        "enabled": True,
        # When true, gateway voice messages are transcribed for the agent and
        # the raw transcript is also echoed back to the user as a mic message.
        # Set false to keep STT for the agent while suppressing that user-facing echo.
        "echo_transcripts": True,
        "provider": "openai",  # "openai" (OpenAI transcription API) or a plugin-registered name
        # Global language hint. Defaults to "en" -- Whisper auto-detection
        # frequently misidentifies short/accented clips, which reads as
        # "STT transcribed the wrong language". Set to "" to restore
        # auto-detect, or to your language code ("es", "zh", "uk", ...).
        "language": "en",
        # Pre-upload silence trim. Cloud endpoints otherwise receive raw
        # audio -- silence inflates upload time, per-audio-minute billing,
        # and hallucination risk. Collapses pauses with ffmpeg client-side;
        # any failure uploads the original.
        "cloud_trim_silence": True,
        "cloud_trim_threshold_db": -40,  # audio quieter than this counts as silence
        "cloud_trim_keep_ms": 300,  # how much of each pause survives (keeps natural pacing)
        "openai": {
            "model": "whisper-1",  # whisper-1, gpt-4o-mini-transcribe, gpt-4o-transcribe, gpt-transcribe
            "language": "",  # auto-detect by default; set to "en", "es", "fr", etc. to force
        },
    },

    "voice": {
        "auto_tts": False,   # Speak every agent reply, not just explicit TTS calls
    },

    
    "human_delay": {
        "mode": "off",
        "min_ms": 800,
        "max_ms": 2500,
    },
    
    # Context engine -- controls how the context window is managed when
    # approaching the model's token limit.
    # "compressor" = built-in lossy summarization (default).
    # Set to a plugin name to activate an alternative engine (e.g. "lcm"
    # for Lossless Context Management).  The engine must be installed as
    # a plugin in plugins/context_engine/<name>/ or ~/.pilotage/plugins/.
    "context": {
        "engine": "compressor",
        # Return freed glibc allocator pages after long-running agent/TUI
        # cleanup boundaries. Unsupported platforms are safe no-ops.
        "memory_trim": {
            "enabled": True,
            "cooldown_seconds": 60.0,
            # Successful trim calls are INFO logged every Nth periodic call;
            # force paths always log so process-close behavior is visible.
            "log_every_n": 1,
            # Suppress INFO logs only when a readable RSS change is smaller.
            # 0 reports every successful configured trim.
            "info_log_min_delta_mb": 0.0,
        },
    },

    # Persistent memory -- bounded curated memory injected into system prompt
    "memory": {
        "memory_enabled": True,
        "user_profile_enabled": True,
        # Approval gate for memory writes (add/replace/remove), applied to BOTH
        # foreground agent turns and the background self-improvement review fork
        # (the source of unprompted "wrong assumption" saves users reported).
        #   false (default) — write freely; the gate is off (pre-gate behaviour)
        #   true            — require approval: foreground writes prompt inline
        #                     (entries are small enough to review in a chat
        #                     bubble); background-review writes are staged
        #                     instead of committed (a daemon thread cannot block
        #                     on a prompt). Review staged entries with
        #                     /memory pending, /memory approve <id>,
        #                     /memory reject <id>.
        # To disable memory entirely, use memory_enabled: false instead.
        "write_approval": False,
        "memory_char_limit": 2200,   # ~800 tokens at 2.75 chars/token
        "user_char_limit": 1375,     # ~500 tokens at 2.75 chars/token
        # External memory provider plugin (empty = built-in only).
        # Set to a provider name to activate: "openviking", "mem0",
        # "hindsight", "holographic", "retaindb", "byterover".
        # Only ONE external provider is allowed at a time.
        "provider": "",
    },

    # Subagent delegation — override the provider:model used by delegate_task
    # so child agents can run on a different (cheaper/faster) provider and model.
    # Uses the same runtime provider resolution as CLI/gateway startup, so all
    # configured providers (OpenAI, custom endpoints) are supported.
    "delegation": {
        "model": "",       # e.g. "gpt-5.4" (empty = inherit parent model)
        "provider": "",    # e.g. "openai-api" (empty = inherit parent provider + credentials)
        "base_url": "",    # direct OpenAI-compatible endpoint for subagents
        "api_key": "",     # API key for delegation.base_url (falls back to OPENAI_API_KEY)
        "api_mode": "",    # wire protocol for delegation.base_url: "chat_completions"
        "max_iterations": 250,  # per-subagent iteration cap (each subagent gets its own budget,
                               # independent of the parent's max_iterations)
        # Subagent summaries return to the parent's context verbatim. A batch
        # fan-out (N children) returns N summaries at once, which can exceed
        # the parent's context window and trigger a compression/429 death
        # spiral. delegate_task sizes each summary against the parent's
        # remaining context headroom (split across the batch); when it must
        # trim, the full text is spilled to ~/.pilotage/cache/delegation/
        # (mounted into remote backends) and the in-context summary becomes a
        # head+tail window plus a footer with the exact read_file offset to
        # page the omitted middle — the same convention web_extract uses for
        # large pages. Nothing is lost. max_summary_chars is a hard per-summary
        # character ceiling layered on top of that dynamic budget
        # (belt-and-suspenders for models that ignore the "be concise"
        # instruction). 0 disables the hard ceiling; the dynamic headroom
        # budget still applies.
        "max_summary_chars": 24000,

        "child_timeout_seconds": 0,  # optional wall-clock cap per child agent. 0 (default)
                                     # = no timeout: children fail only from real errors
                                     # (API, tools, iteration budget), never a delegation
                                     # stopwatch. Set a positive number of seconds
                                     # (floor 30s) to enforce a hard cap.
        "reasoning_effort": "",  # subagent effort: "ultra", "max", "xhigh", "high",
                                 # "medium", "low", "minimal", "none" (empty = inherit)
        "max_concurrent_children": 10,  # unified concurrency cap: max parallel children per batch
                                       # AND max concurrent background (background=true)
                                       # delegation units. New async dispatches beyond the cap
                                       # fall back to synchronous execution. Floor of 1, no ceiling.
                                       # (Replaces the deprecated max_async_children.)
        # Orchestrator role controls (see tools/delegate_tool.py:_get_max_spawn_depth
        # and _get_orchestrator_enabled).  Floored at 1, no upper ceiling —
        # raise deliberately, each level multiplies API cost.
        "max_spawn_depth": 1,        # depth (1 = flat [default], 2 = orchestrator→leaf, 3+ = deeper)
        "orchestrator_enabled": True,  # kill switch for role="orchestrator"
        # When a subagent hits a dangerous-command approval prompt, the parent's
        # prompt_toolkit TUI owns stdin — a thread-local input() call from the
        # subagent worker would deadlock the parent UI. To avoid the deadlock,
        # subagent threads ALWAYS resolve approvals non-interactively:
        #   false (default) → auto-deny with a logger.warning audit line (safe)
        #   true             → auto-approve "once" with a logger.warning audit line
        # Flip to true only if you trust delegated work to run dangerous cmds
        # without human review (cron pipelines, batch automation, etc.).
        "subagent_auto_approve": False,
    },

    # Ephemeral prefill messages file — JSON list of {role, content} dicts
    # injected at the start of every API call for few-shot priming.
    # Never saved to sessions, logs, or trajectories.
    "prefill_messages_file": "",

    # Goals — persistent cross-turn goals (Ralph-style loop).
    # After every turn, a lightweight judge call asks the auxiliary model
    # whether the active /goal is satisfied by the assistant's last
    # response. If not, Pilotage feeds a continuation prompt back into the
    # same session and keeps working until the goal is done, the turn
    # budget is exhausted, or the user pauses/clears it. Judge failures
    # fail OPEN (continue) so a flaky judge never wedges progress — the
    # turn budget is the real backstop.
    "goals": {
        # Max continuation turns before Pilotage auto-pauses the goal and
        # asks the user to /goal resume. Protects against judge false
        # negatives (goal actually done but judge says continue) and
        # unbounded model spend on fuzzy / unachievable goals.
        "max_turns": 20,
    },


    # Skills — external skill directories for sharing skills across tools/agents.
    # Each path is expanded (~, ${VAR}) and resolved.  Read-only — skill creation
    # always goes to ~/.pilotage/skills/.
    "skills": {
        "external_dirs": [],   # e.g. ["~/.agents/skills", "/shared/team-skills"]
        # Substitute ${PILOTAGE_SKILL_DIR} and ${PILOTAGE_SESSION_ID} in SKILL.md
        # content with the absolute skill directory and the active session id
        # before the agent sees it.  Lets skill authors reference bundled
        # scripts without the agent having to join paths.
        "template_vars": True,
        # Pre-execute inline shell snippets written as !`cmd` in SKILL.md
        # body.  Their stdout is inlined into the skill message before the
        # agent reads it, so skills can inject dynamic context (dates, git
        # state, detected tool versions, …).  Off by default because any
        # content from the skill author runs on the host without approval;
        # only enable for skill sources you trust.
        "inline_shell": False,
        # Timeout (seconds) for each !`cmd` snippet when inline_shell is on.
        "inline_shell_timeout": 10,
        # Run the keyword/pattern security scanner on skills the agent
        # writes via skill_manage (create/edit/patch).  Off by default
        # because the agent can already execute the same code paths via
        # terminal() with no gate, so the scan adds friction (blocks
        # skills that mention risky keywords in prose) without meaningful
        # security.  Turn on if you want the belt-and-suspenders — a
        # dangerous verdict will then surface as a tool error to the
        # agent, which can retry with the flagged content removed.
        # External hub installs (trusted/community sources) are always
        # scanned regardless of this setting.
        "guard_agent_created": False,
        # Approval gate for skill_manage (create/edit/patch/write_file/delete/
        # remove_file), applied to BOTH foreground agent turns and the
        # background self-improvement review fork.
        #   false (default) — write freely; the gate is off (pre-gate behaviour)
        #   true            — require approval: stage the write for review
        #                     instead of committing (a SKILL.md is too large to
        #                     review inline, so skills always stage rather than
        #                     prompt). List with /skills pending, inspect with
        #                     /skills diff <id> (full diff — CLI/dashboard/file,
        #                     never crammed into a chat bubble), apply with
        #                     /skills approve <id> or drop with /skills reject <id>.
        "write_approval": False,
    },

    # Honcho AI-native memory -- reads ~/.honcho/config.json as single source of truth.
    # This section is only needed for pilotage-specific overrides; everything else
    # (apiKey, workspace, peerName, sessions, enabled) comes from the global config.
    "honcho": {},

    # IANA timezone (e.g. "Asia/Kolkata", "America/New_York").
    # Empty string means use server-local time.
    "timezone": "",

    # WhatsApp platform settings (gateway mode)
    "whatsapp": {
        # Reply prefix prepended to every outgoing WhatsApp message.
        # Default (None) uses the built-in "⚕ *Pilotage Agent*" header.
        # Set to "" (empty string) to disable the header entirely.
        # Supports \n for newlines, e.g. "🤖 *My Bot*\n──────\n"
    },

    # Telegram platform settings (gateway mode)
    "telegram": {
        "reactions": False,            # Add 👀/✅/❌ reactions to messages during processing
        "channel_prompts": {},         # Per-chat/topic ephemeral system prompts (topics inherit from parent group)
        "allowed_chats": "",           # If set, bot ONLY responds in these group/supergroup chat IDs (whitelist)
        "extra": {
            "rich_messages": False,     # Bot API 10.1 rich messages (tables/task lists/details/math) render natively; set True to opt in. Default stays legacy MarkdownV2 because rich messages can be hard to copy as plain text in Telegram clients.
            "rich_drafts": False,       # Experimental Bot API 10.1 rich draft previews during Telegram DM streaming. Default off because Telegram Desktop/macOS can visually overlay rich draft frames until the chat redraws.
        },
    },

    # Approval mode for dangerous commands:
    #   manual — always prompt the user
    #   smart  — use auxiliary LLM to auto-approve low-risk commands (default)
    #   off    — skip all approval prompts (equivalent to --yolo)
    #
    # cron_mode — what to do when a cron job hits a dangerous command:
    #   deny    — block the command and let the agent find another way (default, safe)
    #   approve — auto-approve all dangerous commands in cron jobs
    #
    # timeout — seconds to wait for the user's approve/deny before failing
    # closed (deny). Shared by the CLI prompt and gateway/messaging waits.
    # Messaging approvals arrive as a push notification the user may not see
    # immediately — 60s proved too tight on Telegram/Discord (the prompt
    # expired before the user reached their phone), so the default is 300.
    "approvals": {
        "mode": "smart",
        "timeout": 300,
        "cron_mode": "deny",
        # Operator-customizable policy text for smart approvals. When
        # non-empty, this is appended to the smart-approval guardian's
        # SYSTEM prompt (trusted channel) as additional rules — e.g.
        # "Always ESCALATE commands touching /etc" or "APPROVE docker
        # compose restarts under ~/deploys". Inspired by ChatGPT Work's
        # customizable auto-review guardian policy.
        "smart_policy": "",
        # Consecutive-denial circuit breaker for smart approvals: after this
        # many guardian DENY verdicts in a row within one session, the deny
        # message returned to the model escalates to a hard-stop instruction
        # (report to the user / ask for manual run or /approve) instead of a
        # plain "Do NOT retry". Any approval resets the count. 0 disables.
        # Inspired by ChatGPT Work's auto-review circuit breaker.
        "denial_breaker_threshold": 3,
        # User-defined deny rules: fnmatch globs matched against terminal
        # commands. A match blocks the command unconditionally — BEFORE the
        # --yolo / /yolo / mode=off bypass — making this the user-editable
        # counterpart to the code-shipped hardline blocklist. Patterns are
        # case-insensitive and must be quoted in YAML when they start with
        # * or contain {}/!/: sequences. Example:
        #   deny:
        #     - "git push --force*"
        #     - "*curl*|*sh*"
        "deny": [],
        # When true, destructive session slash commands (/clear, /new, /reset,
        # /undo) ask the user to confirm before discarding conversation state.
        # Three-option prompt (Approve Once / Always Approve / Cancel) routed
        # through tools.slash_confirm — native yes/no buttons on Telegram;
        # text fallback elsewhere.  Users click "Always
        # Approve" to silence the prompt permanently; that flips this key to
        # false.  TUI also honors this setting for its /clear, /new, and /reset
        # modal; PILOTAGE_TUI_NO_CONFIRM=1 force-skips that modal regardless of
        # the configured value.
        "destructive_slash_confirm": True,
    },

    # Permanently allowed dangerous command patterns (added via "always" approval)
    "command_allowlist": [],
    # User-defined quick commands that bypass the agent loop (type: exec only)
    "quick_commands": {},

    # Per-platform system-prompt hint overrides. Lets an admin append to or
    # replace Pilotage' built-in platform hint for a single messaging platform
    # (WhatsApp, Telegram, ...) without affecting other platforms.
    # Useful for enterprise/managed profiles that ship platform-aware skills.
    # Each key is a platform name; the value is either:
    #   { "append": "extra text" }   — keep the default hint, append text
    #   { "replace": "full text" }   — substitute the default hint entirely
    #   "extra text"                 — shorthand for { "append": ... }
    # `replace` wins over `append` if both are given. Example:
    #   platform_hints:
    #     whatsapp:
    #       append: >
    #         When tabular output would be useful, invoke the
    #         table_formatting skill instead of emitting a Markdown table.
    "platform_hints": {},

    # Shell-script hooks — declarative bridge that invokes shell scripts
    # on plugin-hook events (pre_tool_call, post_tool_call, pre_llm_call,
    # subagent_stop, etc.).  Each entry maps an event name to a list of
    # {matcher, command, timeout} dicts.  First registration of a new
    # command prompts the user for consent; subsequent runs reuse the
    # stored approval from ~/.pilotage/shell-hooks-allowlist.json.
    # See `website/docs/user-guide/features/hooks.md` for schema + examples.
    "hooks": {},

    # Auto-accept shell-hook registrations without a TTY prompt.  Also
    # toggleable per-invocation via --accept-hooks or PILOTAGE_ACCEPT_HOOKS=1.
    # Gateway / cron / non-interactive runs need this (or one of the other
    # channels) to pick up newly-added hooks.
    "hooks_auto_accept": False,
    # Custom personalities — add your own entries here
    # Supports string format: {"name": "system prompt"}
    # Or dict format: {"name": {"description": "...", "system_prompt": "...", "tone": "...", "style": "..."}}
    "personalities": {},

    # Pre-exec security scanning via tirith
    "security": {
        "allow_private_urls": False,  # Allow requests to private/internal IPs (for OpenWrt, proxies, VPNs)
        "redact_secrets": True,
        # Human approval presentation transport. "builtin" preserves the
        # current CLI/TUI/gateway/ACP surfaces. A plugin transport is used only
        # when named explicitly here. Transport timeout/error/invalid response
        # denies unless transport_fallback is explicitly set to "builtin".
        # This is presentation only: plugins cannot detect, suppress, or
        # auto-approve commands outside a correlated human response.
        "approval": {
            "transport": "builtin",
            "transport_fallback": "deny",
        },
        # Writes to agent-instruction files (AGENTS.md/CLAUDE.md/SOUL.md/
        # .cursorrules, project-local .pilotage config) always require human
        # approval — even under auto-approve/yolo. Extra patterns are
        # fnmatch globs matched against the basename (e.g. "*.mdc").
        "protected_instruction_files": True,
        "protected_instruction_extra_patterns": [],
        "tirith_enabled": True,
        "tirith_path": "tirith",
        "tirith_timeout": 5,
        "tirith_fail_open": True,
        "website_blocklist": {
            "enabled": False,
            "domains": [],
            "shared_files": [],
        },
        # Acknowledged supply-chain security advisories. Each entry is the
        # ID of an advisory the user has read and acted on (uninstalled the
        # compromised package, rotated credentials). Acked advisories no
        # longer trigger the startup banner. Add via `pilotage doctor --ack
        # <id>`; remove by editing the list directly. See
        # ``pilotage_cli/security_advisories.py`` for the catalog.
        "acked_advisories": [],
        # Allow Pilotage to lazy-install opt-in backend packages from PyPI
        # the first time the user enables a backend that needs them.
        # Set to false to require explicit
        # ``pip install`` for everything beyond the base set — appropriate
        # for restricted networks, audited environments, or air-gapped
        # systems where any runtime install is unacceptable.
        "allow_lazy_installs": True,
    },

    "cron": {
        # Allow cron-spawned agents to use the cronjob toolset (create/edit/
        # remove scheduled jobs from within a cron run — the "cron-librarian"
        # pattern). Off by default: the cronjob toolset is policy-denied in
        # cron context to prevent unattended scheduling loops. Jobs created
        # this way are user-owned in the same flat jobs table as every other
        # job. Interactive toolsets (messaging/clarify) stay denied in cron
        # context regardless of this setting.
        "allow_agent_scheduling": False,
        # Pre-dispatch configuration validation (T1-26): before constructing
        # any agent machinery for a job, verify the provider API key resolves
        # (unless a fallback chain is configured), attached skills are ready
        # (required env/commands present), and delivery platforms are
        # configured. A failing job is recorded as last_status=blocked_config
        # with ONE alert (no re-alert every tick) and NO LLM call is made.
        # Set to false to restore the old behavior (fail during the run).
        "preflight": True,
        # Fail closed when an unpinned job's current global model/provider
        # differs from its creation-time snapshot. This prevents unattended
        # jobs from silently inheriting a paid default. Set to false only when
        # jobs should deliberately track changing global inference defaults.
        "model_drift_guard": True,
        # Default inference model for cron jobs (Axis A — WHAT model an
        # agent job runs on). Resolution at fire time: per-job user pin >
        # cron.model > global model.default. When set, unpinned jobs follow
        # this deliberately, so the model-drift fail-closed guard does
        # not engage for the model axis — cron spend no longer shadows chat
        # `/model` switches. Empty string = fall through to model.default.
        "model": "",
        # Inference provider paired with cron.model (NOT the scheduler
        # provider below). Empty string = resolve from global config.
        "model_provider": "",
        # Active cron SCHEDULER provider (Axis B — the trigger that decides
        # WHEN a due job fires). Empty string = the built-in in-process 60s
        # ticker (default), and no provider ships bundled. Name a provider
        # installed under $PILOTAGE_HOME/plugins/<name>/ to relocate the trigger.
        # An unknown or unavailable provider falls back to the built-in, so cron
        # never loses its trigger.
        "provider": "",
        # Wrap delivered cron responses with a header (task name) and footer
        # ("The agent cannot see this message").  Set to false for clean output.
        "wrap_response": True,
        # Make cron deliveries CONTINUABLE: a user can reply to a cron brief
        # and the agent has it in context (no "what is Task #2?" amnesia).
        # Default False preserves the historical isolation guarantee (cron
        # deliveries live only in the cron job's own session). Per-job
        # `attach_to_session` overrides this for a single job.
        #
        # Behaviour is THREAD-PREFERRED, scoped to the job's origin chat:
        #   - Thread-capable platforms (Telegram forum/DM topics): a
        #     dedicated thread is opened for the job
        #     via the adapter's create_handoff_thread, the brief is delivered
        #     into it, and that thread's session is seeded so the user's reply
        #     in-thread continues with full context. Each continuable job gets
        #     its own scrollback, isolated from the parent channel.
        #   - DM-only platforms (WhatsApp): no threads exist, so
        #     the brief is mirrored into the origin DM session instead — the
        #     DM itself is the continuation surface.
        # Both paths ride the shipped gateway.mirror.mirror_to_session and are
        # alternation- and cache-safe (appended at a turn boundary, never
        # mid-loop, never mutating the cached system prompt). Only the origin
        # chat is ever touched — fan-out / broadcast targets are never mirrored.
        "mirror_delivery": False,
        # Maximum number of due jobs to run in parallel per tick.
        # null/0 = unbounded (limited only by thread count).
        # 1 = serial (pre-v0.9 behaviour).
        # Also overridable via PILOTAGE_CRON_MAX_PARALLEL env var.
        "max_parallel_jobs": None,
        # Per-job output-file retention: save_job_output keeps the N most
        # recent .md files and prunes older ones. 0 or negative disables
        # pruning (for operators who manage cleanup externally). Default 50.
        "output_retention": 50,
        # Timeout (seconds) for a no-agent cron script. Also overridable via
        # PILOTAGE_CRON_SCRIPT_TIMEOUT. Keep this in sync with
        # cron.scheduler._DEFAULT_SCRIPT_TIMEOUT so config set recognizes the
        # same setting the scheduler reads.
        "script_timeout_seconds": 3600,
        # Timeout (seconds) for SessionDB() init inside cron jobs.
        # SessionDB opens/migrates state.db synchronously and has no timeout
        # of its own against a wedged sqlite3.connect. An unbounded hang here
        # wedges the job's dispatch guard forever. Also overridable via
        # PILOTAGE_CRON_SESSION_DB_TIMEOUT env var. 0 = unlimited (skip the bound).
        "session_db_timeout_seconds": 10,
    },


    # execute_code settings — controls the tool used for programmatic tool calls.
    "code_execution": {
        # Execution mode:
        #   project (default) — scripts run in the session's working directory
        #     with the active virtualenv/conda env's python, so project deps
        #     (pandas, torch, project packages) and relative paths resolve.
        #   strict            — scripts run in an isolated temp directory with
        #     pilotage-agent's own python (sys.executable). Maximum isolation
        #     and reproducibility; project deps and relative paths won't work.
        # Env scrubbing (strips *_API_KEY, *_TOKEN, *_SECRET, ...) and the
        # tool whitelist apply identically in both modes.
        "mode": "project",
    },

    # Tool Search (progressive disclosure for large tool surfaces).
    # When the model is connected to many MCP servers or non-core plugin
    # tools, their JSON schemas can consume a substantial fraction of the
    # context window on every turn. When enabled, those tools are replaced
    # in the model-facing tools array with three bridge tools —
    # tool_search / tool_describe / tool_call — and surfaced on demand.
    #
    # Core Pilotage tools (terminal, read_file, write_file, patch,
    # search_files, todo, memory, browser_*, etc.) are NEVER deferred.
    # See tools/tool_search.py for full design notes and the
    # openclaw-tool-search-report PDF in this PR for the rationale.
    "tools": {
        "tool_search": {
            # Tiered disclosure: any deferrable (MCP/plugin) tool activates
            # the bridge; the listing then scales with catalog size.
            #   Tier 0 — no MCP/plugin tools: everything stays eager.
            #   Tier 1 — catalog listing fits the budget: bridge + skills-style
            #     name+description manifest (degrades to names-only).
            #   Tier 2 — per-tool listing over budget even names-only (e.g.
            #     Cloudflare's ~3,300-tool flat API surface): bare bridge +
            #     a one-line-per-server summary (name + tool count) so the
            #     model knows which domains are reachable; individual tools
            #     discoverable through tool_search only.
            # "auto"/"on" — activate when at least one deferrable tool exists.
            # "off" — disable entirely. Tools-array assembly is a pass-through.
            "enabled": "auto",
            # Listing budget as a percentage of the active model's context
            # length. Effective budget = min(this % of context,
            # listing_max_tokens). Range 0..100.
            "threshold_pct": 5,
            # When the model calls tool_search without a ``limit`` argument,
            # how many hits to return. Range 1..max_search_limit.
            "search_default_limit": 5,
            # Hard upper bound the model can request via ``limit``. Range 1..50.
            "max_search_limit": 20,
            # Skills-style catalog listing embedded in the tool_search bridge
            # description: every deferred tool's name + first sentence of its
            # description (≤60 chars), grouped by MCP server / toolset. Keeps
            # capabilities discoverable while schemas stay deferred.
            # "auto" (default) — include when the listing fits the budget
            #   (falls back to names-only, then to the bare tier-2 bridge).
            # "on"  — same rendering, but explicit intent to always list.
            # "off" — always the bare bridge (tier 2 for every catalog).
            "listing": "auto",
            # Absolute cap on the embedded listing in tokens (chars/4
            # estimate), regardless of context size. Range 200..60000.
            "listing_max_tokens": 4000,
        },
    },

    # Logging — controls file logging to ~/.pilotage/logs/.
    # agent.log captures INFO+ (all agent activity); errors.log captures WARNING+.
    "logging": {
        "level": "INFO",       # Minimum level for agent.log: DEBUG, INFO, WARNING
        "max_size_mb": 5,      # Max size per log file before rotation
        "backup_count": 3,     # Number of rotated backup files to keep
    },

    # Remotely-hosted model catalog manifest.  When enabled, the CLI fetches
    # curated model lists from this URL,
    # falling back to the in-repo snapshot on network failure.  Lets us
    # update model picker lists without shipping a pilotage-agent release.
    # The default URL is served by the docs site GitHub Pages deploy.
    "model_catalog": {
        # Remote catalog disabled: no hosted Pilotage manifest exists yet.
        # Point url at a self-hosted JSON (same schema) and set enabled:
        # true to restore live fetching.
        "enabled": False,
        "url": "",
        # Disk cache TTL in hours.  Beyond this, the CLI refetches on the
        # next /model or `pilotage model` invocation; network failures
        # silently fall back to the stale cache.
        "ttl_hours": 1,
        # Optional per-provider override URLs for third parties that want
        # to self-host their own curation list using the same schema.
        # Example:
        #   providers:
        #     openai:
        #       url: https://example.com/my-curation.json
        "providers": {},
    },

    # Per-model metadata overrides — manually declare context_window,
    # max_output_tokens, capabilities, or model family for any
    # provider+model. Recognized fields: context_window,
    # max_output_tokens, supports_tools, supports_vision,
    # supports_reasoning, model_family.
    #
    # Semantics:
    #   1. Explicit (model_overrides.<provider>.<model_id>): wins over
    #      models.dev and hardcoded defaults for the fields
    #      it sets. NOTE: an explicit model.context_length (global) and a
    #      custom_providers per-model context_length are user settings at
    #      other layers and are consulted in the resolution chain order
    #      documented in agent/model_metadata.py.
    #   2. Fill-gap defaults (model_overrides.<provider>._default and
    #      model_overrides._default): apply ONLY to models the catalog
    #      does not know. They never displace catalog data for known
    #      models, so a _default cannot accidentally clamp every model
    #      of a provider.
    #
    # An unknown model id (not in models.dev) starts from safe defaults
    # (200K context, tools on, vision/reasoning off) and the override
    # patches the fields it sets — overriding a model the catalog
    # doesn't know yet is the supported self-unblock path,
    #).
    #
    # Provider keys accept the Pilotage provider id (as used elsewhere in
    # this file) or the models.dev provider id; model ids match
    # case-insensitively.
    #
    # Example:
    #   model_overrides:
    #     custom:my-local-server:
    #       my-llava-model:
    #         context_window: 8192
    #         supports_vision: true
    #         supports_reasoning: false
    #         supports_tools: true
    #     _default:            # fill-gap only: models not in the catalog
    #       context_window: 128000
    "model_overrides": {},

    # models.dev registry — provider/model metadata (context windows,
    # capabilities, pricing, modalities).  The agent fetches this on startup
    # and serves from cache; a background daemon refreshes stale data.
    # Override ``url`` to point at a mirror (e.g. a self-hosted copy behind
    # a corporate proxy).  ETag conditional GET ensures refreshes are
    # cheap (304 = no download).
    "models_dev": {
        "url": "",  # empty = default https://models.dev/api.json
    },

    # Network settings — workarounds for connectivity issues.
    "network": {
        # Force IPv4 connections.  On servers with broken or unreachable IPv6,
        # Python tries AAAA records first and hangs for the full TCP timeout
        # before falling back to IPv4.  Set to true to skip IPv6 entirely.
        "force_ipv4": False,
    },

    # Gateway monitoring — Service Health Monitoring plus redacted Operational
    # Diagnostics for the gateway daemon, exported over OTLP to an
    # operator-configured endpoint (OTEL Collector, DataDog, ...). Content-free
    # by construction: no prompts, messages, tool args/results, session
    # history, usage analytics, audit logs, or trajectories. Off by default;
    # nothing is collected or sent until an operator enables it and sets an
    # endpoint.
    "monitoring": {
        # Stable install identifier attached to exported health signals so an
        # operator can tell instances apart in their collector. Empty string
        # means "mint a fresh UUID on first use"; clear it to rotate. Carries
        # no account identity.
        "install_id": "",
        # Gateway health & diagnostics export.
        "gateway_health_export": {
            "enabled": False,
            "metrics_enabled": True,
            "diagnostic_events_enabled": True,
            "warning_error_events_enabled": True,
            "export_interval_seconds": 60,
            "logs_export_interval_seconds": 5,
            "resource_attributes": {
                "service.name": "pilotage-gateway",
                "deployment.environment.name": "production",
            },
        },
        # OTLP destination. headers_env maps header names to ENVIRONMENT
        # VARIABLE NAMES (never secret values); values are read from the
        # environment at export time.
        "export": {
            "otlp": {
                "enabled": False,
                "endpoint": "",
                "headers_env": {},
            },
        },
    },

    # Gateway settings — control how messaging platforms (Telegram,
    # WhatsApp) deliver agent-produced files as native attachments.
    "gateway": {
        # Optional named-profile allowlist for multiplex mode. None preserves
        # the historical serve-all behavior; [] serves only the default.
        "multiplex_profile_allowlist": None,

        # Durable delivery-obligation ledger: final agent responses are
        # recorded in state.db around the platform send, and a gateway that
        # died between finalize and platform ACK redelivers the stored
        # response on the next boot (ambiguous cases carry a visible
        # "recovered reply — may be a duplicate" marker; honest
        # at-least-once). Disable to lose in-flight final responses on
        # crash/restart, as before.
        "delivery_ledger": True,

        # Seconds the gateway waits for a single messaging platform to finish
        # connecting during startup (and on reconnect). A platform can blow
        # past the old fixed 30s when an account has many slash commands to
        # sync (: 90-173 skills → ~28-31s sync). Raise this if your
        # gateway hits connect-timeout restart loops. ``0`` or negative disables
        # the timeout entirely (wait indefinitely). Bridged at startup to the
        # internal PILOTAGE_GATEWAY_PLATFORM_CONNECT_TIMEOUT env var, which still
        # works as a manual override and wins if set explicitly.
        "platform_connect_timeout": 30,

        # In-process event-loop liveness watchdog. A daemon OS thread
        # probes the gateway asyncio loop; after consecutive missed probes it
        # dumps all-thread stacks and hard-exits with the service-restart exit
        # code so the supervisor (systemd/launchd) revives the process instead
        # of leaving a wedged-but-alive zombie. Set to false to disable.
        "loop_watchdog": True,

        # Whether the gateway keeps writing the legacy sessions.json mirror of
        # its routing index. The primary copy lives in state.db (the
        # gateway_routing table). Default True for backward compatibility with
        # external tooling and downgrade safety; set to false to stop
        # producing ~/.pilotage/sessions/sessions.json entirely.
        "write_sessions_json": True,

        # Auto-resume restart-loop breaker (, defense-3). When the
        # gateway is killed mid-turn (SIGTERM) and revived by a supervisor
        # (launchd KeepAlive / systemd Restart=), it auto-resumes the
        # restart-interrupted session on the next boot. If the resumed turn
        # keeps triggering another kill (e.g. the agent runs a raw
        # `launchctl kickstart ai.pilotage.gateway` that defenses 1-2 don't
        # cover), the result is a tight SIGTERM-respawn loop. This breaker
        # chains restart-interrupted boots together and, once `max_restarts`
        # of them chain up, SKIPS auto-resume for that boot — the gateway
        # still starts and serves real inbound messages, it just stops
        # replaying the session that keeps killing it. Set `max_restarts` to
        # 0 to disable the breaker.
        # Two boots belong to the same chain when they are no more than
        # `max_gap_seconds` apart (floored by `window_seconds`). Chaining on
        # the GAP rather than on a fixed window is what makes the breaker see
        # SLOW crash cycles: a loop whose period exceeds the window used to
        # prune its own history on every boot, so the counter never left 1 and
        # the breaker never tripped — e.g. the ~150s wedged-event-loop cycle in
        # (stall -> ~90s liveness-watchdog hard-exit -> respawn ->
        # auto-resume replays the same session), which also makes
        # `pilotage update` hang because it can never drain the gateway.
        "restart_loop_guard": {
            "max_restarts": 3,
            "window_seconds": 60,
            "max_gap_seconds": 300,
        },

        # Portable respawn-storm circuit breaker (complements
        # ``restart_loop_guard`` above). Counts gateway (re)starts in a sliding
        # window and, when too many land, sleeps an exponential backoff before
        # booting so a crash-looping supervisor (launchd KeepAlive, systemd
        # Restart=always) can't hammer the process into a respawn storm.
        # ``max_starts <= 0`` disables the breaker. The env vars
        # ``PILOTAGE_GATEWAY_MAX_STARTS`` / ``PILOTAGE_GATEWAY_START_WINDOW_S``
        # override these defaults for escape-hatch use.
        "respawn_storm": {
            "max_starts": 5,
            "window_seconds": 120,
        },

        # Inject a human-readable timestamp prefix (e.g.
        # "[Tue 2026-04-28 13:40:53 CEST]") onto user messages IN THE MODEL'S
        # CONTEXT so the agent has temporal awareness of when each message was
        # sent. Off by default — when off, the model sees clean message text.
        # Persisted transcripts always stay clean (the timestamp is stored as
        # message metadata regardless of this toggle), so turning it on later
        # surfaces send-times for past messages too.
        "message_timestamps": {
            "enabled": False,
        },

        # Maximum bytes for an inbound image / audio / video payload the
        # gateway will buffer into memory and cache to disk. Inbound media is
        # read fully into RAM before being written, so an unbounded upload
        # or a remote media URL pointing at a
        # huge file can spike memory and OOM-kill the gateway on constrained
        # deployments. Enforced in the shared cache helpers
        # (gateway/platforms/base.py), so the cap holds across every platform
        # adapter. ``0`` disables the cap. Default 128 MiB.
        "max_inbound_media_bytes": 134217728,

        # When false (default), any file path the agent emits is delivered
        # as a native attachment as long as it isn't under the credential /
        # system-path denylist (/etc, /proc, ~/.ssh, ~/.aws, ~/.pilotage/.env,
        # auth.json, etc.). This matches the symmetry of inbound delivery
        # — we accept any document type the user uploads, and the agent
        # can hand back any file that isn't a credential.
        #
        # When true, fall back to the older allowlist+recency-window
        # behavior: files must live under the Pilotage cache, under
        # ``media_delivery_allow_dirs``, or be freshly produced inside the
        # ``trust_recent_files_seconds`` window. Recommended for
        # public-facing gateways where prompt injection from one user
        # shouldn't be able to exfiltrate the host's secrets to that same
        # user. Bridged to PILOTAGE_MEDIA_DELIVERY_STRICT.
        "strict": False,
        # Extra directories from which model-emitted bare file paths may be
        # uploaded as native gateway attachments. Files inside the Pilotage
        # cache (~/.pilotage/cache/{documents,images,audio,video,screenshots})
        # are always trusted; this list adds operator-controlled roots
        # (project dirs, scratch dirs, mounted shares). Accepts a list of
        # absolute paths or a single os.pathsep-separated string. Bridged
        # to PILOTAGE_MEDIA_ALLOW_DIRS at gateway startup. Tilde paths are
        # expanded. Honored in both default and strict mode.
        "media_delivery_allow_dirs": [],
        # When true, files whose mtime is within ``trust_recent_files_seconds``
        # of "now" are trusted for native delivery even outside the cache /
        # operator allowlist — useful for ``pandoc -o /tmp/report.pdf`` or
        # PDFs the agent writes into a working directory. System paths
        # (/etc, /proc, ~/.ssh, ~/.aws, etc.) remain blocked regardless.
        # Disable to fall back to pure-allowlist mode. Bridged to
        # PILOTAGE_MEDIA_TRUST_RECENT_FILES. Only consulted when ``strict``
        # is true; in default mode the denylist alone gates delivery.
        "trust_recent_files": True,
        # Recency window in seconds. 600 (10 min) comfortably covers a
        # multi-tool agent turn. Bridged to PILOTAGE_MEDIA_TRUST_RECENT_SECONDS.
        # Only consulted when ``strict`` is true.
        "trust_recent_files_seconds": 600,

        # OpenAI-compatible API server platform
        # (gateway/platforms/api_server.py).
        "api_server": {
            # Maximum number of agent runs the API server will service
            # concurrently. Requests to /v1/chat/completions, /v1/responses,
            # and /v1/runs that arrive while this many runs are already
            # in flight are rejected with HTTP 429 + a Retry-After header,
            # bounding CPU / memory / upstream-LLM-quota exhaustion from a
            # request flood. Set to 0 to disable the cap entirely.
            "max_concurrent_runs": 10,
        },
    },

    # Real-time token streaming to messaging platforms (Telegram,
    # WhatsApp). Read at the top level by the gateway; absent this block the
    # gateway falls back to these same defaults, so adding it here only makes
    # the feature discoverable in config.yaml — it does not change behavior.
    #
    # Disabled by default: streaming costs extra edit/draft API calls per
    # response. Set ``enabled: true`` and restart the gateway to turn it on.
    "streaming": {
        # Master switch. When false, each response is delivered as a single
        # final message (no progressive updates).
        "enabled": False,
        # Transport selection:
        #   "auto"  — prefer native draft streaming where the platform
        #             supports it (Telegram DMs via sendMessageDraft,
        #             Bot API 9.5+) and fall back to edit-based elsewhere.
        #             Safe global default: platforms without draft support
        #             (WhatsApp, Telegram groups) transparently
        #             use the edit path, so "auto" only upgrades chats that
        #             can render the smoother native preview.
        #   "draft" — explicitly request native drafts; falls back to edit
        #             when the platform/chat doesn't support them.
        #   "edit"  — progressive editMessageText only (legacy behavior).
        #   "off"   — disable streaming entirely (same as enabled: false).
        "transport": "auto",
        # Minimum seconds between progressive edits — tuned for Telegram's
        # ~1 edit/s flood envelope.
        "edit_interval": 0.8,
        # Flush the buffer to the platform once this many characters have
        # accumulated, so short replies feel near-instant.
        "buffer_threshold": 24,
        # Cursor glyph appended to the in-progress message while streaming.
        "cursor": " \u2589",
        # When >0, the final edit for a long-running streamed response is
        # delivered as a fresh message if the preview has been visible at
        # least this many seconds, so the platform timestamp reflects
        # completion time. Telegram only; other platforms ignore it.
        "fresh_final_after_seconds": 0.0,
    },

    # Session storage — controls automatic cleanup of ~/.pilotage/state.db.
    # state.db accumulates every session, message, tool call, and FTS5 index
    # entry forever.  Without auto-pruning, a heavy user (gateway + cron)
    # reports 384MB+ databases with 68K+ messages, which slows down FTS5
    # inserts, /resume listing, and insights queries.
    "sessions": {
        # When true, prune ended sessions inactive for retention_days once
        # per (roughly) min_interval_hours at CLI/gateway/cron startup.
        # Activity is the latest message timestamp, falling back to creation
        # time for empty sessions. Active sessions are always preserved.
        # Default false: session history is valuable for search recall, and
        # silently deleting it could surprise users.  Opt in explicitly.
        "auto_prune": False,
        # How many inactive days of ended-session history to keep. Matches
        # the default of ``pilotage sessions prune``.
        "retention_days": 90,
        # When true, auto-archive (soft-hide, never delete) sessions that
        # haven't been touched in ``auto_archive_days`` days, once per
        # (roughly) min_interval_hours.  "Touched" is last activity, not
        # creation, so an old-but-recently-used session is spared.  Pinned
        # sessions are always exempt.  Off by default — opt in explicitly.
        "auto_archive": False,
        # Idle threshold (days of no activity) before auto-archive hides a
        # session.  Only applies when auto_archive is true.
        "auto_archive_days": 3,
        # VACUUM after a prune that actually deleted rows.  SQLite does not
        # reclaim disk space on DELETE — freed pages are just reused on
        # subsequent INSERTs — so without VACUUM the file stays bloated
        # even after pruning.  VACUUM blocks writes for a few seconds per
        # 100MB, so it only runs at startup, and only when prune deleted
        # ≥1 session.
        "vacuum_after_prune": True,
        # Minimum days between successful VACUUM rewrites. Pruning can still
        # run on its normal cadence while SQLite reuses the freed pages.
        "min_vacuum_interval_days": 30,
        # Minimum hours between auto-maintenance runs (avoids repeating
        # the sweep on every CLI invocation).  Tracked via state_meta in
        # state.db itself, so it's shared across all processes.
        "min_interval_hours": 24,
        # Legacy per-session JSON snapshot writer.  When true, the agent
        # rewrites ``~/.pilotage/sessions/session_{sid}.json`` on every turn
        # boundary with the full message list.  state.db is canonical and
        # has every field the snapshot stored (plus per-message timestamps
        # and token counts), so this is off by default — the snapshots had
        # no consumer outside their own overwrite guard and accumulated
        # GBs of disk on heavy users.  Opt in only if you have an external
        # tool that consumes the JSON files directly.
        "write_json_snapshots": False,
        # Search-index (FTS) storage optimization — the compact v23 layout
        # that drops duplicate content copies and stops trigram-indexing tool
        # output (typically reclaims ~60%+ of state.db on heavy users). It is
        # OPT-IN: existing databases keep their working legacy index until the
        # user runs `pilotage sessions optimize-storage`, because the rebuild is
        # disk-heavy and long on large DBs (see that command's disk preflight).
        #
        #   "advise" (default): `pilotage update` prints a one-line notice with
        #     the reclaimable size and the command, when a legacy index is
        #     detected. Nothing is changed automatically.
        #   "require": the notice is shown as a REQUIRED upgrade (firmer copy),
        #     and future tooling may gate on it. Flip this default in a future
        #     release when we're ready to make the v23 layout mandatory — the
        #     command, progress bar, and resumability are already in place, so
        #     enforcement is a copy/gating change, not new migration code.
        #   "off": suppress the notice entirely.
        "fts_optimize_notice": "advise",
        # CJK-bigram search index (messages_fts_cjk, cjk_unicode61 loadable
        # tokenizer). When the extension is built (native/fts5_cjk/build.sh →
        # ~/.pilotage/lib/libfts5_cjk.so), 1-2 char CJK terms (일본, 项目, ...)
        # get index-speed exact matching instead of LIKE full-table scans.
        # True (default): use the index when the extension is present; the
        # setting is inert when it isn't. False: never load the extension or
        # serve the cjk index. Bridged to PILOTAGE_CJK_FTS (internal carrier).
        "cjk_fts": True,
        # Slow session-search log threshold in milliseconds: searches at or
        # above it log one INFO line with the routing path taken (fts_cjk /
        # fts5 / trigram / like_scan) so latency regressions stay
        # attributable per query shape. 0 logs every search. Bridged to
        # PILOTAGE_SEARCH_SLOW_MS (internal carrier).
        "search_slow_ms": 1000,
        # Transcript safety limits. A runaway session (hundreds of thousands
        # of rows) can exhaust memory when its transcript is materialized in
        # one shot, so interactive resume and in-memory export are guarded by
        # bounded row counts. Set a limit to 0 to disable that guard.
        # Max active messages (across the full compression lineage) a session
        # may hold and still be resumed interactively (CLI/TUI/desktop).
        "max_resume_messages": 20000,
        # Max active messages a single session may hold for an in-memory
        # (non-streaming) export such as `pilotage sessions export`. Checked
        # per session, so full-DB backups of many small sessions still work.
        "max_export_messages": 20000,
    },

    # Contextual first-touch onboarding hints (see agent/onboarding.py).
    # Each hint is shown once per install and then latched here so it
    # never fires again.  Users can wipe the section to re-see all hints.
    "onboarding": {
        "seen": {},
        # Structured profile-build path offered on the very first gateway
        # message ever. "ask" (default) -> offer to build a user profile
        # (opt-in, consent-gated; the agent asks before any lookup and never
        # reads connected accounts silently). "off" -> plain intro only.
        # The offer fires at most once (latched under onboarding.seen).
        "profile_build": "ask",
    },

    # Privacy-safe aggregate metrics written only to this profile's local
    # telemetry directory. Collection is opt-in and no remote sink exists.
    "telemetry": {
        "shared_metrics": {
            "enabled": False,
        },
    },

    # ``pilotage doctor`` behaviour.
    "doctor": {
        # Per-probe timeout (seconds) for the opt-in `pilotage doctor --live`
        # real-call backend probes (Firecrawl/FAL/browser/MCP/TTS/STT).
        "live_probe_timeout": 10,
    },

    # ``pilotage update`` behaviour.
    "updates": {
        # Pre-update safety backup — ONE consolidated mechanism, three modes:
        #
        #   quick (default) — snapshot critical small state files (pairing
        #     JSONs, cron jobs, config.yaml, .env, auth.json, per-profile
        #     DBs) into <PILOTAGE_HOME>/state-snapshots/ before the update.
        #     Files over 1 GiB (e.g. a bloated state.db) are skipped with a
        #     warning so the snapshot stays fast. Restore via ``/snapshot``.
        # This is the (lost pairing data) / (emptied cron
        #     jobs) safety net.
        #   full — the quick snapshot PLUS a full ``pilotage backup``-style zip
        #     of PILOTAGE_HOME into <PILOTAGE_HOME>/backups/, restorable with
        #     ``pilotage import``. Can add minutes on large homes. This is the
        # (wrong-path wipe) safety net. ``--backup`` forces this
        #     for a single run.
        #   off — no pre-update backup of any kind. ``--no-backup`` forces
        #     this for a single run.
        #
        # Legacy boolean values are honored: true -> full, false -> off.
        "pre_update_backup": "quick",
        # How many full pre-update backup zips to retain (mode ``full``).
        # Older ones are pruned automatically after each successful backup.
        # Values below 1 are floored to 1 — the backup just created is
        # always preserved. The quick snapshot always keeps exactly 1.
        "backup_keep": 5,
        # What `pilotage update` does with uncommitted local changes to the
        # source tree when it runs NON-interactively — i.e. triggered from
        # the desktop/chat app or the gateway, where there's no TTY to answer
        # a restore prompt. Interactive (terminal) updates are unaffected:
        # they always stash the changes and ask whether to restore, exactly
        # as they always have.
        #   "stash"   — auto-stash the changes, pull, then auto-restore them
        #               on top of the updated code (the safe default; nothing
        #               is ever lost — conflicts are preserved in a git stash).
        #   "discard" — auto-stash the changes and throw the stash away after
        #               the pull. Use this only if you never intend to keep
        #               local edits to the source tree on this machine.
        #               Stash-and-drop (not `reset --hard` + `clean -fd`) so
        #               ignored paths — node_modules, venv, build outputs —
        #               are never touched.
        "non_interactive_local_changes": "stash",
    },

    # Language Server Protocol — semantic diagnostics from real
    # language servers (pyright, gopls, rust-analyzer, etc.) wired
    # into the post-write lint check used by ``write_file`` and
    # ``patch``.
    #
    # LSP is gated on git-workspace detection: when the agent's
    # cwd (or the file being edited) is inside a git worktree, LSP
    # runs against that workspace.  When neither is in a git repo,
    # LSP stays dormant and the in-process syntax check is the only
    # tier — handy for messaging chats where the cwd is the
    # user's home directory.
    "lsp": {
        # Master toggle.  Setting this to false disables the entire
        # subsystem — no servers spawn, no background event loop, no
        # cost.
        "enabled": True,

        # Diagnostic-wait mode for the post-write check.
        # ``"document"`` waits up to ``wait_timeout`` seconds for the
        # current file's diagnostics; ``"full"`` additionally requests
        # workspace-wide diagnostics (slower).
        "wait_mode": "document",
        "wait_timeout": 5.0,

        # How to handle missing server binaries.
        # ``"auto"`` — try to install via npm/go/pip into
        #              ``<PILOTAGE_HOME>/lsp/bin/`` on first use.
        # ``"manual"`` — only use binaries already on PATH.
        # ``"off"`` — alias for ``manual``.
        "install_strategy": "auto",

        # Idle language servers are shut down automatically after this
        # many seconds with no file activity, then respawned on demand.
        # Prevents long-running gateway/CLI processes from accumulating
        # stale pyright/gopls/tsserver children (hundreds of MB each,
        # plus pipe FDs) as the agent moves across worktrees.  Set to 0
        # to disable idle reaping and keep servers for process lifetime.
        "idle_timeout": 600.0,

        # Per-server overrides.  Each key is a server_id from the
        # registry (``pyright``, ``typescript``, ``gopls``,
        # ``rust-analyzer``, etc.) and accepts:
        #   disabled: true
        #     — skip this server even when its extensions match
        #   command: ["full/path/to/server", "--stdio"]
        #     — pin a custom binary path; bypasses auto-install
        #   env: {"KEY": "value"}
        #     — extra env vars passed to the spawned process
        #   initialization_options: {...}
        #     — merged into the LSP ``initializationOptions``
        # Empty by default; the registry defaults work for typical
        # setups.
        "servers": {},
    },


    # =========================================================================
    # External secret sources
    # =========================================================================
    # Pull credentials from external secret managers at process startup
    # rather than storing them in ~/.pilotage/.env.
    "secrets": {
        # Optional explicit ordering of enabled secret sources.  When
        # omitted, sources run in registration order (bundled first,
        # then plugin-registered).  Regardless of this list, "mapped"
        # sources (explicit VAR→ref bindings, e.g. a future 1Password
        # env: map) always take precedence over "bulk" sources
        # (project dumps like Bitwarden BSM), and the first source to
        # claim a var wins — later claims are skipped with a warning.
        # Example: sources: [onepassword, bitwarden]
        # "sources": [],
        "bitwarden": {
            # Master switch.  When false, BSM is never contacted and the
            # bws binary is never auto-installed — same as not having
            # this section at all.
            "enabled": False,
            # Name of the env var that holds the Bitwarden machine-account
            # access token.  This is the one bootstrap secret; it lives
            # in ~/.pilotage/.env (or your shell) and never in config.yaml.
            "access_token_env": "BWS_ACCESS_TOKEN",
            # UUID of the BSM project to sync from.
            "project_id": "",
            # Seconds to reuse a fresh disk/memory cache entry before contacting
            # Bitwarden again. 0 disables normal fresh-cache reuse.
            "cache_ttl_seconds": 300,
            # Optional encrypted last-good fallback for network/timeout outages.
            # When enabled, successful BWS fetches write AES-GCM encrypted cache
            # material under ~/.pilotage/cache/. If a later startup cannot reach
            # Bitwarden due to NETWORK/TIMEOUT, Pilotage may use this encrypted
            # cache for up to max_stale_seconds. Auth failures do not fall back.
            "encrypted_cache": {
                "enabled": False,
                "max_stale_seconds": 0,
            },
            # When True, BSM values overwrite existing env vars.  Default
            # True because the point of using BSM is centralized rotation —
            # if .env had the final say, rotating in Bitwarden wouldn't
            # take effect until you also cleared the matching .env line.
            "override_existing": True,
            # When True, the bws binary is auto-downloaded into
            # ~/.pilotage/bin/ on first use.  When False you must install
            # bws yourself and have it on PATH.
            "auto_install": True,
            # Bitwarden region / self-hosted endpoint.  Empty string
            # means use the bws CLI default (US Cloud,
            # https://vault.bitwarden.com).  Set to
            # https://vault.bitwarden.eu for EU Cloud, or your own URL
            # for self-hosted Bitwarden.  Plumbed into the bws subprocess
            # as BWS_SERVER_URL.  Prompted for during
            # `pilotage secrets bitwarden setup`.
            "server_url": "",
        },
        "onepassword": {
            # Master switch.  When false, the op CLI is never invoked —
            # same as not having this section at all.
            "enabled": False,
            # Mapping of env-var name → 1Password secret reference
            # (op://vault/item/field).  Each entry is resolved with a
            # single `op read` at startup.
            "env": {},
            # Optional account shorthand / sign-in address passed as
            # `op read --account <account>`.  Empty = op's default account.
            "account": "",
            # Name of the env var holding a 1Password service-account token
            # for headless auth.  Sourced from ~/.pilotage/.env (or the shell)
            # and exported to the op child as OP_SERVICE_ACCOUNT_TOKEN.
            # Leave the var unset to use an interactive/desktop op session.
            "service_account_token_env": "OP_SERVICE_ACCOUNT_TOKEN",
            # Optional absolute path to the op binary.  When set it is used
            # verbatim (PATH is not consulted) — pin this to avoid trusting
            # whatever `op` appears first on PATH.  Empty = resolve via PATH.
            "binary_path": "",
            # Seconds to cache resolved values in-process and on disk.  0
            # disables BOTH cache layers (no values are written to disk).
            "cache_ttl_seconds": 300,
            # When True (default), resolved values overwrite existing env
            # vars so rotating a secret in 1Password takes effect on next
            # start.  Flip to false to let .env / shell exports win locally.
            "override_existing": True,
        },
    },

    # Paste collapse thresholds (TUI + CLI).
    #
    # paste_collapse_threshold (default 5)
    #   Bracketed-paste handler. Pastes with this many newlines or more
    #   collapse to a file reference. Set 0 to disable.
    #
    # paste_collapse_threshold_fallback (default 5)
    #   Fallback heuristic for terminals without bracketed paste support.
    #   Same line count test but heuristically gated by chars-added /
    #   newlines-added to avoid false positives from normal typing.
    #   Set 0 to disable.
    #
    # paste_collapse_char_threshold (default 2000)
    #   Long single-line paste guard. Pastes whose total char length
    #   reaches this value collapse to a file reference even if line
    #   count is below the line threshold. Catches the "8000 chars of
    #   minified JSON / log output on one line" case. Set 0 to disable.
    "paste_collapse_threshold": 5,
    "paste_collapse_threshold_fallback": 5,
    "paste_collapse_char_threshold": 2000,

    # =========================================================================
    # Egress credential-injection proxy (iron-proxy)
    # =========================================================================
    # When enabled, outbound traffic from remote terminal sandboxes (Docker
    # today; Modal/SSH in follow-ups) is routed through a managed iron-proxy
    # subprocess.  The sandbox sees opaque proxy tokens; iron-proxy swaps in
    # real API credentials at the egress boundary.  Compromising the sandbox
    # leaks tokens that only work behind the configured trusted proxy boundary
    # (CA private key + proxy endpoint integrity are part of that boundary).
    #
    # Configure with `pilotage egress setup`.  Disabled by default — the rest of
    # Pilotage works exactly as before with `enabled: false`.
    "proxy": {
        # Master switch.  When false, iron-proxy is never started, no docker
        # mounts are added, no binaries are auto-installed — feature is a
        # complete no-op.
        "enabled": False,
        # Tunnel listener port.  Sandboxes get `HTTPS_PROXY=http://<host>:<port>`.
        # 9090 is the default; collide-aware setup wizard can reassign.
        "tunnel_port": 9090,
        # Auto-download the pinned iron-proxy binary into ~/.pilotage/bin/ on
        # first use.  When false, you must place `iron-proxy` on PATH yourself.
        "auto_install": True,
        # Where iron-proxy looks up the real upstream secrets at egress time.
        # "env"        — process env (default; what bitwarden integration
        #                already populates if you use it)
        # "bitwarden"  — refetch via `bws secret list` on each proxy restart;
        #                rotation in the Bitwarden web app propagates without
        #                touching .env (requires `secrets.bitwarden.enabled`).
        "credential_source": "env",
        # When true, the Docker backend refuses to start a sandbox if the
        # proxy is enabled but not running.  False = fall back to direct
        # outbound with real credentials in the sandbox (the legacy posture).
        "enforce_on_docker": True,
        # NOTE: ``fail_on_uncovered_providers`` was removed.  It gated a
        # refuse-start when third-party provider env vars were
        # present — those providers are now first-class swapped providers
        # via per-provider match_headers rules, so the fail-closed tier is
        # empty.  A leftover
        # key in existing user configs is ignored harmlessly.
        # When credential_source is bitwarden but the BWS access token /
        # project_id is missing OR the bws fetch returns no values for
        # mapped providers, the daemon raises by default.  Set this to
        # True to opt back in to the legacy "silently fall back to host
        # env" behaviour — useful for migrations where the operator wants
        # to switch credential_source to bitwarden but hasn't fully wired
        # BWS yet.  Defaults to false (strict).
        "allow_env_fallback": False,
        # SSRF deny list applied to outbound traffic.  Omit / leave empty
        # to use the safe default: loopback, link-local (incl. cloud
        # metadata IPs at 169.254.169.254), and RFC1918.  Set to an
        # explicit ``[]`` to opt out entirely (only sensible in hermetic
        # tests that need to reach a loopback upstream).
        "upstream_deny_cidrs": None,
        # Extra allowed upstream hosts beyond the bundled defaults (which
        # cover OpenAI).  Wildcards (`*.foo.com`) are supported.
        "extra_allowed_hosts": [],
    },

    # Config schema version - bump this when adding new required fields
    "_config_version": 37,
}

# Optional environment variables that enhance functionality
OPTIONAL_ENV_VARS = {
    # ── Tool API keys ──
    "EXA_API_KEY": {
        "description": "Exa API key for AI-native web search and contents",
        "prompt": "Exa API key",
        "url": "https://exa.ai/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "PARALLEL_API_KEY": {
        "description": "Parallel API key for AI-native web search and extract",
        "prompt": "Parallel API key",
        "url": "https://parallel.ai/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_API_KEY": {
        "description": "Firecrawl API key for web search and scraping",
        "prompt": "Firecrawl API key",
        "url": "https://firecrawl.dev/",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "FIRECRAWL_API_URL": {
        "description": "Firecrawl API URL for self-hosted instances (optional)",
        "prompt": "Firecrawl API URL (leave empty for cloud)",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "FIRECRAWL_GATEWAY_URL": {
        "description": "Exact Firecrawl tool-gateway origin override for Nous Subscribers only (optional)",
        "prompt": "Firecrawl gateway URL (leave empty to derive from domain)",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "TOOL_GATEWAY_DOMAIN": {
        "description": "Shared tool-gateway domain suffix for Nous Subscribers only, used to derive vendor hosts, e.g. nousresearch.com -> firecrawl-gateway.nousresearch.com",
        "prompt": "Tool-gateway domain suffix",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "TOOL_GATEWAY_SCHEME": {
        "description": "Shared tool-gateway URL scheme for Nous Subscribers only, used to derive vendor hosts (`https` by default, set `http` for local gateway testing)",
        "prompt": "Tool-gateway URL scheme",
        "url": None,
        "password": False,
        "category": "tool",
        "advanced": True,
    },
    "TOOL_GATEWAY_USER_TOKEN": {
        "description": "Explicit Nous Subscriber access token for tool-gateway requests (optional; otherwise read from the Pilotage auth store)",
        "prompt": "Tool-gateway user token",
        "url": None,
        "password": True,
        "category": "tool",
        "advanced": True,
    },
    "TAVILY_API_KEY": {
        "description": "Tavily API key for AI-native web search and extract",
        "prompt": "Tavily API key",
        "url": "https://app.tavily.com/home",
        "tools": ["web_search", "web_extract"],
        "password": True,
        "category": "tool",
    },
    "SEARXNG_URL": {
        "description": "URL of your SearXNG instance for free self-hosted web search",
        "prompt": "SearXNG URL (e.g. http://localhost:8080)",
        "url": "https://searxng.github.io/searxng/",
        "tools": ["web_search"],
        "password": False,
        "category": "tool",
    },
    "BRAVE_SEARCH_API_KEY": {
        "description": "Brave Search API subscription token (free tier: 2,000 queries/mo)",
        "prompt": "Brave Search subscription token",
        "url": "https://brave.com/search/api/",
        "tools": ["web_search"],
        "password": True,
        "category": "tool",
    },
    "FAL_KEY": {
        "description": "FAL API key for image generation",
        "prompt": "FAL API key",
        "url": "https://fal.ai/",
        "tools": ["image_generate"],
        "password": True,
        "category": "tool",
    },
    "KREA_API_KEY": {
        "description": "Krea API key for Krea 2 image generation (Medium + Large)",
        "prompt": "Krea API key",
        "url": "https://www.krea.ai/settings/api-tokens",
        "tools": ["image_generate"],
        "password": True,
        "category": "tool",
    },
    "VOICE_TOOLS_OPENAI_KEY": {
        "description": "OpenAI API key for voice transcription (Whisper) and OpenAI TTS",
        "prompt": "OpenAI API Key (for Whisper STT + TTS)",
        "url": "https://platform.openai.com/api-keys",
        "tools": ["voice_transcription", "openai_tts"],
        "password": True,
        "category": "tool",
    },
    "MISTRAL_API_KEY": {
        "description": "Mistral API key for Voxtral TTS and transcription (STT)",
        "prompt": "Mistral API key",
        "url": "https://console.mistral.ai/",
        "password": True,
        "category": "tool",
    },
    "GITHUB_TOKEN": {
        "description": "GitHub token for Skills Hub (higher API rate limits, skill publish)",
        "prompt": "GitHub Token",
        "url": "https://github.com/settings/tokens",
        "password": True,
        "category": "tool",
    },

    # ── Bundled skills (opt-in: only needed if the user uses that skill) ──
    # These use category="skill" (distinct from "tool") so the sandbox
    # env blocklist in tools/environments/local.py does NOT rewrite them —
    # skills legitimately need these passed through to curl via
    # tools/env_passthrough.py when the user's skill calls out.
    "NOTION_API_KEY": {
        "description": "Notion integration token (used by the `notion` skill)",
        "prompt": "Notion API key",
        "url": "https://www.notion.so/my-integrations",
        "password": True,
        "category": "skill",
        "advanced": True,
    },
    "LINEAR_API_KEY": {
        "description": "Linear personal API key (used by the `linear` skill)",
        "prompt": "Linear API key",
        "url": "https://linear.app/settings/account/security",
        "password": True,
        "category": "skill",
        "advanced": True,
    },
    "AIRTABLE_API_KEY": {
        "description": "Airtable personal access token (used by the `airtable` skill)",
        "prompt": "Airtable API key",
        "url": "https://airtable.com/create/tokens",
        "password": True,
        "category": "skill",
        "advanced": True,
    },
    "TENOR_API_KEY": {
        "description": "Tenor API key for GIF search (used by the `gif-search` skill)",
        "prompt": "Tenor API key",
        "url": "https://developers.google.com/tenor/guides/quickstart",
        "password": True,
        "category": "skill",
        "advanced": True,
    },

    # ── Honcho ──
    "HONCHO_API_KEY": {
        "description": "Honcho API key for AI-native persistent memory",
        "prompt": "Honcho API key",
        "url": "https://app.honcho.dev",
        "tools": ["honcho_context"],
        "password": True,
        "category": "tool",
    },
    "HONCHO_BASE_URL": {
        "description": "Base URL for self-hosted Honcho instances (no API key needed)",
        "prompt": "Honcho base URL (e.g. http://localhost:8000)",
        "category": "tool",
    },

    # ── Hindsight ──
    "HINDSIGHT_API_KEY": {
        "description": "Hindsight API key for graph-aware persistent memory",
        "prompt": "Hindsight API key",
        "url": "https://hindsight.vectorize.io",
        "tools": ["hindsight_recall"],
        "password": True,
        "category": "tool",
    },
    "HINDSIGHT_API_URL": {
        "description": "Base URL for the Hindsight API (default: https://api.hindsight.vectorize.io)",
        "prompt": "Hindsight API URL",
        "category": "tool",
        "advanced": True,
    },

    # ── Supermemory ──
    "SUPERMEMORY_API_KEY": {
        "description": "Supermemory API key for conversation-scoped persistent memory",
        "prompt": "Supermemory API key",
        "url": "https://supermemory.ai",
        "tools": ["supermemory_search"],
        "password": True,
        "category": "tool",
    },

    # ── Mem0 ──
    "MEM0_API_KEY": {
        "description": "Mem0 Platform API key for semantic persistent memory",
        "prompt": "Mem0 API key",
        "url": "https://app.mem0.ai",
        "tools": ["mem0_search"],
        "password": True,
        "category": "tool",
    },

    # ── RetainDB ──
    "RETAINDB_API_KEY": {
        "description": "RetainDB API key for persistent memory",
        "prompt": "RetainDB API key",
        "url": "https://retaindb.com",
        "tools": ["retaindb_search"],
        "password": True,
        "category": "tool",
    },
    "RETAINDB_BASE_URL": {
        "description": "Base URL for self-hosted RetainDB instances (default: https://api.retaindb.com)",
        "prompt": "RetainDB base URL",
        "category": "tool",
        "advanced": True,
    },

    # ── ByteRover ──
    "BRV_API_KEY": {
        "description": "ByteRover API key (optional, for cloud sync — local-first by default)",
        "prompt": "ByteRover API key",
        "url": "https://app.byterover.dev",
        "tools": ["brv_query"],
        "password": True,
        "category": "tool",
    },

    # ── OpenViking ──
    "OPENVIKING_API_KEY": {
        "description": "OpenViking API key (leave blank for local dev mode)",
        "prompt": "OpenViking API key",
        "tools": ["viking_search"],
        "password": True,
        "category": "tool",
    },
    "OPENVIKING_ENDPOINT": {
        "description": "OpenViking server URL (default: http://127.0.0.1:1933)",
        "prompt": "OpenViking endpoint",
        "category": "tool",
        "advanced": True,
    },


    # ── Messaging platforms ──
    "TELEGRAM_BOT_TOKEN": {
        "description": "Complete Telegram bot token created by @BotFather (numeric bot ID followed by a colon and secret)",
        "prompt": "Telegram bot token",
        "url": "https://t.me/BotFather",
        "password": True,
        "category": "messaging",
    },
    "TELEGRAM_ALLOWED_USERS": {
        "description": "Optional comma-separated numeric Telegram user IDs allowed immediately; leave blank to approve new users through DM pairing",
        "prompt": "Allowed Telegram user IDs (comma-separated)",
        "url": "https://t.me/userinfobot",
        "password": False,
        "category": "messaging",
    },
    "TELEGRAM_PROXY": {
        "description": "Proxy URL for Telegram connections (overrides HTTPS_PROXY). Supports http://, https://, socks5://",
        "prompt": "Telegram proxy URL (optional)",
        "password": False,
        "category": "messaging",
    },
    "GATEWAY_ALLOW_ALL_USERS": {
        "description": "Allow all users to interact with messaging bots (true/false). Default: false.",
        "prompt": "Allow all users (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_ENABLED": {
        "description": "Enable the OpenAI-compatible API server (true/false). Allows frontends like Open WebUI, LobeChat, etc. to connect.",
        "prompt": "Enable API server (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_KEY": {
        "description": "Bearer token for API server authentication. Required whenever the API server is enabled; server refuses to start without it.",
        "prompt": "API server auth key",
        "url": None,
        "password": True,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_PORT": {
        "description": "Port for the API server (default: 8642).",
        "prompt": "API server port",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_HOST": {
        "description": "Host/bind address for the API server (default: 127.0.0.1). API_SERVER_KEY is still required even on loopback binds.",
        "prompt": "API server host",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "API_SERVER_MODEL_NAME": {
        "description": "Model name advertised on /v1/models. Defaults to the profile name (or 'pilotage-agent' for the default profile). Useful for multi-user setups with OpenWebUI.",
        "prompt": "API server model name",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "GATEWAY_PROXY_URL": {
        "description": "URL of a remote Pilotage API server to forward messages to (proxy mode). When set, the gateway handles platform I/O only — all agent work is delegated to the remote server. Use for Docker E2EE containers that relay to a host agent. Also configurable via gateway.proxy_url in config.yaml.",
        "prompt": "Remote Pilotage API server URL (e.g. http://192.168.1.100:8642)",
        "url": None,
        "password": False,
        "category": "messaging",
        "advanced": True,
    },
    "GATEWAY_PROXY_KEY": {
        "description": "Bearer token for authenticating with the remote Pilotage API server (proxy mode). Must match the API_SERVER_KEY on the remote host.",
        "prompt": "Remote API server auth key",
        "url": None,
        "password": True,
        "category": "messaging",
        "advanced": True,
    },
    "WEBHOOK_ENABLED": {
        "description": "Enable the webhook platform adapter for receiving events from GitHub, GitLab, etc.",
        "prompt": "Enable webhooks (true/false)",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "WEBHOOK_PORT": {
        "description": "Port for the webhook HTTP server (default: 8644).",
        "prompt": "Webhook port",
        "url": None,
        "password": False,
        "category": "messaging",
    },
    "WEBHOOK_SECRET": {
        "description": "Global HMAC secret for webhook signature validation (overridable per route in config.yaml).",
        "prompt": "Webhook secret",
        "url": None,
        "password": True,
        "category": "messaging",
    },

    # ── Agent settings ──
    # NOTE: MESSAGING_CWD was removed here — use terminal.cwd in config.yaml
    # instead.  The gateway reads TERMINAL_CWD (bridged from terminal.cwd).
    "SUDO_PASSWORD": {
        "description": "Sudo password for terminal commands requiring root access; set to an explicit empty string to try empty without prompting",
        "prompt": "Sudo password",
        "url": None,
        "password": True,
        "category": "setting",
    },
    # PILOTAGE_TOOL_PROGRESS_MODE is deprecated — tool progress is configured via
    # display.tool_progress in config.yaml (off|new|all|verbose|log). The
    # gateway still falls back to PILOTAGE_TOOL_PROGRESS_MODE for backward
    # compatibility, so it lives in _EXTRA_ENV_KEYS (known to reload and
    # compatibility paths) but is intentionally NOT listed here:
    # OPTIONAL_ENV_VARS feeds user-facing surfaces (dashboard keys page, setup
    # checklists) and deprecated knobs shouldn't be offered there. The boolean
    # PILOTAGE_TOOL_PROGRESS is fully unsupported since the v12 config support
    # floor retired its only consumer (the v3→4 migration).
    "PILOTAGE_PREFILL_MESSAGES_FILE": {
        "description": "Path to JSON file with ephemeral prefill messages for few-shot priming",
        "prompt": "Prefill messages file path",
        "url": None,
        "password": False,
        "category": "setting",
    },
    "PILOTAGE_EPHEMERAL_SYSTEM_PROMPT": {
        "description": "Ephemeral system prompt injected at API-call time (never persisted to sessions)",
        "prompt": "Ephemeral system prompt",
        "url": None,
        "password": False,
        "category": "setting",
    },
}
