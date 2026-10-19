# Gander τ-Voice adapter (first-pass)

Copy this directory to `src/tau2/voice/audio_native/gander/` in the
`feat-moshi-support` branch, then apply `registration.patch` manually (the
patch has deliberately small context because the target branch may drift).

Set `GANDER_WS_URL` on the machine that runs τ-Voice:

```bash
export GANDER_WS_URL=ws://127.0.0.1:8000/ws/duplex
```

Run an initial smoke test with the same one-second unit used by the official
Gander server:

```bash
tau2 run --domain retail --audio-native --audio-native-provider gander \
  --tick-duration 1.0 --speech-complexity control --num-tasks 1 --verbose-logs
```

## Important current limitation

The stock `./scripts/serve.sh` endpoint fixes Gander's system prompt and tool
schemas when the server starts.  Its documented WebSocket protocol has no
`session.configure` operation.  Therefore this adapter can transport audio,
events and tool responses, but a valid τ-Voice run still requires a small
Gander server-side change that accepts the τ policy/tool schemas at session
creation (or starts the service with the exact schemas for one selected τ
domain).  Do **not** run its default Codex Brain for the benchmark: τ-Voice
must execute the tool calls itself.

This is intentionally left as the first integration checkpoint: first verify
the transport/format/event protocol, then make the server-side schema injection
change once the exact error or current `serve.local.yaml` is available.
