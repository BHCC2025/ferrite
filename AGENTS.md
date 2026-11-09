# AGENTS.md: installing Ferrite on a DGX Spark

Instructions for an AI coding agent asked to install Ferrite and start its server on an NVIDIA DGX Spark. Follow the
steps in order, run each verification before going on, and when a stop condition applies, stop and ask the human
instead of working around it.

When you are done, an OpenAI-compatible server answers at `http://127.0.0.1:8000/v1` with the model id `spark`
(Qwen3.8-27B, 128K context, up to 4 requests at once).

## Rules

- Do not use `sudo` and do not install system packages. If a step needs that, stop and ask.
- Do not stop, kill or restart processes you did not start.
- Do not change system settings (sysctl, drivers, CUDA, services, the shell profile).
- Use the default folders unless the human gives others: Ferrite in `~/ferrite` (`FERRITE_HOME`), the models in
  `/var/tmp/models` (`FERRITE_MODELS`).
- Report each command you ran and what its verification printed.

## 1. Check the machine

Run all of these first. Every one must pass.

| Check | Command | Pass when |
|---|---|---|
| Architecture | `uname -m` | `aarch64` |
| GPU | `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader` | the name contains `GB10`, compute capability `12.1`, driver 580 or newer |
| Memory | `free -g` | the `available` value of the `Mem:` line is 95 or more |
| Disk | `df -h /var/tmp` (or the folder for `FERRITE_MODELS`) | at least 45 GB free (38 GB of models, plus Ferrite itself); a source build also needs ~6 GB under your home folder (rustup, cargo, the build); the release tarball needs almost nothing there |
| GPU in use | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` | prints nothing |
| Port | `ss -ltn 'sport = :8000'` | no listening socket on port 8000 |
| Tools | `command -v curl python3 git` | prints three paths |
| Network | `curl -sI https://huggingface.co \| head -n 1` | prints an HTTP status line |

**Stop and ask the human if:**
- the GPU is not a GB10 with compute capability 12.1 (Ferrite's kernels run only on DGX Spark);
- less than 95 GB of memory is available, or another process uses the GPU: the GPU shares system memory, and running
  out of it can stall or reboot the machine;
- port 8000 is taken, or a tool is missing (installing it needs `sudo`);
- there is less disk space than above, or no network access.

## 2. Install and start

Run one command. It installs Ferrite into `~/ferrite`, the Hugging Face CLI into `~/.venvs/hf` and the models into
`/var/tmp/models`, starts the server in the background and sends it a test request. It needs no `sudo`, asks no
questions, and can be run again: finished steps are skipped.

```bash
curl -fsSL https://raw.githubusercontent.com/BHCC2025/ferrite/main/scripts/install.sh | bash
```

It prints six steps (`==> [1/6] Checking this machine` to `==> [6/6]`). It uses the release's ready-made binaries
when the release exists, else it clones the repository and builds from source (a few minutes). Downloading the
models (about 38 GB) takes most of the time; checking them takes about 2 minutes and loading the server about a
minute.

**Success** is a final line like
`OK: Ferrite is serving http://127.0.0.1:8000/v1 (model id 'spark')`, followed by the log path and the stop command.

**Stop and ask the human if** it ends with `install: ERROR: ...`. The message says what is wrong: a missing system
package (with the `sudo apt install` command for the human to run), too little memory, the wrong GPU, a failed
download or build, or "the server stopped while starting" (followed by the end of the server log). Report the
message; do not retry with other settings unless the human says so.

## 3. Verify

```bash
curl -s http://127.0.0.1:8000/health
```

Pass: JSON with `"status":"ok"` and `"model":"spark"`.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "spark",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
  "max_tokens": 300,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Pass: JSON whose `choices[0].message.content` holds the function.

## 4. Use, stop and restart the server

- **API:** base URL `http://127.0.0.1:8000/v1`, model `spark`, any non-empty API key. Any OpenAI-compatible client
  works. The server listens on 127.0.0.1 only; serving the network (`FERRITE_HOST=0.0.0.0`) has no API key, so ask
  the human first.
- **Log:** `~/ferrite/logs/server.log`.
- **Stop:** `kill -- -$(cat ~/ferrite/logs/server.pid)`. Verify: the `/health` request fails, and the GPU check of
  section 1 prints nothing again.
- **Start again:** `~/ferrite/scripts/install.sh` (it skips the finished steps, starts the server and tests it).

## Troubleshooting

| Symptom | Meaning | What to do |
|---|---|---|
| `install: ERROR: ... sudo apt install -y <package>` | a system package is missing | stop; give the human the command |
| `install: ERROR: only N GB of memory is available` | other programs use the memory | stop; ask the human |
| `the server stopped while starting` | the server exited during load; the log tail is printed | stop; report the log tail |
| `--slots N is not supported with DFlash2 drafting` | more than 4 slots were asked for | leave `FERRITE_SLOTS` at 4 (the default) |
| `another Ferrite GPU job holds /var/tmp/ferrite-gpu.lock` | a Ferrite server or job is already running | stop; ask the human whether to use or stop it |
| a download fails | network or Hugging Face error | run the install command again (downloads resume); if it keeps failing, stop and report |

More: [docs/quickstart.md](docs/quickstart.md) (manual steps, every setting, two Sparks),
[docs/prompt-cache.md](docs/prompt-cache.md) (the prompt cache), [README.md](README.md).
