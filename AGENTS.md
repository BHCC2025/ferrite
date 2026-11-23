# AGENTS.md: installing Ferrite on a DGX Spark

Instructions for an AI coding agent asked to install Ferrite and start its server on an NVIDIA DGX Spark. Follow the
steps in order, run each verification before going on, and when a stop condition applies, stop and ask the human
instead of working around it.

These steps are for a DGX Spark only. On any other hardware (RTX 5080, RTX 5060 Ti, RTX PRO 4000 Blackwell, Intel
Arc Pro B70), stop and follow that hardware's guide from the table in [README.md](README.md#what-runs-where).
This file is what an agent follows on its own; its figures (memory, disk, port) are those of
[docs/quickstart.md](docs/quickstart.md#requirements).

When you are done, an OpenAI-compatible server answers at `http://<spark>:8000/v1` (and `http://127.0.0.1:8000/v1`
on the Spark) with the model id `qwen3.8-27b` (Qwen3.8-27B, 128K context, up to 4 requests at once).

**Tell the human before you start:** the server listens on every interface by default, so any machine on their
local network can use it, and it does not check an API key. That is what lets a client on another PC reach it.
Otherwise they should use it on a trusted network or firewall port 8000. If they want it reachable from the Spark
only, use this install command in step 2 instead (the variable goes after the pipe, before `bash`):

```bash
curl -fsSL https://raw.githubusercontent.com/BHCC2025/ferrite/main/scripts/install.sh | FERRITE_HOST=127.0.0.1 bash
```

## Rules

- Do not use `sudo` and do not install system packages. If a step needs that, stop and ask.
- Do not stop, kill or restart processes you did not start.
- Do not change system settings (sysctl, drivers, CUDA, services, the shell profile).
- Use the default folders unless the human gives others: Ferrite in `~/ferrite` (`FERRITE_HOME`), the models in
  `/var/tmp/models` (`FERRITE_MODELS`).
- Report each command you ran and what its verification printed. At the end, hand back what
  [section 4](#4-use-stop-and-restart-the-server) lists: the base URL, the model id, the log path and the stop
  command, and that the server is open to the local network unless `FERRITE_HOST=127.0.0.1` was used.

## 1. Check the machine

Run all of these first. Every one must pass.

| Check | Command | Pass when |
|---|---|---|
| Architecture | `uname -m` | `aarch64` |
| GPU | `nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader` | the name contains `GB10`, compute capability `12.1`, driver 580 or newer |
| Memory | `free -g` | the `available` value of the `Mem:` line is 95 or more |
| Disk | `df -h /var/tmp` (or the folder for `FERRITE_MODELS`) | at least 45 GB free (about 38 GB of models, plus Ferrite itself); the source build also needs ~6 GB under the user's home folder (rustup, cargo, the build) |
| GPU in use | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` | prints nothing |
| Port | `ss -ltn 'sport = :8000'` | no listening socket on port 8000 |
| Tools | `command -v curl python3 git` | prints three paths |
| Network | `for u in https://huggingface.co https://github.com https://api.github.com https://sh.rustup.rs; do curl -sI -m 20 $u \| head -n 1; done` | prints four HTTP status lines: the models come from Hugging Face, the installer, the release name and the source from GitHub, and rustup (if it is not installed yet) from sh.rustup.rs |

**Stop and ask the human if:**
- the GPU is not a GB10 with compute capability 12.1 (Ferrite's kernels run only on DGX Spark);
- less than 95 GB of memory is available, or another process uses the GPU: the GPU shares system memory, and running
  out of it can stall or reboot the machine;
- port 8000 is taken, or a tool is missing (installing it needs `sudo`);
- there is less disk space than above, or one of the four hosts of the network check does not answer.

## 2. Install and start

Run one command. It installs Ferrite into `~/ferrite`, the Hugging Face CLI into `~/.venvs/hf` and the models into
`/var/tmp/models`, starts the server in the background and sends it a test request. It needs no `sudo`, asks no
questions, and can be run again: finished steps are skipped.

```bash
curl -fsSL https://raw.githubusercontent.com/BHCC2025/ferrite/main/scripts/install.sh | bash
```

It prints six steps (`==> [1/6] Checking this machine` to `==> [6/6]`). It clones the latest published release
(step 2 names it) and builds it from source (a few minutes; Ferrite ships no prebuilt binaries). Downloading the
models (about 38 GB) takes most of the time. Verifying the downloaded files takes about 2 minutes and loading the
server about a minute.

**Success** is a final line like
`OK: Ferrite is serving http://spark:8000/v1 (model id 'qwen3.8-27b'), and http://127.0.0.1:8000/v1 on this machine`
(`spark` is an example: the line shows this machine's own host name), followed by a line about the open network
access, the log path and the stop command.

**Stop and ask the human if** it ends with `install: ERROR: ...`. The message says what is wrong: a missing system
package (with the `sudo apt install` command for the human to run), too little memory, the wrong GPU, a failed
download or build, or "the server stopped while starting" (followed by the end of the server log). Report the
message; do not retry with other settings unless the human says so.

Run again later, the same command upgrades an older release in `~/ferrite`: the old folder is kept as
`~/ferrite-<old version>`. If that release's server is still running, the upgrade stops and asks for it to be stopped
first; ask the human before you stop it.

## 3. Verify

```bash
curl -s http://127.0.0.1:8000/health
```

Pass: JSON with `"status":"ok"` and `"model":"qwen3.8-27b"`.

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "Write a Python function that reverses a string."}],
  "max_tokens": 300,
  "chat_template_kwargs": {"enable_thinking": false}
}'
```

Pass: JSON whose `choices[0].message.content` holds the function.

## 4. Use, stop and restart the server

- **API:** base URL `http://<spark>:8000/v1` from other machines, `http://127.0.0.1:8000/v1` on the Spark; model
  `qwen3.8-27b`. The server does not check the API key: a client that insists on one can be given any non-empty
  text. Any OpenAI-compatible client works. The server listens on every interface (see the top of this page);
  `FERRITE_HOST=127.0.0.1` restricts it to the Spark.
- **Log:** `~/ferrite/logs/server-8000.log` (`~/ferrite/scripts/ferritectl.sh log`).
- **Status:** `~/ferrite/scripts/ferritectl.sh status`.
- **Stop:** `~/ferrite/scripts/ferritectl.sh stop`. Verify: the `/health` request fails, and the GPU check of
  section 1 prints nothing again.
- **Start again:** `~/ferrite/scripts/ferritectl.sh start` (or `~/ferrite/scripts/install.sh`, which skips the
  finished steps, starts the server and tests it).

## Troubleshooting

| Symptom | Meaning | What to do |
|---|---|---|
| `install: ERROR: ... sudo apt install -y <package>` | a system package is missing | stop; give the human the command |
| `install: ERROR: only N GB of memory is available` | other programs use the memory | stop; ask the human |
| `the server stopped while starting` | the server exited during load; the log tail is printed | stop; report the log tail |
| `--slots N is not supported with DFlash2 drafting` | more than 4 slots were asked for | leave `FERRITE_SLOTS` at 4 (the default) |
| `another Ferrite GPU job holds /var/tmp/ferrite-gpu.lock (pid ...)` | a Ferrite server or job is already running | stop; ask the human whether to use or stop it |
| `port 8000 is taken by another server` / `is already in use (by ...)` | another program has the port | stop; ask the human (or use another `FERRITE_PORT` if they say so) |
| a download fails | network or Hugging Face error | run the install command again (it fetches only what is missing or wrong); if it keeps failing, stop and report |

More: [docs/quickstart.md](docs/quickstart.md) (manual steps, every setting, two Sparks, uninstalling),
[docs/prompt-cache.md](docs/prompt-cache.md) (the prompt cache), [README.md](README.md).
