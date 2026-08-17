# Video Understanding.app

A native macOS front end for this project. It bundles the Python package,
installs its own runtime on first launch, runs the API on a private loopback
port, and shows the existing web UI in a real app window.

The point is that nothing else is required of you: no terminal, no virtualenv,
no `uvicorn` command, no remembering which port it was on.

```bash
make mac-install     # build, install to /Applications, launch
make mac             # build only, into macapp/build/
```

Requires macOS 12 or newer and the Xcode Command Line Tools
(`xcode-select --install`). There is no Xcode project — `build.sh` drives
`swiftc` directly.

---

## What happens on first launch

1. **FFmpeg check.** If it is missing and Homebrew is present, the app offers to
   run `brew install ffmpeg` for you and shows the output as it goes.
2. **Python check.** It looks for 3.12+ in the Homebrew prefixes, the python.org
   framework locations and on `PATH`, and rejects anything that cannot build a
   virtualenv. Same offer: `brew install python@3.12`.
3. **Runtime install.** A private virtualenv is created at
   `~/Library/Application Support/VideoUnderstanding/runtime`, and the bundled
   package is pip-installed into it with the `whisper` and `scenes` extras. This
   is the slow step — a few hundred megabytes, once, and it needs the network.
4. **Configuration.** A `config.yaml` is written with models sized for this
   machine's RAM, following the table in the main README. It is written once and
   never overwritten; the app never edits it behind your back.
5. **Server start.** `uvicorn` is launched on the first free port in 8756-8790,
   bound to `127.0.0.1` only, and the app waits for `/health` before showing the
   window.

Local models are deliberately *not* part of that sequence. They are several
gigabytes, and the app works without them (with heuristic fallbacks, recorded in
each result's `warnings`). Once the window is up, the app offers to install
Ollama and pull the recommended models in the background — or you can do it
later from **Video Understanding ▸ Local Models…**.

---

## Where things live

Everything mutable is under `~/Library/Application Support/VideoUnderstanding`:

| Path | What it is |
|---|---|
| `runtime/` | The private virtualenv. Safe to delete; rebuilt on next launch. |
| `data/` | Uploads, thumbnails, renders and the SQLite database. |
| `output/` | JSON exports. |
| `config.yaml` | Your configuration. |
| `logs/server.log` | The API server's output for this session. |
| `logs/setup.log` | The last setup run, including pip and Homebrew output. |

Replacing or deleting the app never touches any of it. **Server ▸ Reset
Library…** deletes `data/` after asking twice.

---

## Using it

- **Analyse a video** — drop it on the window, drop it on the dock icon, use
  **File ▸ Analyse Video…** (⌘O), or the Upload button in the UI. All four go
  through the same `POST /v1/videos` endpoint.
- **Downloads** — rendered edits and SRT files land in `~/Downloads` and are
  revealed in Finder.
- **Diagnostics** (⌘D) — the in-app equivalent of `video-understand check`:
  Python, FFmpeg, Ollama, models, port, and the live `/health` payload. The Copy
  button gives you the whole report for a bug report.
- **Configuration** — **Edit Configuration…** opens `config.yaml` in your
  editor; **Restart Server** (⇧⌘R) applies it.

The API is still an API. While the app is running, `http://127.0.0.1:<port>` is
a normal FastAPI server — the port is shown in Diagnostics, and **Help ▸ API
Documentation** opens `/docs` in your browser.

---

## How it is built

```
macapp/
├── build.sh                 assembles the .app: stage, compile, icon, sign
├── Resources/Info.plist     bundle metadata, ATS loopback exception, doc types
├── scripts/MakeIcon.swift   draws the icon with Core Graphics at build time
└── Sources/
    ├── main.swift                   entry point
    ├── AppDelegate.swift            orchestration and menu actions
    ├── MainMenu.swift               the menu bar
    ├── Bootstrapper.swift           first-run setup, step by step
    ├── ServerController.swift       the uvicorn child process, and uploads
    ├── Toolchain.swift              finding Python/FFmpeg/Homebrew/Ollama
    ├── Support.swift                paths, subprocesses, logging, ports
    ├── MainWindowController.swift   the WKWebView window
    ├── SetupWindowController.swift  the first-run window
    ├── InfoWindows.swift            models, diagnostics, logs
    └── UI.swift                     small AppKit helpers
```

`build.sh` produces a universal binary when both slices compile, and falls back
to a native-only build otherwise (`--native-only` forces that). The bundle is
signed ad-hoc, which is all a locally built app needs.

The app is not sandboxed. It could not be: it spawns Python, Homebrew and
Ollama, which is the entire point of it.

---

## Troubleshooting

**"Python 3.12 or newer is required"** — the app tells you what it found. Either
accept the Homebrew offer, or install from python.org and press Try Again.

**Setup fails at the pip step** — almost always the network. The setup log shows
the real error; **Try Again** resumes from where it left off, since the
virtualenv is kept.

**The window is blank** — the server died. **Server ▸ Server Log** says why,
and **Restart Server** brings it back.

**Everything is degraded / results are vague** — the models are not installed.
Diagnostics will say so; **Local Models…** fixes it.

**Starting over completely** — delete
`~/Library/Application Support/VideoUnderstanding` and relaunch. That is the
whole of the app's state.
