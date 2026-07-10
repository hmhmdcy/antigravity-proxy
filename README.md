# Antigravity Proxy

An OpenAI-compatible proxy that bridges any OpenAI-compatible client to **Google Antigravity's** internal Cloud Code Assist API. Uses the OAuth token from the [Antigravity CLI](https://antigravity.google/) (`agy`) — no API key, no billing.

Access **Gemini 3.5 Flash**, **Gemini 3.1 Pro**, **Claude Sonnet 4.6**, **Claude Opus 4.6**, and more through a single local endpoint.

## Features

- **OpenAI-compatible** `/v1/chat/completions` endpoint (streaming + non-streaming)
- **Native Gemini passthrough** `/v1beta/models/{model}:generateContent` endpoint
- **Tool/function calling** support (OpenAI tools → Gemini functionDeclarations, round-tripped)
- **Model auto-mapping** — use friendly names (`gemini-3.5-flash`), the proxy translates to backend names automatically
- **Token auto-refresh** — OAuth token refreshed automatically when expired
- **Zero dependencies** — pure Python 3.10+, stdlib only
- **systemd integration** — run as a background service

## Available Models

| Friendly Name        | Backend Model              | Notes                          |
|----------------------|----------------------------|--------------------------------|
| `gemini-3.1-pro`     | `gemini-3.1-pro-low`       | Thinking level: low            |
| `gemini-3.1-pro-high`| `gemini-3.1-pro-low`       | Thinking level: high           |
| `gemini-3-flash`     | `gemini-3-flash`           |                                |
| `gemini-3.5-flash`   | `gemini-3.5-flash-low`     |                                |
| `gemini-2.5-pro`     | `gemini-2.5-pro`           |                                |
| `gemini-2.5-flash`   | `gemini-2.5-flash`         |                                |
| `claude-sonnet-4.6`  | `claude-sonnet-4-6`        | Via Antigravity backend        |
| `claude-opus-4.6`    | `claude-opus-4-6-thinking` | Thinking enabled               |

Model names are mapped automatically on both endpoints. You always use the friendly name in your client config.

---

## Prerequisites

You need the **Antigravity CLI** (`agy`) installed and authenticated. The proxy reads the OAuth token that `agy` stores on disk.

### Install Antigravity CLI

**macOS / Linux:**
```bash
curl -fsSL https://storage.googleapis.com/antigravity-releases/install.sh | bash
```

**Windows:**
Download from [antigravity.google](https://antigravity.google/) or use `winget`:
```
winget install Google.Antigravity
```

### Authenticate

Run the CLI once to complete the Google OAuth flow:
```bash
agy
```
This opens a browser, you sign in with your Google account, and the token is saved to `~/.gemini/antigravity-cli/antigravity-oauth-token`.

> **Note:** The proxy supports both consumer Google accounts and Google Workspace accounts.

---

## Quick Start

### Option A: Run directly

```bash
git clone https://github.com/usamashehab/antigravity-proxy.git
cd antigravity-proxy
python3 antigravity_proxy.py
```

The proxy starts on `http://127.0.0.1:8877`.

### Option B: Run as a systemd service (recommended)

```bash
# 1. Clone to a permanent location
git clone https://github.com/usamashehab/antigravity-proxy.git ~/antigravity-proxy

# 2. Copy the service file
mkdir -p ~/.config/systemd/user
cp ~/antigravity-proxy/antigravity-proxy.service ~/.config/systemd/user/

# 3. If you use a venv, edit the ExecStart path in the service file to point to your python

# 4. Reload systemd and start
systemctl --user daemon-reload
systemctl --user enable antigravity-proxy
systemctl --user start antigravity-proxy

# 5. Verify
systemctl --user status antigravity-proxy
curl http://127.0.0.1:8877/health
```

### Option C: Run with custom host/port

```bash
python3 antigravity_proxy.py --port 9000 --host 0.0.0.0
```

> **Warning:** Binding to `0.0.0.0` exposes the proxy to your network. The proxy has no authentication layer — anyone who can reach it can use your Antigravity quota. Use `127.0.0.1` unless you know what you're doing.

---

## Configuration

### Point your client at the proxy

Any OpenAI-compatible client works. Set:

| Setting     | Value                          |
|-------------|--------------------------------|
| Base URL    | `http://127.0.0.1:8877/v1`    |
| API Key     | `antigravity` (any string)     |
| Model       | Any name from the table above  |

#### Examples

**OpenAI Python SDK:**
```python
from openai import OpenAI
client = OpenAI(
    base_url="http://127.0.0.1:8877/v1",
    api_key="antigravity",
)
response = client.chat.completions.create(
    model="gemini-3.5-flash",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

**curl:**
```bash
curl http://127.0.0.1:8877/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.5-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Hermes Agent:**
```bash
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8877/v1
hermes config set model.api_key antigravity
hermes config set model.default gemini-3.5-flash
```

**Claude Code:**
```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8877/v1
claude  # uses claude-sonnet-4.6 through the proxy
```

**Continue / Aider / any OpenAI-compatible tool:** Set base URL to `http://127.0.0.1:8877/v1` and use any model from the table.

---

## Two Endpoints

The proxy serves two distinct API surfaces:

### 1. OpenAI Chat Completions (default)

```
POST /v1/chat/completions
```

Standard OpenAI format. Messages, tools, streaming — all supported. Use this for any OpenAI-compatible client.

### 2. Native Gemini Passthrough

```
POST /v1beta/models/{model}:generateContent
```

Accepts the exact same request format as Google's `generativelanguage.googleapis.com` API. Supports `inline_data` (images), `responseSchema` (structured output), `thinkingConfig`, and everything else — all passed through untouched.

Use this when you need features that don't map to OpenAI format (image inputs, structured JSON schema output, etc.).

**Example — structured output with a response schema:**
```bash
curl http://127.0.0.1:8877/v1beta/models/gemini-3.5-flash:generateContent \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [{"parts": [{"text": "Extract: Ahmed is 30 years old"}]}],
    "generationConfig": {
      "temperature": 0,
      "responseMimeType": "application/json",
      "responseSchema": {
        "type": "OBJECT",
        "properties": {
          "name": {"type": "STRING"},
          "age": {"type": "NUMBER"}
        },
        "required": ["name", "age"]
      }
    }
  }'
```

---

## Management

### Check health
```bash
curl http://127.0.0.1:8877/health
# {"status": "ok", "service": "antigravity-proxy"}
```

### List available models
```bash
curl http://127.0.0.1:8877/v1/models
```

### View logs (systemd)
```bash
journalctl --user -u antigravity-proxy -f
```

### Restart
```bash
systemctl --user restart antigravity-proxy
```

### Stop
```bash
systemctl --user stop antigravity-proxy
```

### Token expired?

If the proxy logs show authentication errors, re-authenticate with the Antigravity CLI:
```bash
agy  # opens browser for re-login
systemctl --user restart antigravity-proxy
```

---

## Adding / Updating Models

Edit `MODEL_MAP` at the top of `antigravity_proxy.py`:

```python
MODEL_MAP = {
    "gemini-3.5-flash": ("gemini-3.5-flash-low", None),
    # format: "friendly-name": ("backend-name", thinking_level_or_None)
}
```

- **Friendly name**: what clients send (e.g. `gemini-3.5-flash`)
- **Backend name**: the model ID Antigravity's API expects (e.g. `gemini-3.5-flash-low`)
- **Thinking level**: `"low"`, `"high"`, or `None`

After editing, restart the proxy.

---

## How It Works

```
Your Client (OpenAI format)
    │
    ▼
Antigravity Proxy (localhost:8877)
    │  ┌─ Translates OpenAI → Gemini format
    │  ├─ Manages OAuth token (auto-refresh)
    │  └─ Wraps in Antigravity envelope
    │
    ▼
cloudcode-pa.googleapis.com (Google's internal Code Assist API)
    │
    ▼
Gemini / Claude models
```

The proxy:
1. Reads the OAuth token from `~/.gemini/antigravity-cli/antigravity-oauth-token`
2. Refreshes it automatically via `oauth2.googleapis.com` when expired
3. Discovers your Cloud Code Assist project ID via `v1internal:loadCodeAssist`
4. Wraps each request in the Antigravity envelope (`{project, model, request, requestType}`)
5. Forwards to `v1internal:generateContent` (or `streamGenerateContent`)
6. Unwraps the response and converts back to OpenAI format

---

## Requirements

- Python 3.10+
- Antigravity CLI (`agy`) installed and authenticated
- No pip dependencies — stdlib only

---

## FAQ

**Is this legal?**

The proxy uses the same OAuth token and API endpoints that the Antigravity CLI itself uses. It does not bypass authentication or payment — you still need a valid Google account with Antigravity access. Multiple similar projects exist on GitHub. That said, Google's terms of service may restrict programmatic API access. Use at your own discretion.

**Does it support streaming?**

Yes. Both SSE streaming (`stream: true`) on the OpenAI endpoint and the native Gemini stream endpoint.

**Can I use it with images?**

Yes, through the native Gemini passthrough endpoint (`/v1beta/models/{model}:generateContent`). Send `inline_data` with base64-encoded images, exactly like Google's API. The OpenAI chat endpoint does not currently support image inputs.

**Is there a rate limit?**

Rate limits are determined by Google's Antigravity quota, not the proxy. The proxy itself is multithreaded and has no concurrency limit.

**Does the token expire?**

Yes, the OAuth access token expires (~1 hour). The proxy auto-refreshes it using the refresh token. The refresh token itself can expire if unused for extended periods — re-run `agy` if that happens.

---

## License

MIT
