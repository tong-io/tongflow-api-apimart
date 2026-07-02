# tongflow-api-apimart

Official [TongFlow](https://github.com/tong-io/tongflow) plugin for [APIMart](https://apimart.ai) — an OpenAI-compatible aggregation gateway. One API key routes to many upstream models (GPT / Claude / Gemini / Kling / Seedream / Sora / …).

This is a **router-style plugin**: each slot declares a model list (`TONGFLOW_SLOT_MODELS`), so the node shows a **model dropdown** next to the plugin selector and you can pick the backing model per node.

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

- **Generate image** (`image-gen`) — `z-image-turbo` (default), Seedream 4.5 / 4.0 / 5.0-Lite, Nano Banana Pro / 2 / classic (`gemini-3-pro` / `3.1-flash` / `2.5-flash` image previews), `gpt-image-1-official`, `gpt-image-2`, `imagen-4.0-apimart`, `qwen-image-2.0`, `wan2.7-image`, `grok-imagine-1.5-apimart`.
- **Edit image** (`image-edit`) — the image-capable subset of the above (Imagen 4.0 and Grok Imagine are text-to-image only).
- **Generate / rewrite text** (`gen-text`) — `gpt-5` (default), `gpt-5.1`, `gpt-5-mini`, `claude-sonnet-4-6`, `claude-opus-4-8`, `gemini-2.5-pro`, `gemini-3.5-flash`, `gemini-2.5-flash`, `deepseek-v4-pro`, `deepseek-r1-250528`.
- **Text → video** (`text-gen-video`) — `kling-v3` (default), `kling-3.0-turbo`, `kling-v2-6`, `veo3.1-fast` / `-quality` / `-lite`, `sora-2`, `sora-2-pro`, `doubao-seedance-2.0` / `-fast`, `doubao-seedance-1-5-pro`.
- **Image → video** (`image-gen-video`) — same list minus `veo3.1-lite` (text-only); the input image is uploaded to APIMart first.
- **Transcribe audio** (`transcribe`) — `whisper-1`.
- **Text → speech** (`text-gen-speech-preset`) — `gpt-4o-mini-tts` (voices: alloy / echo / fable / onyx / nova / shimmer via the node's speaker field).

Image/video generation is asynchronous on APIMart's side: the plugin submits a task and polls `GET /v1/tasks/{task_id}`, streaming progress to the canvas. Result URLs expire within 24–72 h, so files are downloaded immediately and stored by TongFlow.

## Credentials

Add in TongFlow **Settings** (gear icon, top-right):

| Key | Required | Notes |
| --- | --- | --- |
| `APIMART_API_KEY` | ✅ | Create one in the [APIMart console](https://apimart.ai). |
| `APIMART_BASE_URL` | optional | Override the default `https://api.apimart.ai`. |
| `APIMART_POLL_TIMEOUT_S` | optional | Max seconds to wait for an async task (default `600`). |

Values are stored locally and take effect without a restart.
