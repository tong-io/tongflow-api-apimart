# tongflow-api-apimart

Official [TongFlow](https://github.com/tong-io/tongflow) plugin for [APIMart](https://apimart.ai) — an OpenAI-compatible aggregation gateway. One API key routes to many upstream models (GPT / Claude / Gemini / Kling / Seedream / Sora / …).

This is a **router-style plugin**: each slot declares a model list (`TONGFLOW_SLOT_MODELS`), so the node shows a **model dropdown** next to the plugin selector and you can pick the backing model per node.

## Capabilities

Implements these ABI slots (runs locally as a Python process, no GPU):

- **Generate image** (`image-gen`) — `z-image-turbo` (default), `doubao-seedream-4-5`, `gemini-3-pro-image-preview`, `gpt-image-1-official`.
- **Edit image** (`image-edit`) — `gemini-3-pro-image-preview` (default), `doubao-seedream-4-5`, `gpt-image-1-official`.
- **Generate / rewrite text** (`gen-text`) — `gpt-5` (default), `claude-sonnet-4-6`, `gemini-2.5-pro`, `deepseek-v4-pro`.
- **Text → video** (`text-gen-video`) — `kling-v3` (default), `veo3.1-fast`, `sora-2`, `doubao-seedance-2.0`.
- **Image → video** (`image-gen-video`) — same model list; the input image is uploaded to APIMart first.
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
