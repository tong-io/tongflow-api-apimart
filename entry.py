"""TongFlow plugin for APIMart (https://docs.apimart.ai) — an OpenAI-compatible
aggregation gateway. One API key routes to many upstream models (GPT / Claude /
Gemini / Kling / Seedream / Sora / ...), so this is a router-style plugin: each
slot declares its model list in TONGFLOW_SLOT_MODELS and the platform passes the
node's selection back via the request envelope's top-level ``model`` field.

Generation endpoints are asynchronous: POST returns ``{code, data:[{task_id}]}``
and the plugin polls ``GET /v1/tasks/{task_id}`` until completion, forwarding
progress to the canvas.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tongflow.node_slots import NodeSlots
from tongflow.progress import progress
from tongflow.protocol import Asset, asset
from tongflow.slots import node_slot
from tongflow.models.gen_text import GenTextInput, GenTextOutput
from tongflow.models.image_edit import ImageEditInput, ImageEditOutput
from tongflow.models.image_gen import ImageGenInput, ImageGenOutput
from tongflow.models.image_gen_video import ImageGenVideoInput, ImageGenVideoOutput
from tongflow.models.text_gen_speech_preset import (
    TextGenSpeechPresetInput,
    TextGenSpeechPresetOutput,
)
from tongflow.models.text_gen_video import TextGenVideoInput, TextGenVideoOutput
from tongflow.models.transcribe import TranscribeInput, TranscribeOutput

# Per-slot model lists surfaced as the node's model dropdown. Must stay a pure
# dict literal — the platform reads it by AST without importing this module.
# First entry per slot = default. Keys must match the spec tables below
# (checked in main()).
TONGFLOW_SLOT_MODELS = {
    "image-gen": [
        "z-image-turbo",
        "doubao-seedream-4-5",
        "gemini-3-pro-image-preview",
        "gpt-image-1-official",
    ],
    "image-edit": [
        "gemini-3-pro-image-preview",
        "doubao-seedream-4-5",
        "gpt-image-1-official",
    ],
    "gen-text": [
        "gpt-5",
        "claude-sonnet-4-6",
        "gemini-2.5-pro",
        "deepseek-v4-pro",
    ],
    "text-gen-video": [
        "kling-v3",
        "veo3.1-fast",
        "sora-2",
        "doubao-seedance-2.0",
    ],
    "image-gen-video": [
        "kling-v3",
        "veo3.1-fast",
        "sora-2",
        "doubao-seedance-2.0",
    ],
    "transcribe": ["whisper-1"],
    "text-gen-speech-preset": ["gpt-4o-mini-tts"],
}

# Plugin logs go to stderr — stdout is reserved for the ABI JSON response.
logging.basicConfig(
    level=os.environ.get("TONGFLOW_PLUGIN_LOG_LEVEL", "INFO").upper(),
    stream=sys.stderr,
    format="[apimart] %(levelname)s %(message)s",
)
log = logging.getLogger("tongflow.plugins.apimart")

DEFAULT_BASE_URL = "https://api.apimart.ai"
DEFAULT_POLL_TIMEOUT_S = 600.0
POLL_INTERVAL_S = 10.0

# Model chosen on the node; set by main() from the request envelope. Empty →
# each slot's default (first entry in its spec table).
_REQUEST_MODEL: str = ""


def _require_api_key() -> str:
    api_key = os.environ.get("APIMART_API_KEY")
    if not api_key:
        raise RuntimeError(
            "APIMART_API_KEY is not set. Create one at https://apimart.ai "
            "and add it in TongFlow Settings."
        )
    return api_key


def _base_url() -> str:
    return (os.environ.get("APIMART_BASE_URL") or "").strip() or DEFAULT_BASE_URL


def _poll_timeout() -> float:
    raw = (os.environ.get("APIMART_POLL_TIMEOUT_S") or "").strip()
    try:
        return float(raw) if raw else DEFAULT_POLL_TIMEOUT_S
    except ValueError:
        return DEFAULT_POLL_TIMEOUT_S


# ── HTTP helpers ───────────────────────────────────────────────────────────


def _http(
    method: str,
    path: str,
    data: bytes | None,
    content_type: str,
    timeout: float = 180,
) -> Tuple[bytes, str]:
    """Raw request against the APIMart base URL; returns (body, content_type)."""
    url = _base_url().rstrip("/") + path
    headers = {
        "Authorization": f"Bearer {_require_api_key()}",
        "Content-Type": content_type,
    }
    log.info("%s %s", method, path)
    req = Request(url, data=data, headers=headers, method=method)
    try:
        resp = urlopen(req, timeout=timeout)  # noqa: S310
    except HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        log.error("HTTP %s on %s\nresponse body: %s", e.code, path, err_body)
        raise RuntimeError(
            f"HTTP {e.code} from APIMart: {err_body or e.reason}"
        ) from e
    except URLError as e:
        log.error("network error contacting %s: %s", path, e.reason)
        raise RuntimeError(f"Network error: {e.reason}") from e
    return resp.read(), resp.headers.get_content_type() or ""


def _request_json(
    method: str, path: str, body: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    raw, _mime = _http(method, path, data, "application/json")
    text = raw.decode("utf-8", errors="replace")
    obj = json.loads(text) if text.strip() else {}
    if not isinstance(obj, dict):
        raise RuntimeError(f"Unexpected non-object response from {path}: {text[:200]}")
    return obj


def _multipart(
    fields: Dict[str, str], files: List[Tuple[str, str, str, bytes]]
) -> Tuple[bytes, str]:
    boundary = "----tongflow" + os.urandom(16).hex()
    line = boundary.encode()
    parts: List[bytes] = []
    for name, value in fields.items():
        parts.append(b"--" + line)
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(b"")
        parts.append(value.encode("utf-8"))
    for fname, filename, mime, content in files:
        parts.append(b"--" + line)
        parts.append(
            f'Content-Disposition: form-data; name="{fname}"; filename="{filename}"'.encode()
        )
        parts.append(f"Content-Type: {mime}".encode())
        parts.append(b"")
        parts.append(content)
    parts.append(b"--" + line + b"--")
    parts.append(b"")
    return b"\r\n".join(parts), f"multipart/form-data; boundary={boundary}"


def _download(url: str) -> Tuple[bytes, str]:
    """Download a result immediately — APIMart URLs expire within 24-72h."""
    try:
        resp = urlopen(url, timeout=300)  # noqa: S310
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} downloading result: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Network error downloading result: {e.reason}") from e
    return resp.read(), resp.headers.get_content_type() or ""


def _upload_image(a: Asset) -> str:
    """Upload an input image and return its temporary public URL (valid 72h).

    Generation endpoints take image inputs as URLs; uploading is the one
    transport that works across every model family.
    """
    content = base64.b64decode(a.bytesBase64)
    mime = (a.mime or "image/png").strip() or "image/png"
    ext = mime.rsplit("/", 1)[-1] or "png"
    filename = a.filename or f"input.{ext}"
    data, content_type = _multipart({}, [("file", filename, mime, content)])
    raw, _mime = _http("POST", "/v1/uploads/images", data, content_type)
    obj = json.loads(raw.decode("utf-8", errors="replace"))
    url = obj.get("url") or (obj.get("data") or {}).get("url") if isinstance(obj, dict) else None
    if not isinstance(url, str) or not url:
        raise RuntimeError(f"Upload returned no url: {str(obj)[:200]}")
    return url


# ── Async task submit + poll ───────────────────────────────────────────────


def _submit(path: str, body: Dict[str, Any]) -> str:
    obj = _request_json("POST", path, body)
    data = obj.get("data")
    entry: Any = None
    if isinstance(data, list) and data:
        entry = data[0]
    elif isinstance(data, dict):
        entry = data
    task_id = entry.get("task_id") if isinstance(entry, dict) else None
    if not isinstance(task_id, str) or not task_id:
        raise RuntimeError(f"APIMart submit returned no task_id: {str(obj)[:300]}")
    log.info("submitted task %s", task_id)
    return task_id


def _first_url(value: Any) -> Optional[str]:
    """Extract the first URL from APIMart's result shapes.

    Observed forms: ``result.images -> [{url: [<str>, ...]}]``; videos may use
    ``{url: <str>}`` or plain strings. Walk tolerantly instead of pinning one.
    """
    if isinstance(value, str):
        return value if value.startswith("http") else None
    if isinstance(value, list):
        for item in value:
            url = _first_url(item)
            if url:
                return url
        return None
    if isinstance(value, dict):
        for key in ("url", "urls", "video_url", "image_url"):
            if key in value:
                url = _first_url(value[key])
                if url:
                    return url
        return None
    return None


def _poll(task_id: str, kind: str) -> str:
    """Poll until completed and return the first result URL for `kind`
    (images / videos)."""
    deadline = time.monotonic() + _poll_timeout()
    while True:
        obj = _request_json("GET", f"/v1/tasks/{task_id}")
        data = obj.get("data") if isinstance(obj.get("data"), dict) else obj
        status = str(data.get("status") or "").lower()
        if status == "completed":
            result = data.get("result") or {}
            url = _first_url(result.get(kind) if isinstance(result, dict) else None)
            if not url:
                url = _first_url(result)
            if not url:
                raise RuntimeError(
                    f"APIMart task {task_id} completed without a {kind} URL: "
                    f"{str(result)[:300]}"
                )
            return url
        if status in {"failed", "cancelled"}:
            err = data.get("error")
            msg = err.get("message") if isinstance(err, dict) else err
            raise RuntimeError(
                f"APIMart task {task_id} {status}: {msg or 'unknown error'}"
            )
        pct = data.get("progress")
        progress(
            f"APIMart task {status or 'pending'}",
            percent=float(pct) if isinstance(pct, (int, float)) else None,
        )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"APIMart task {task_id} did not finish within "
                f"{int(_poll_timeout())}s (last status: {status})"
            )
        time.sleep(POLL_INTERVAL_S)


def _generate_asset(path: str, body: Dict[str, Any], kind: str, mime: str) -> Asset:
    task_id = _submit(path, body)
    url = _poll(task_id, kind)
    content, got_mime = _download(url)
    return asset(content, mime=got_mime or mime)


# ── Model registry ─────────────────────────────────────────────────────────


def _snap_ratio(
    width: Optional[int], height: Optional[int], ratios: List[str], default: str
) -> str:
    """Snap an explicit width×height to the nearest allowed ratio bucket."""
    if not width or not height or height <= 0:
        return default
    target = width / height

    def value(r: str) -> float:
        w, h = r.split(":")
        return int(w) / int(h)

    numeric = [r for r in ratios if ":" in r]
    if not numeric:
        return default
    return min(numeric, key=lambda r: abs(value(r) - target))


@dataclass
class ImageSpec:
    ratios: List[str]
    default_ratio: str = "1:1"
    resolution: Optional[str] = None  # model's preferred tier, if it has one
    extra: Dict[str, Any] = field(default_factory=dict)

    def build(self, text: str, width: Optional[int], height: Optional[int]) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "prompt": text,
            "size": _snap_ratio(width, height, self.ratios, self.default_ratio),
        }
        if self.resolution:
            body["resolution"] = self.resolution
        body.update(self.extra)
        return body


IMAGE_SPECS: Dict[str, ImageSpec] = {
    "z-image-turbo": ImageSpec(
        ratios=["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3"],
        resolution="1K",
    ),
    "doubao-seedream-4-5": ImageSpec(
        ratios=["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"],
        resolution="2K",
    ),
    "gemini-3-pro-image-preview": ImageSpec(
        ratios=["1:1", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9", "21:9"],
        resolution="1K",
    ),
    "gpt-image-1-official": ImageSpec(ratios=["1:1", "3:2", "2:3"]),
}

# image-edit reuses IMAGE_SPECS; only these models accept image_urls.
IMAGE_EDIT_MODELS = [
    "gemini-3-pro-image-preview",
    "doubao-seedream-4-5",
    "gpt-image-1-official",
]

VideoBuilder = Callable[
    [str, Optional[int], Optional[int], Optional[float], List[str]],
    Dict[str, Any],
]


def _video_kling(
    text: str, w: Optional[int], h: Optional[int], duration: Optional[float], urls: List[str]
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "prompt": text,
        "aspect_ratio": _snap_ratio(w, h, ["16:9", "9:16", "1:1"], "16:9"),
    }
    if duration:
        body["duration"] = max(3, min(15, int(duration)))
    if urls:
        body["image_urls"] = urls[:2]
    return body


def _video_veo3(
    text: str, w: Optional[int], h: Optional[int], duration: Optional[float], urls: List[str]
) -> Dict[str, Any]:
    # VEO3.1 only supports 8s clips; the ABI duration is ignored by design.
    body: Dict[str, Any] = {
        "prompt": text,
        "aspect_ratio": _snap_ratio(w, h, ["16:9", "9:16"], "16:9"),
    }
    if urls:
        body["image_urls"] = urls[:3]
    return body


def _video_sora2(
    text: str, w: Optional[int], h: Optional[int], duration: Optional[float], urls: List[str]
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "prompt": text,
        "aspect_ratio": _snap_ratio(w, h, ["16:9", "9:16"], "16:9"),
    }
    if duration:
        allowed = [4, 8, 12, 16, 20]
        body["duration"] = min(allowed, key=lambda d: abs(d - duration))
    if urls:
        body["image_urls"] = urls[:1]
    return body


def _video_seedance(
    text: str, w: Optional[int], h: Optional[int], duration: Optional[float], urls: List[str]
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "prompt": text,
        "size": _snap_ratio(w, h, ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], "adaptive"),
        "resolution": "720p",
    }
    if duration:
        body["duration"] = max(4, min(15, int(duration)))
    if urls:
        body["image_urls"] = urls[:9]
    return body


VIDEO_BUILDERS: Dict[str, VideoBuilder] = {
    "kling-v3": _video_kling,
    "veo3.1-fast": _video_veo3,
    "sora-2": _video_sora2,
    "doubao-seedance-2.0": _video_seedance,
}

CHAT_MODELS = ["gpt-5", "claude-sonnet-4-6", "gemini-2.5-pro", "deepseek-v4-pro"]
TRANSCRIBE_MODELS = ["whisper-1"]
TTS_MODELS = ["gpt-4o-mini-tts"]
TTS_VOICES = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}


def _active_model(slot: str) -> str:
    models = TONGFLOW_SLOT_MODELS[slot]
    if not _REQUEST_MODEL:
        return models[0]
    if _REQUEST_MODEL not in models:
        raise RuntimeError(
            f"unknown model {_REQUEST_MODEL!r} for {slot}; available: {', '.join(models)}"
        )
    return _REQUEST_MODEL


# ── ABI slot handlers ──────────────────────────────────────────────────────


@node_slot(NodeSlots.IMAGE_GEN)
def image_gen(input: ImageGenInput) -> ImageGenOutput:
    text = (input.text or "").strip()
    if not text:
        return ImageGenOutput(success=False, error="Missing text prompt")
    model = _active_model("image-gen")
    body = IMAGE_SPECS[model].build(text, input.width, input.height)
    body["model"] = model
    image = _generate_asset("/v1/images/generations", body, "images", "image/png")
    return ImageGenOutput(success=True, image=image)


@node_slot(NodeSlots.IMAGE_EDIT)
def image_edit(input: ImageEditInput) -> ImageEditOutput:
    text = (input.text or "").strip()
    if not text:
        return ImageEditOutput(success=False, error="Missing text prompt")
    model = _active_model("image-edit")
    body = IMAGE_SPECS[model].build(text, input.width, input.height)
    body["model"] = model
    body["image_urls"] = [_upload_image(input.image)]
    image = _generate_asset("/v1/images/generations", body, "images", "image/png")
    return ImageEditOutput(success=True, image=image)


@node_slot(NodeSlots.GEN_TEXT)
def gen_text(input: GenTextInput) -> GenTextOutput:
    text = (input.text or "").strip()
    if not text:
        return GenTextOutput(success=False, error="Missing input text")
    model = _active_model("gen-text")
    messages: List[Dict[str, str]] = []
    system = (input.userPrompt or "").strip()
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": text})
    obj = _request_json(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": messages, "stream": False},
    )
    # Standard OpenAI shape, with tolerance for a {data: {...}} wrapper.
    root = obj.get("data") if isinstance(obj.get("data"), dict) else obj
    choices = root.get("choices") or []
    content = ""
    if choices and isinstance(choices[0], dict):
        content = str(((choices[0].get("message") or {}).get("content")) or "")
    if not content.strip():
        return GenTextOutput(
            success=False, error=f"Empty completion from {model}: {str(obj)[:200]}"
        )
    return GenTextOutput(success=True, text=content)


@node_slot(NodeSlots.TEXT_GEN_VIDEO)
def text_gen_video(input: TextGenVideoInput) -> TextGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return TextGenVideoOutput(success=False, error="Missing text prompt")
    model = _active_model("text-gen-video")
    body = VIDEO_BUILDERS[model](text, input.width, input.height, input.duration, [])
    body["model"] = model
    video = _generate_asset("/v1/videos/generations", body, "videos", "video/mp4")
    return TextGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.IMAGE_GEN_VIDEO)
def image_gen_video(input: ImageGenVideoInput) -> ImageGenVideoOutput:
    text = (input.text or "").strip()
    if not text:
        return ImageGenVideoOutput(success=False, error="Missing text prompt")
    model = _active_model("image-gen-video")
    url = _upload_image(input.image)
    body = VIDEO_BUILDERS[model](text, input.width, input.height, input.duration, [url])
    body["model"] = model
    video = _generate_asset("/v1/videos/generations", body, "videos", "video/mp4")
    return ImageGenVideoOutput(success=True, video=video)


@node_slot(NodeSlots.TRANSCRIBE)
def transcribe(input: TranscribeInput) -> TranscribeOutput:
    model = _active_model("transcribe")
    content = base64.b64decode(input.audio.bytesBase64)
    mime = (input.audio.mime or "audio/mpeg").strip() or "audio/mpeg"
    ext = mime.rsplit("/", 1)[-1] or "mp3"
    fields = {"model": model}
    if input.language:
        fields["language"] = input.language
    if input.context:
        fields["prompt"] = input.context
    data, content_type = _multipart(
        fields, [("file", input.audio.filename or f"audio.{ext}", mime, content)]
    )
    raw, _mime = _http("POST", "/v1/audio/transcriptions", data, content_type)
    obj = json.loads(raw.decode("utf-8", errors="replace"))
    text = obj.get("text") if isinstance(obj, dict) else None
    if not isinstance(text, str):
        return TranscribeOutput(
            success=False, error=f"No text in transcription response: {str(obj)[:200]}"
        )
    return TranscribeOutput(success=True, text=text)


@node_slot(NodeSlots.TEXT_GEN_SPEECH_PRESET)
def text_gen_speech_preset(
    input: TextGenSpeechPresetInput,
) -> TextGenSpeechPresetOutput:
    text = (input.text or "").strip()
    if not text:
        return TextGenSpeechPresetOutput(success=False, error="Missing input text")
    model = _active_model("text-gen-speech-preset")
    speaker = (input.speaker or "").strip().lower()
    voice = speaker if speaker in TTS_VOICES else "alloy"
    raw, mime = _http(
        "POST",
        "/v1/audio/speech",
        json.dumps(
            {"model": model, "input": text, "voice": voice, "response_format": "wav"}
        ).encode("utf-8"),
        "application/json",
        timeout=300,
    )
    if mime.startswith("application/json"):
        raise RuntimeError(
            f"TTS returned JSON instead of audio: {raw.decode('utf-8', 'replace')[:200]}"
        )
    return TextGenSpeechPresetOutput(
        success=True, audio=asset(raw, mime=mime or "audio/wav")
    )


# Runtime dispatcher. The @node_slot wrapper accepts a raw dict here (it
# deep-constructs the typed BaseModel internally) and dumps the BaseModel
# return to a dict. `Any` reflects the I/O boundary, not the plugin contract.
_SLOT_HANDLERS: Dict[str, Any] = {
    NodeSlots.IMAGE_GEN: image_gen,
    NodeSlots.IMAGE_EDIT: image_edit,
    NodeSlots.GEN_TEXT: gen_text,
    NodeSlots.TEXT_GEN_VIDEO: text_gen_video,
    NodeSlots.IMAGE_GEN_VIDEO: image_gen_video,
    NodeSlots.TRANSCRIBE: transcribe,
    NodeSlots.TEXT_GEN_SPEECH_PRESET: text_gen_speech_preset,
}


def _check_model_tables() -> None:
    """The literal TONGFLOW_SLOT_MODELS and the spec tables must not drift."""
    expected = {
        "image-gen": list(IMAGE_SPECS),
        "image-edit": IMAGE_EDIT_MODELS,
        "gen-text": CHAT_MODELS,
        "text-gen-video": list(VIDEO_BUILDERS),
        "image-gen-video": list(VIDEO_BUILDERS),
        "transcribe": TRANSCRIBE_MODELS,
        "text-gen-speech-preset": TTS_MODELS,
    }
    if TONGFLOW_SLOT_MODELS != expected:
        raise RuntimeError(
            "TONGFLOW_SLOT_MODELS drifted from the spec tables — keep them in sync"
        )


def _write(out: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    sys.stdout.flush()


def main() -> int:
    global _REQUEST_MODEL
    try:
        _check_model_tables()
        raw = sys.stdin.read()
        req = json.loads(raw) if raw.strip() else {}
        prompt = req.get("prompt") if isinstance(req, dict) else {}
        if not isinstance(prompt, dict):
            prompt = {}
        slot = str(req.get("nodeSlot") or "") if isinstance(req, dict) else ""
        _REQUEST_MODEL = (
            str(req.get("model") or "").strip() if isinstance(req, dict) else ""
        )

        handler = _SLOT_HANDLERS.get(slot)
        if handler is None:
            raise RuntimeError(f"unsupported nodeSlot: {slot!r}")
        out = handler(prompt)
    except Exception as e:  # noqa: BLE001 — surfaced as ABI failure
        _write({"success": False, "error": str(e)})
        return 1

    _write(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
