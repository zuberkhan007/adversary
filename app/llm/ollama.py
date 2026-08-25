"""Ollama management helpers: server health, model listing, model pulling.

These use Ollama's native REST API (not the OpenAI-compatible layer) so they
work against any Ollama install. Pulling supports any free model in the
Ollama registry (https://ollama.com/library).
"""

from __future__ import annotations

from typing import Callable, Iterator

import requests


PullCallback = Callable[[dict], None]


def is_server_up(base_url: str, timeout: float = 3.0, api_key: str | None = None) -> bool:
    """Return True if the Ollama HTTP server is reachable."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", headers=headers, timeout=timeout)
        return r.status_code == 200
    except (requests.RequestException, OSError):
        # ``OSError`` covers low-level socket/SSL errors (e.g.
        # ``[Errno 22] Invalid argument`` on Windows) that ``requests``
        # doesn't always wrap in ``RequestException``.
        return False


def list_models(base_url: str, timeout: float = 10.0, api_key: str | None = None) -> list[str]:
    """List installed model names. Returns [] if the server is unreachable."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except (requests.RequestException, OSError, ValueError, KeyError):
        return []


def model_exists(base_url: str, model: str, timeout: float = 10.0, api_key: str | None = None) -> bool:
    return model in list_models(base_url, timeout=timeout, api_key=api_key)


def pull_model(base_url: str, model: str, callback: PullCallback | None = None, timeout: float = 600.0, api_key: str | None = None) -> Iterator[dict]:
    """Stream pull progress for any free Ollama registry model.

    Yields each JSON status line from ``/api/pull``. Pass a ``callback`` to
    receive every status dict as it arrives. Raises ``RuntimeError`` on HTTP
    failure before the stream starts.
    """
    url = f"{base_url.rstrip('/')}/api/pull"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        with requests.post(url, json={"name": model, "stream": True}, headers=headers, timeout=timeout, stream=True) as r:
            if r.status_code != 200:
                try:
                    err = r.json()
                except ValueError:
                    err = {"error": r.text}
                raise RuntimeError(f"Ollama pull failed ({r.status_code}): {err}")
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                import json as _json
                try:
                    evt = _json.loads(line)
                except ValueError:
                    continue
                if callback is not None:
                    callback(evt)
                yield evt
                if evt.get("status") == "success":
                    return
    except (requests.RequestException, OSError) as exc:
        raise RuntimeError(f"Ollama pull request failed: {exc}") from exc


def parse_pull_progress(evt: dict) -> tuple[str, float | None]:
    """Extract a (status_label, fraction_in_0..1) pair from a pull event.

    Returns ``(label, None)`` when no byte-progress is available.
    """
    status = str(evt.get("status") or "")
    completed = evt.get("completed")
    total = evt.get("total")
    if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total > 0:
        return status, min(1.0, float(completed) / float(total))
    return status, None