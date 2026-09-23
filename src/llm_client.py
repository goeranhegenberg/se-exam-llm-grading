"""Modellanbindung (OpenAI) sowie ein deterministischer Mock-Client.

Der ``GradingClient`` kapselt die OpenAI-Chat-Completions-API inklusive
JSON-Modus, Wiederholungslogik (Backoff) und Token-Erfassung. Der
``MockClient`` erzeugt reproduzierbare Pseudo-Bewertungen ohne Netzwerkzugriff
und dient ausschließlich dem Testen der Pipeline -- seine Werte dürfen NICHT in
die Arbeit übernommen werden.
"""
import hashlib
import json

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

class _BadResponse(Exception):
    """Antwort ohne brauchbaren Inhalt (z.B. OpenRouter/Gemini liefern bei einem
    Upstream-Hiccup gelegentlich ``finish_reason='error'`` mit abgeschnittenem
    oder leerem Inhalt). Wird wie ein transienter Fehler behandelt und erneut
    versucht -- sonst gingen einzelne Aufrufe als nicht-parsbar verloren."""


_RETRYABLE = (RateLimitError, APITimeoutError, APIConnectionError,
              InternalServerError, _BadResponse)


class GradingClient:
    def __init__(self, api_key, base_url=None, extra_body=None):
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        # Zusätzliche, nicht-OpenAI-Standard-Felder im Request-Body (z.B. bei
        # OpenRouter ``{"reasoning": {"enabled": false}}``, um das modell-native
        # "Thinking" abzuschalten -- für die Konsistenz-/Determinismus-Studie).
        self.extra_body = extra_body or None

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=40),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    def _call(self, model, system, user, temperature, top_p, max_tokens, seed,
              json_mode, extra_body=None):
        kwargs = dict(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if seed is not None:
            kwargs["seed"] = seed
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # Pro-Aufruf-Override (z.B. reasoning an/aus je Prompt-Version) hat Vorrang
        # vor einem ggf. global gesetzten extra_body.
        eb = extra_body if extra_body is not None else self.extra_body
        if eb:
            kwargs["extra_body"] = eb
        resp = self.client.chat.completions.create(**kwargs)
        choice = resp.choices[0] if resp.choices else None
        content = choice.message.content if (choice and choice.message) else None
        # Transiente Upstream-Fehler (finish_reason='error') oder leere Antworten
        # erneut versuchen statt sie als nicht-parsbar zu zählen. 'length'
        # (Truncation) bleibt absichtlich ausgenommen -- ein Retry brächte dasselbe.
        if choice is None or choice.finish_reason == "error" or not (content and content.strip()):
            reason = choice.finish_reason if choice else "no-choice"
            raise _BadResponse(f"Unbrauchbare Antwort (finish_reason={reason})")
        return resp

    def grade(self, model, system, user, temperature=0.0, top_p=1.0,
              max_tokens=900, seed=None, json_mode=True, extra_body=None,
              **_ignored):
        resp = self._call(model, system, user, temperature, top_p, max_tokens,
                          seed, json_mode, extra_body=extra_body)
        choice = resp.choices[0]
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
            # Reasoning-Tokens (falls Thinking aktiv) best-effort miterfassen.
            details = getattr(resp.usage, "completion_tokens_details", None)
            rt = getattr(details, "reasoning_tokens", None) if details else None
            if rt is not None:
                usage["reasoning_tokens"] = rt
        return {
            "text": choice.message.content,
            "usage": usage,
            "model": resp.model,
            "finish_reason": choice.finish_reason,
        }


class MockClient:
    """Deterministische Pseudo-Bewertungen ohne API. Nur für Offline-Tests!"""

    def __init__(self, noise=1):
        self.noise = noise

    def grade(self, model, system, user, temperature=0.0, ground_truth=None,
              max_points=None, **_ignored):
        gt = int(ground_truth) if ground_truth is not None else 0
        # Reproduzierbares "Rauschen" aus dem Prompt-Hash; bei temperature>0 wird
        # zusätzlich variiert, um Mehrfach-Sampling (Self-Consistency) zu testen.
        h = int(hashlib.sha1((user + f"|{temperature}").encode("utf-8")).hexdigest(), 16)
        delta = (h % (2 * self.noise + 1)) - self.noise if self.noise else 0
        pred = gt + delta
        if max_points is not None:
            pred = max(0, min(int(max_points), pred))
        payload = {
            "punkte": pred,
            "punkte_gesamt": pred,
            "begruendung": "MOCK -- kein echter Modellaufruf.",
        }
        return {
            "text": json.dumps(payload, ensure_ascii=False),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model": f"mock::{model}",
            "finish_reason": "stop",
        }
