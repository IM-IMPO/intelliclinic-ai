"""Environment-driven configuration. No secret is ever hardcoded or shipped.

Every value here is read from the environment (or a ``.env`` file). A missing
value falls back to a development default, and the fallback is deliberately
labelled as such by :func:`settings_report` so the UI cannot mistake a stub for
a production capability.
"""

from __future__ import annotations

import os
import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from .logging_util import get_logger

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- application -----------------------------------------------------
    app_name: str = "IntelliClinic AI"
    app_env: Literal["development", "staging", "production"] = "development"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    log_level: str = "INFO"
    request_timeout_s: float = 30.0
    retention_days: int = 365

    # ---- transport / secrets --------------------------------------------
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    session_secret: str = "dev-insecure-session-secret-change-me"
    session_ttl_minutes: int = 480
    argon2_time_cost: int = 3
    rate_limit_per_minute: int = 120

    @field_validator("session_secret")
    @classmethod
    def _prod_needs_secret(cls, v: str, info: Any) -> str:
        if info.data.get("app_env") == "production" and v.startswith("dev-insecure"):
            raise ValueError(
                "SESSION_SECRET must be overridden in production "
                "(generate one with: python -c \"import secrets;print(secrets.token_urlsafe(48))\")"
            )
        return v

    # ---- application database (jobs, docs metadata, review state) --------
    # Deliberately NOT OpenMRS's MySQL - see ARCHITECTURE.md.
    database_url: str = "sqlite+aiosqlite:///./data/intelliclinic.db"

    # ---- Algorithm 1: intent classification ------------------------------
    intent_model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    intent_threshold: float = 0.72
    intent_low_confidence_threshold: float = 0.55
    #: A message routes when the top score clears the threshold OR the margin
    #: over the runner-up clears this. See
    #: :meth:`IntentClassifier._decide` for why both are needed.
    intent_margin: float = 0.25
    #: Below this the two intents are effectively tied and the message is
    #: retained for clarification even if the absolute score looks acceptable.
    intent_low_margin: float = 0.10
    intent_backend: Literal["real", "stub"] = "real"

    # ---- Algorithm 2: clinical NER ---------------------------------------
    ner_model_name: str = "samrawal/bert-base-uncased_clinical-ner"
    ner_confidence_floor: float = 0.45
    ner_backend: Literal["real", "stub"] = "real"

    # ---- Algorithm 3: clinical validation --------------------------------
    # Two bands per vital, and they are evaluated independently - this matters.
    #
    # "plausible" = a value a living patient can actually have. Outside it the
    # number is almost certainly a transcription or unit error, so it is
    # reported as a PLAUSIBILITY_WARNING. The value is never corrected.
    #
    # "urgent" = a value that is physically possible but immediately
    # life-threatening (hyperpyrexia, hypoxaemia, hypotension). Evaluating this
    # band *inside* the plausible band is deliberate: 41 C is a real reading and
    # a medical emergency, and it must not pass merely because 41 < 45.
    #
    # These are generic ADULT defaults for a general clinic. They are not
    # paediatric, obstetric or ICU ranges and must be re-configured per site.
    temp_c_min: float = 34.0
    temp_c_max: float = 42.0
    temp_urgent_high: float = 40.5
    temp_urgent_low: float = 35.0
    bp_sys_min: float = 60.0
    bp_sys_max: float = 250.0
    bp_sys_urgent_low: float = 90.0
    bp_sys_urgent_high: float = 180.0
    bp_dia_min: float = 30.0
    bp_dia_max: float = 150.0
    bp_dia_urgent_low: float = 60.0
    bp_dia_urgent_high: float = 110.0
    pulse_min: float = 25.0
    pulse_max: float = 220.0
    pulse_urgent_low: float = 50.0
    pulse_urgent_high: float = 130.0
    rr_min: float = 6.0
    rr_max: float = 60.0
    rr_urgent_low: float = 10.0
    rr_urgent_high: float = 30.0
    spo2_min: float = 70.0
    spo2_max: float = 100.0
    spo2_urgent_low: float = 90.0

    # ---- Algorithm 4: ChromaDB retrieval ---------------------------------
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "clinical_guidelines"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_min_score: float = 0.28
    retrieval_conflict_gap: float = 0.08
    ingestion_concurrency: int = 2

    # ---- Algorithm 5: grounded answering ---------------------------------
    llm_provider: Literal["ollama", "stub"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    llm_max_context_tokens: int = 8000
    # Generation is far slower than a health check: on CPU Ollama can take
    # tens of seconds per answer, so this is a separate, much longer timeout.
    llm_timeout_s: float = 240.0
    # Disable the reasoning pass. qwen3:8b is a thinking model: with thinking
    # left on it writes its output to the `thinking` field and returns an empty
    # `response`, which reads as "generation failed" even though the model
    # answered. For a grounded, cited clinical answer we want the answer text,
    # not a reasoning trace.
    llm_disable_thinking: bool = True

    # ---- OpenMRS --------------------------------------------------------
    openmrs_base_url: str | None = None
    openmrs_username: str | None = None
    openmrs_password: str | None = None
    openmrs_concept_map: str = "./config/concepts.json"
    openmrs_timeout_s: float = 20.0

    # ---- audit ----------------------------------------------------------
    redact_patient_in_logs: bool = True

    # ---- derived --------------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def openmrs_configured(self) -> bool:
        return bool(
            self.openmrs_base_url and self.openmrs_username and self.openmrs_password
        )

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        return p if p.is_absolute() else BACKEND_ROOT / p

    @property
    def data_dir(self) -> Path:
        return BACKEND_ROOT / "data"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def settings_report(s: Settings | None = None) -> dict[str, Any]:
    """Capability map surfaced by ``GET /api/v1/system/status``.

    This is what lets the UI say "simulated" instead of implying a capability
    exists when it does not.

    Backend fields come from the *live* services that actually serve requests
    (the same singletons :mod:`app.main` builds), never from the configuration
    defaults. The two can disagree, and the live value is the true one: a service
    that fell back to a stub still has ``intent_backend="real"`` in its config,
    so reporting config would claim a capability this process does not have.

    If the app package cannot be imported - a unit test importing
    ``app.config`` on its own, for instance - the services are probed directly
    instead. Probing constructs its own instances, which is slower and therefore
    only the fallback.
    """

    s = s or get_settings()
    try:
        from . import main as app_main

        return _report_from(s, app_main)
    except Exception as exc:  # pragma: no cover - import-order fallback
        get_logger("config").warning(f"[config] reporting from probe instances: {exc}")
        return _report_from(s, None)


def _report_from(s: Settings, app_main: Any) -> dict[str, Any]:
    """Build the report from live services (``app_main``) or fresh probes."""

    def row(key: str) -> dict[str, Any]:
        return _algo_row(s, key, app_main)

    return {
        "algorithms": {
            "intent": row("intent"),
            "extraction": row("extraction"),
            "validation": {
                "backend": "real",
                "configured": True,
                "note": (
                    "Deterministic rules. Reproducible and auditable by "
                    "construction - no model, no variance."
                ),
            },
            "retrieval": row("retrieval"),
            "answering": row("answering"),
        },
        "integrations": {
            "openmrs": {"configured": s.openmrs_configured},
            "chroma_persist_dir": str(s.chroma_path),
        },
        "environment": {
            "app_env": s.app_env,
            "version": s.app_version,
            "rate_limit_per_minute": s.rate_limit_per_minute,
        },
    }


#: Config key that names each service's model, the service class, and the
#: instance attribute holding the live backend string. Mirrors the singletons in
#: ``app.main`` - a divergence here would make the report lie, so the probe path
#: constructs the same classes the app constructs.
_SERVICE_SPECS: dict[str, tuple[str, str]] = {
    "intent": ("intent_model_name", "IntentClassifier"),
    "extraction": ("ner_model_name", "EntityExtractor"),
    "retrieval": ("embedding_model", "RetrievalEngine"),
    "answering": ("llm_model", "GroundedAnswerer"),
}

#: Maps a report key to the attribute name on ``app.main`` holding that service.
_MAIN_ATTRS: dict[str, str] = {
    "intent": "intent_svc",
    "extraction": "extraction_svc",
    "retrieval": "retrieval_svc",
    "answering": "answering_svc",
}#: Only notes that add something the backend field does not already say. A note
#: that contradicts the row (or merely restates it) is worse than no note.
_SERVICE_NOTES: dict[str, dict[str, str]] = {
    "intent": {
        "stub": "No trained classifier loaded; messages route on keyword rules.",
    },
    "extraction": {
        "stub": "The Bio_ClinicalBERT checkpoint did not load (torch missing or "
        "download blocked); a deterministic rule extractor is running instead.",
    },
    "retrieval": {
        "unavailable": "ChromaDB is not importable; retrieval is disabled.",
    },
    "answering": {
        "stub": "No Ollama server reachable; answers are not generated.",
    },
}

_SERVICE_CACHE: dict[str, Any] = {}


def _algo_row(s: Settings, key: str, app_main: Any) -> dict[str, Any]:
    """Describe one algorithm from a running service, never from config alone."""

    model_key, cls_name = _SERVICE_SPECS[key]
    svc = _service(key, cls_name, app_main)
    if svc is None:
        return {
            "backend": "unavailable",
            "configured": False,
            "model": getattr(s, model_key),
            "note": "Service could not be initialised in this process.",
        }

    backend = getattr(svc, "backend", "unavailable")
    return {
        "backend": backend,
        # "sklearn" is a real trained classifier, not a stub: it is the strongest
        # backend available here and the only one that is not a placeholder.
        "configured": backend not in ("stub", "unavailable"),
        "model": getattr(svc, "_model_name", None) or getattr(s, model_key),
        **(
            {"note": _SERVICE_NOTES.get(key, {})[backend]}
            if backend in _SERVICE_NOTES.get(key, {})
            else {}
        ),
    }


def _service(key: str, cls_name: str, app_main: Any) -> Any:
    """The live service instance: ``app.main``'s singleton, else a fresh probe.

    Importing ``app.main`` is expensive (it constructs every model). The report
    is only read from a request handler, by which point ``app.main`` is already
    in ``sys.modules``, so this costs nothing in the real app path.
    """

    if app_main is not None:
        return getattr(app_main, _MAIN_ATTRS[key])

    if key in _SERVICE_CACHE:
        return _SERVICE_CACHE[key]
    try:
        svc = getattr(importlib.import_module(f"app.algorithms.{key}"), cls_name)()
    except Exception as exc:  # pragma: no cover - defensive
        get_logger("config").warning(f"[config] {key} service unavailable: {exc}")
        svc = None
    _SERVICE_CACHE[key] = svc
    return svc


def _model_present(model_name: str) -> bool:
    """True when ``model_name`` resolves to a local HF cache entry.

    Retained for callers that need to know whether a checkpoint is *downloadable*
    offline. It is deliberately not used to answer "is this capability real" -
    that answer comes from the live service, because a checkpoint can be cached
    and still fail to load.
    """

    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = cache / "hub"
    if not hub.exists():
        return False
    return any(hub.glob(f"models--{model_name.replace('/', '--')}*"))
