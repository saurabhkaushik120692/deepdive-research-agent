from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Every LLM / embedding / moderation call must go through Mesh (hackathon rule).
MESH_BASE_URL = "https://api.meshapi.ai/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    mesh_api_key: str = ""
    mesh_base_url: str = MESH_BASE_URL
    tavily_api_key: str = ""

    data_dir: Path = Path("data")
    database_url: str = "sqlite:///data/deepdive.db"

    # One model per agent role; the verifier must come from a different family than the drafter.
    model_planner: str = "openai/gpt-5-nano"
    model_drafter: str = "openai/gpt-5-nano"
    model_synthesizer: str = "openai/gpt-5-nano"
    model_contradictions: str = "openai/gpt-5-nano"
    model_verifier: str = "anthropic/claude-haiku-5.5"
    model_judge: str = "google/gemini-2.5-flash-lite"
    model_embed: str = "google/embeddinggemma-300m"
    model_moderation: str = "openai/omni-moderation-latest"
    # Follow-up questions under a report: an on-topic check, drafting and the answer.
    model_scope: str = "openai/gpt-5-nano"
    model_followup_drafter: str = "openai/gpt-5-nano"
    model_followup_answer: str = "openai/gpt-5-nano"

    min_supported_claims: int = 2
    max_extra_rounds: int = 2
    quote_match_threshold: int = 90

    search_results_per_query: int = 5
    chunk_words: int = 600
    chunk_overlap_words: int = 75
    chunk_max_chars: int = 3000
    embed_max_chars: int = 2400  # most of a chunk; model input limit is 2048 tokens
    embed_fallback_chars: int = 1000  # retry size when unusually dense text is still too long
    max_chunks_per_source: int = 40
    top_k_chunks: int = 6

    llm_max_retries: int = 5
    llm_timeout_retries: int = 1  # a timed-out call already took llm_timeout_s
    llm_rpm: int = Field(default=80, gt=0)  # Mesh allows 100 requests/minute per account
    llm_concurrency: int = 8
    llm_timeout_s: float = 180.0  # reasoning models (gpt-5-nano) often need 60-120 s per call
    # At the default effort the report writer thinks ~7,000 hidden tokens (about 2 min, sometimes
    # past the timeout); "low" writes comparable reports in 20-45 s. Empty = model default.
    reasoning_effort_synthesizer: str = "low"
    # Follow-ups should answer in about a minute; the verifier keeps its default effort.
    reasoning_effort_scope: str = "low"
    reasoning_effort_followup_drafter: str = "low"
    reasoning_effort_followup_answer: str = "low"

    # Google sign-in. site_host is the public hostname (the deploy sets it for Caddy too):
    # when set, sign-in returns to https://<site_host>/ and the session cookie is HTTPS-only.
    google_client_id: str = ""
    google_client_secret: str = ""
    session_secret: str = ""  # empty = a random key per process (sign-ins end on restart)
    site_host: str = ""
    contact_email: str = ""  # shown on the Privacy and Terms pages

    # Reports each Google account may create; failed runs don't count. Accounts listed in
    # unlimited_emails (comma-separated) have no limit.
    free_reports_per_user: int = 5
    unlimited_emails: str = ""

    # Follow-up questions: only on the owner's own reports, closed after followup_max_age_days.
    followups_per_report: int = 10
    followup_max_age_days: int = 30

    runs_per_hour: int = 10
    moderation_timeout_s: float = 10.0  # the safety screen must never stall run creation
    max_concurrent_runs: int = 2

    @model_validator(mode="after")
    def _verifier_family_differs(self) -> "Settings":
        verifier_family = self.model_verifier.split("/")[0]
        for name in ("model_drafter", "model_followup_drafter"):
            drafter = getattr(self, name)
            if drafter.split("/")[0] == verifier_family:
                raise ValueError(
                    f"model_verifier must be from a different model family than {name} "
                    f"(got {drafter!r} and {self.model_verifier!r})"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
