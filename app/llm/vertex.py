from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from google.auth import default as google_auth_default
from google.auth.credentials import Credentials
from google.auth.transport.requests import Request
from google.oauth2 import service_account
import httpx

from app.llm.openai import OpenAIProvider


DEFAULT_VERTEX_AI_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
VERTEX_AI_REGION_PATTERN = re.compile(r"^[a-z]+(?:-[a-z]+)*\d$")


class VertexAIProvider(OpenAIProvider):
    def __init__(
        self,
        project: str,
        location: str = "global",
        credentials_path: str | Path | None = None,
        default_model: str = "google/gemini-2.5-flash",
        scopes: Sequence[str] | None = None,
        credentials: Credentials | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not project or not project.strip():
            raise ValueError("Vertex AI project is required")
        if not location or not location.strip():
            raise ValueError("Vertex AI location is required")

        self.project = project.strip()
        self.location = location.strip()
        if self.location != "global" and not VERTEX_AI_REGION_PATTERN.fullmatch(self.location):
            raise ValueError(
                "Vertex AI location must be 'global' or a full region id "
                "such as 'us-central1' or 'europe-west1'"
            )

        self.credentials_path = str(credentials_path) if credentials_path else None
        self.scopes = tuple(scopes or (DEFAULT_VERTEX_AI_SCOPE,))
        self._credentials = credentials

        base_url = self._base_url()
        super().__init__(
            api_key="vertex-ai-token-loaded-per-request",
            base_url=base_url,
            default_model=default_model,
            transport=transport,
        )

    @property
    def provider_name(self) -> str:
        return "vertex-ai"

    def _base_url(self) -> str:
        host = (
            "aiplatform.googleapis.com"
            if self.location == "global"
            else f"{self.location}-aiplatform.googleapis.com"
        )
        return f"https://{host}/v1/projects/{self.project}/locations/{self.location}/endpoints/openapi"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._access_token()}",
            "Content-Type": "application/json",
        }

    def _access_token(self) -> str:
        credentials = self._load_credentials()
        if not credentials.valid:
            credentials.refresh(Request())

        token = credentials.token
        if not token:
            raise RuntimeError("Vertex AI credentials did not provide an access token")
        return token

    def _load_credentials(self) -> Credentials:
        if self._credentials is not None:
            return self._credentials

        if self.credentials_path:
            self._credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=self.scopes,
            )
            return self._credentials

        self._credentials, _ = google_auth_default(scopes=self.scopes)
        return self._credentials
