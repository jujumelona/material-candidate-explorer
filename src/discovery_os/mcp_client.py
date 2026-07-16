"""Bounded MCP Streamable HTTP client for optional evidence retrieval tools.

The endpoint and tool name are configuration-owned. Model output can never
select an arbitrary MCP server or tool. This client implements the stable
2025-11-25 initialize/initialized/tools-call lifecycle and accepts JSON or SSE
responses. Task-augmented and elicitation flows are intentionally unsupported.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping
from urllib.parse import urlparse

import requests


class McpClientError(RuntimeError):
    pass


class _McpSessionExpired(McpClientError):
    pass


class StreamableHttpMcpClient:
    protocol_version = "2025-11-25"
    _tool_name_pattern = re.compile(r"[A-Za-z0-9_.-]{1,128}")

    def __init__(
        self,
        endpoint: str,
        *,
        token: str | None = None,
        timeout: float = 60.0,
        session: requests.Session | None = None,
        allow_loopback_http: bool = False,
        max_response_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        endpoint = endpoint.strip()
        parsed = urlparse(endpoint)
        loopback = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
        if parsed.scheme != "https" and not (
            parsed.scheme == "http" and loopback and allow_loopback_http
        ):
            raise ValueError(
                "MCP endpoint must use HTTPS; loopback HTTP requires explicit opt-in"
            )
        if not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
            raise ValueError("invalid MCP endpoint")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("MCP timeout must be a positive finite number")
        if max_response_bytes <= 0:
            raise ValueError("MCP response size limit must be positive")
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "User-Agent": "discovery-os-mcp-rag/1.0",
            }
        )
        clean_token = token.strip() if token else ""
        if "\r" in clean_token or "\n" in clean_token:
            raise ValueError("MCP bearer token contains invalid characters")
        if clean_token:
            self.session.headers["Authorization"] = f"Bearer {clean_token}"
        self._session_id: str | None = None
        self._initialized = False
        self._next_id = 1

    def initialize(self) -> None:
        if self._initialized:
            return
        self._session_id = None
        request_id = self._id()
        try:
            response = self._post(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": self.protocol_version,
                        "capabilities": {},
                        "clientInfo": {"name": "discovery-os", "version": "0.4.0"},
                    },
                },
                include_protocol_header=False,
                include_session=False,
            )
            result = self._result(response)
            negotiated = str(result.get("protocolVersion", ""))
            if negotiated != self.protocol_version:
                raise McpClientError(
                    f"MCP server negotiated unsupported protocol {negotiated!r}"
                )
            capabilities = result.get("capabilities")
            if not isinstance(capabilities, dict) or not isinstance(
                capabilities.get("tools"), dict
            ):
                raise McpClientError(
                    "MCP server did not declare the required tools capability"
                )
            self._post(
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                notification=True,
            )
        except Exception:
            self._session_id = None
            self._initialized = False
            raise
        self._initialized = True

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        self.validate_tool_name(name)
        for attempt in range(2):
            if not self._initialized:
                self.initialize()
            try:
                response = self._post(
                    {
                        "jsonrpc": "2.0",
                        "id": self._id(),
                        "method": "tools/call",
                        "params": {"name": name, "arguments": dict(arguments)},
                    }
                )
                break
            except _McpSessionExpired:
                if attempt:
                    raise
        else:  # pragma: no cover - the bounded loop always breaks or raises.
            raise McpClientError("MCP tool call could not establish a session")
        result = self._result(response)
        if result.get("isError") is True:
            raise McpClientError("MCP evidence tool returned isError=true")
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        for item in result.get("content", []):
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                try:
                    value = json.loads(item["text"])
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return value
        raise McpClientError("MCP evidence tool did not return a JSON object")

    def close(self) -> None:
        session_id = self._session_id
        try:
            if session_id:
                response = self.session.delete(
                    self.endpoint,
                    headers={
                        "MCP-Protocol-Version": self.protocol_version,
                        "Mcp-Session-Id": session_id,
                    },
                    timeout=self.timeout,
                    allow_redirects=False,
                )
                if response.status_code != 405:
                    response.raise_for_status()
        finally:
            self._session_id = None
            self._initialized = False

    def _id(self) -> int:
        value = self._next_id
        self._next_id += 1
        return value

    def _post(
        self,
        payload: dict[str, Any],
        *,
        notification: bool = False,
        include_protocol_header: bool = True,
        include_session: bool = True,
    ) -> dict[str, Any]:
        headers: dict[str, str] = {}
        if include_protocol_header:
            headers["MCP-Protocol-Version"] = self.protocol_version
        if include_session and self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        response = self.session.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
            stream=True,
            allow_redirects=False,
        )
        try:
            if response.status_code == 404 and include_session and self._session_id:
                self._session_id = None
                self._initialized = False
                raise _McpSessionExpired("MCP session expired")
            if 300 <= response.status_code < 400:
                raise McpClientError(
                    "MCP endpoint redirects are refused; configure the final endpoint URL"
                )
            response.raise_for_status()
            session_id = response.headers.get("Mcp-Session-Id")
            if session_id:
                if any(ord(char) < 0x21 or ord(char) > 0x7E for char in session_id):
                    raise McpClientError("MCP server returned an invalid session id")
                if self._session_id and session_id != self._session_id:
                    raise McpClientError("MCP server changed the active session id")
                if not include_session and payload.get("method") != "initialize":
                    raise McpClientError(
                        "MCP server returned a session id outside initialization"
                    )
                self._session_id = session_id
            if notification:
                if response.status_code != 202:
                    raise McpClientError(
                        "MCP server did not acknowledge a notification with HTTP 202"
                    )
                return {}
            if response.status_code == 202:
                raise McpClientError(
                    "MCP server acknowledged a request without a JSON-RPC response"
                )
            body = self._read_bounded(response)
            if not body:
                raise McpClientError("MCP request returned an empty response")
            content_type = response.headers.get("Content-Type", "").lower()
            expected_id = payload.get("id")
            if "text/event-stream" in content_type:
                value = self._parse_sse(body.decode("utf-8"), expected_id=expected_id)
                return self._validate_response(value, expected_id=expected_id)
            try:
                value = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise McpClientError("MCP response is not valid JSON") from exc
            if not isinstance(value, dict):
                raise McpClientError("MCP response must be a JSON object")
            return self._validate_response(value, expected_id=expected_id)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    @classmethod
    def validate_tool_name(cls, name: str) -> None:
        if cls._tool_name_pattern.fullmatch(name) is None:
            raise ValueError(
                "MCP tool name must use 1-128 ASCII letters, digits, dots, "
                "hyphens, or underscores"
            )

    def _read_bounded(self, response: requests.Response) -> bytes:
        declared_length = response.headers.get("Content-Length")
        if declared_length:
            try:
                if int(declared_length) > self.max_response_bytes:
                    raise McpClientError(
                        "MCP response exceeds the configured size limit"
                    )
            except ValueError:
                pass
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > self.max_response_bytes:
                raise McpClientError("MCP response exceeds the configured size limit")
            chunks.append(chunk)
        return b"".join(chunks)

    @staticmethod
    def _parse_sse(text: str, *, expected_id: Any) -> dict[str, Any]:
        values: list[dict[str, Any]] = []
        data_lines: list[str] = []

        def finish_event() -> None:
            if not data_lines:
                return
            raw = "\n".join(data_lines)
            data_lines.clear()
            if not raw.strip():
                return
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise McpClientError("MCP SSE event contains invalid JSON") from exc
            if isinstance(item, dict):
                values.append(item)

        for line in text.splitlines():
            if not line:
                finish_event()
                continue
            if line.startswith(":"):
                continue
            if line == "data":
                data_lines.append("")
            elif line.startswith("data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(" ") else value)
        finish_event()
        if not values:
            raise McpClientError("MCP SSE response contained no JSON-RPC message")
        for value in values:
            if value.get("id") == expected_id:
                return value
        raise McpClientError(
            "MCP SSE response did not contain the matching JSON-RPC response"
        )

    @staticmethod
    def _validate_response(
        response: dict[str, Any], *, expected_id: Any
    ) -> dict[str, Any]:
        if response.get("jsonrpc") != "2.0":
            raise McpClientError("MCP response is not JSON-RPC 2.0")
        if response.get("id") != expected_id:
            raise McpClientError("MCP response id does not match the request id")
        return response

    @staticmethod
    def _result(response: Mapping[str, Any]) -> dict[str, Any]:
        if isinstance(response.get("error"), dict):
            error = response["error"]
            raise McpClientError(
                f"MCP error {error.get('code')}: {error.get('message')}"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise McpClientError("MCP response is missing a result object")
        return result


__all__ = ["McpClientError", "StreamableHttpMcpClient"]
