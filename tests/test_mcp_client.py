from datetime import date
import json

import pytest

from discovery_os.literature_rag import (
    build_literature_rag_from_environment,
    LiteratureQuery,
    LiteratureRagError,
    LiteratureSource,
    MultiSourceLiteratureRetriever,
)
from discovery_os.mcp_client import McpClientError, StreamableHttpMcpClient


class Response:
    def __init__(self, payload=None, *, status=200, headers=None):
        self._payload = payload
        self.status_code = status
        self.headers = headers or {"Content-Type": "application/json"}
        self.content = b"" if payload is None else json.dumps(payload).encode()
        self.text = self.content.decode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=65536):
        if self.content:
            yield self.content

    def close(self):
        pass


class Session:
    def __init__(self):
        self.headers = {}
        self.calls = []
        self.responses = [
            Response(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "fixture", "version": "1"},
                    },
                },
                headers={
                    "Content-Type": "application/json",
                    "Mcp-Session-Id": "session-1",
                },
            ),
            Response(status=202),
            Response(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "result": {
                        "structuredContent": {
                            "records": [
                                {
                                    "source_id": "doc-1",
                                    "title": "Li O stability",
                                    "abstract": "Measured stable phase",
                                }
                            ]
                        },
                        "content": [],
                    },
                }
            ),
        ]

    def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return self.responses.pop(0)

    def delete(self, *args, **kwargs):
        return Response(status=204)


def test_streamable_http_mcp_initializes_and_calls_configured_tool():
    session = Session()
    client = StreamableHttpMcpClient("https://mcp.example/evidence", session=session)
    result = client.call_tool("search_materials", {"query": "Li O"})
    assert result["records"][0]["source_id"] == "doc-1"
    assert session.calls[2][1]["headers"]["Mcp-Session-Id"] == "session-1"
    assert session.calls[2][1]["json"]["params"]["name"] == "search_materials"
    assert session.calls[2][1]["allow_redirects"] is False


def test_stateless_mcp_server_is_initialized_only_once():
    session = Session()
    session.responses[0].headers.pop("Mcp-Session-Id")
    session.responses.append(
        Response(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "result": {"structuredContent": {"records": []}, "content": []},
            }
        )
    )
    client = StreamableHttpMcpClient("https://mcp.example/evidence", session=session)
    client.call_tool("search_materials", {"query": "Li O"})
    client.call_tool("search_materials", {"query": "Na O"})
    methods = [call[1]["json"].get("method") for call in session.calls]
    assert methods == [
        "initialize",
        "notifications/initialized",
        "tools/call",
        "tools/call",
    ]


def test_expired_session_is_reinitialized_once_without_the_old_session_header():
    session = Session()
    session.responses = [
        session.responses[0],
        session.responses[1],
        Response(status=404),
        Response(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "result": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fixture", "version": "1"},
                },
            },
            headers={"Content-Type": "application/json", "Mcp-Session-Id": "session-2"},
        ),
        Response(status=202),
        Response(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "result": {"structuredContent": {"records": []}, "content": []},
            }
        ),
    ]
    client = StreamableHttpMcpClient("https://mcp.example/evidence", session=session)
    assert client.call_tool("search_materials", {}) == {"records": []}
    assert "Mcp-Session-Id" not in session.calls[3][1]["headers"]
    assert session.calls[5][1]["headers"]["Mcp-Session-Id"] == "session-2"


def test_mcp_redirect_is_refused_without_following_it():
    session = Session()
    session.responses = [
        Response(status=307, headers={"Location": "https://other.example/mcp"})
    ]
    client = StreamableHttpMcpClient("https://mcp.example/evidence", session=session)
    with pytest.raises(McpClientError, match="redirects are refused"):
        client.initialize()
    assert session.calls[0][1]["allow_redirects"] is False


def test_sse_parser_selects_matching_response_and_supports_multiline_data():
    text = "\n".join(
        [
            'data: {"jsonrpc":"2.0","method":"notifications/progress"}',
            "",
            'data: {"jsonrpc":"2.0","id":7,',
            'data: "result":{"content":[]}}',
            "",
        ]
    )
    value = StreamableHttpMcpClient._parse_sse(text, expected_id=7)
    assert value["id"] == 7


def test_mcp_tool_name_rejects_unbounded_or_special_character_names():
    client = StreamableHttpMcpClient("https://mcp.example/evidence", session=Session())
    with pytest.raises(ValueError):
        client.call_tool("search/materials", {})


def test_mcp_literature_records_are_strict_and_source_grounded():
    class Client:
        endpoint = "https://mcp.example/evidence"

        def call_tool(self, name, arguments):
            return {
                "records": [
                    {
                        "source_id": "doc-1",
                        "title": "Li O stability",
                        "abstract": "Measured stable phase",
                        "publication_date": "2025-01-02",
                        "url": "https://example.test/doc-1",
                    }
                ]
            }

    retriever = MultiSourceLiteratureRetriever(
        mcp_client=Client(), mcp_tool="search_materials"
    )
    records = retriever._search_mcp(
        LiteratureQuery(
            query_id="mcp-query",
            source=LiteratureSource.MCP,
            query="Li O stability",
            rationale="fixture",
            from_date=date(2024, 1, 1),
        )
    )
    assert records[0].source_ids == {"mcp": "doc-1"}
    assert records[0].source_queries == ["mcp-query"]


def test_mcp_literature_rejects_null_required_fields():
    class Client:
        endpoint = "https://mcp.example/evidence"

        def call_tool(self, name, arguments):
            return {"records": [{"source_id": "doc-1", "title": None}]}

    retriever = MultiSourceLiteratureRetriever(
        mcp_client=Client(), mcp_tool="search_materials"
    )
    with pytest.raises(LiteratureRagError, match="invalid record"):
        retriever._search_mcp(
            LiteratureQuery(
                query_id="mcp-query",
                source=LiteratureSource.MCP,
                query="Li O stability",
                rationale="fixture",
            )
        )


def test_mcp_environment_configuration_is_paired_and_fail_fast():
    with pytest.raises(LiteratureRagError, match="must be configured together"):
        build_literature_rag_from_environment(
            environ={"MATERIAL_RAG_MCP_URL": "https://mcp.example/evidence"}
        )
    with pytest.raises(LiteratureRagError, match="Invalid MCP RAG configuration"):
        build_literature_rag_from_environment(
            environ={
                "MATERIAL_RAG_MCP_URL": "https://mcp.example/evidence",
                "MATERIAL_RAG_MCP_TOOL": "invalid/tool",
            }
        )
    pipeline = build_literature_rag_from_environment(
        environ={
            "MATERIAL_RAG_MCP_URL": "https://mcp.example/evidence",
            "MATERIAL_RAG_MCP_TOOL": "search_materials",
            "MATERIAL_RAG_MCP_TIMEOUT_SECONDS": "15",
        }
    )
    assert pipeline.retriever.mcp_tool == "search_materials"
    assert pipeline.retriever.mcp_client.timeout == 15.0
