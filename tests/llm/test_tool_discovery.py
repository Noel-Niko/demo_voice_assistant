"""
Unit tests for ToolDiscoveryService.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Now import normally
from llm.tool_discovery import DiscoveredTools, ToolDiscoveryService, transform_mcp_to_openai_format  # noqa: E402, I001


@pytest.fixture
def mock_discovery_response():
    """Mock response from /tools/discovery/all endpoint."""
    return {
        "servers": [
            {
                "name": "product_retrieval",
                "endpoint": "/product_retrieval/mcp",
                "description": "Product search and retrieval tools",
            },
            {
                "name": "parse_query",
                "endpoint": "/parse_query/mcp",
                "description": "Query parsing tools",
            },
            {
                "name": "assortment_api",
                "endpoint": "/assortment_api/mcp",
                "description": "Category and assortment tools",
            },
        ],
        "tools": [
            {
                "name": "get_product_docs",
                "description": "Search for products using semantic search",
                "server": "product_retrieval",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language search query"},
                        "vector_store": {
                            "type": "string",
                            "enum": ["Product"],
                            "description": "Vector store to search",
                        },
                    },
                    "required": ["query", "vector_store"],
                },
            },
            {
                "name": "parse_customer_query",
                "description": "Parse customer query to extract entities",
                "server": "parse_query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Customer query to parse"}
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_category_description",
                "description": "Get category description by ID",
                "server": "assortment_api",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category_id": {"type": "string", "description": "Category ID to look up"}
                    },
                    "required": ["category_id"],
                },
            },
        ],
        "metadata": {"version": "1.0.0", "total_tools": 3},
    }


@pytest.fixture
def tool_discovery_service():
    """Create ToolDiscoveryService instance."""
    return ToolDiscoveryService(
        base_url="https://grainger-mcp-servers.svc.ue2.prod.mlops.prod.aws.grainger.com",
        auth_token="test_token_123",
    )


class TestSchemaTransformation:
    """Test MCP to OpenAI schema transformation."""

    def test_transform_basic_tool(self):
        """Test transforming a basic MCP tool to OpenAI format."""
        mcp_tool = {
            "name": "get_product_docs",
            "description": "Search for products",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query"}},
                "required": ["query"],
            },
        }

        result = transform_mcp_to_openai_format(mcp_tool)

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_product_docs"
        assert result["function"]["description"] == "Search for products"
        assert result["function"]["parameters"]["type"] == "object"
        assert "query" in result["function"]["parameters"]["properties"]
        assert result["function"]["parameters"]["required"] == ["query"]

    def test_transform_tool_with_enum(self):
        """Test transforming tool with enum values."""
        mcp_tool = {
            "name": "search_tool",
            "description": "Search tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "store": {
                        "type": "string",
                        "enum": ["Product", "Document"],
                        "description": "Store type",
                    }
                },
                "required": ["store"],
            },
        }

        result = transform_mcp_to_openai_format(mcp_tool)

        store_prop = result["function"]["parameters"]["properties"]["store"]
        assert store_prop["type"] == "string"
        assert store_prop["enum"] == ["Product", "Document"]

    def test_transform_tool_with_nested_objects(self):
        """Test transforming tool with nested object properties."""
        mcp_tool = {
            "name": "complex_tool",
            "description": "Complex tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filters": {
                        "type": "object",
                        "properties": {
                            "brand": {"type": "string"},
                            "price_range": {
                                "type": "object",
                                "properties": {
                                    "min": {"type": "number"},
                                    "max": {"type": "number"},
                                },
                            },
                        },
                    }
                },
                "required": [],
            },
        }

        result = transform_mcp_to_openai_format(mcp_tool)

        filters = result["function"]["parameters"]["properties"]["filters"]
        assert filters["type"] == "object"
        assert "brand" in filters["properties"]
        assert "price_range" in filters["properties"]
        assert filters["properties"]["price_range"]["type"] == "object"

    def test_transform_tool_with_array(self):
        """Test transforming tool with array properties."""
        mcp_tool = {
            "name": "batch_tool",
            "description": "Batch tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "skus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of SKUs",
                    }
                },
                "required": ["skus"],
            },
        }

        result = transform_mcp_to_openai_format(mcp_tool)

        skus = result["function"]["parameters"]["properties"]["skus"]
        assert skus["type"] == "array"
        assert skus["items"]["type"] == "string"


class TestToolDiscoveryService:
    """Test ToolDiscoveryService."""

    @pytest.mark.asyncio
    async def test_discover_tools_success(self, tool_discovery_service, mock_discovery_response):
        """Test successful tool discovery."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_discovery_response)
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await tool_discovery_service.discover_tools()

            assert isinstance(result, DiscoveredTools)
            assert len(result.tools) == 3
            assert len(result.tool_endpoints) == 3
            assert result.tool_endpoints["get_product_docs"] == "/product_retrieval/mcp"
            assert result.tool_endpoints["parse_customer_query"] == "/parse_query/mcp"
            assert result.tool_endpoints["get_category_description"] == "/assortment_api/mcp"

    @pytest.mark.asyncio
    async def test_discover_tools_network_error(self, tool_discovery_service):
        """Test tool discovery with network error."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_get.side_effect = Exception("Network error")

            result = await tool_discovery_service.discover_tools()

            assert result is None

    @pytest.mark.asyncio
    async def test_discover_tools_invalid_response(self, tool_discovery_service):
        """Test tool discovery with invalid response."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"invalid": "response"})
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await tool_discovery_service.discover_tools()

            assert result is None

    @pytest.mark.asyncio
    async def test_discover_tools_http_error(self, tool_discovery_service):
        """Test tool discovery with HTTP error."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await tool_discovery_service.discover_tools()

            assert result is None

    @pytest.mark.asyncio
    async def test_discover_tools_uses_correct_headers(self, tool_discovery_service):
        """Test that discovery uses correct authentication headers."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={"servers": [], "tools": [], "metadata": {}}
            )
            mock_session.get.return_value.__aenter__.return_value = mock_response

            await tool_discovery_service.discover_tools()

            # Verify session was created with correct headers
            call_args = mock_session_class.call_args
            headers = call_args[1]["headers"]
            assert headers["Authorization"] == "Bearer test_token_123"
            assert headers["Accept"] == "application/json"

    def test_get_openai_tool_schemas(self, tool_discovery_service, mock_discovery_response):
        """Test getting OpenAI-formatted tool schemas."""
        discovered = DiscoveredTools(
            tools=mock_discovery_response["tools"],
            servers=mock_discovery_response["servers"],
            metadata=mock_discovery_response["metadata"],
        )

        openai_schemas = discovered.get_openai_tool_schemas()

        assert len(openai_schemas) == 3
        for schema in openai_schemas:
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    def test_get_tool_endpoint(self, mock_discovery_response):
        """Test getting tool endpoint mapping."""
        discovered = DiscoveredTools(
            tools=mock_discovery_response["tools"],
            servers=mock_discovery_response["servers"],
            metadata=mock_discovery_response["metadata"],
        )

        assert discovered.get_tool_endpoint("get_product_docs") == "/product_retrieval/mcp"
        assert discovered.get_tool_endpoint("parse_customer_query") == "/parse_query/mcp"
        assert discovered.get_tool_endpoint("unknown_tool") is None


class TestToolDiscoveryCaching:
    """Test tool discovery caching."""

    @pytest.mark.asyncio
    async def test_cache_hit(self, tool_discovery_service, mock_discovery_response):
        """Test that cached tools are returned without making API call."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_discovery_response)
            mock_get.return_value.__aenter__.return_value = mock_response

            # First call - should hit API
            result1 = await tool_discovery_service.discover_tools()
            assert mock_get.call_count == 1

            # Second call - should use cache
            result2 = await tool_discovery_service.discover_tools()
            assert mock_get.call_count == 1  # No additional call
            assert result1 == result2

    @pytest.mark.asyncio
    async def test_cache_expiry(self, tool_discovery_service, mock_discovery_response):
        """Test that cache expires after TTL."""
        tool_discovery_service.cache_ttl_seconds = 0.1  # 100ms TTL

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_discovery_response)
            mock_get.return_value.__aenter__.return_value = mock_response

            # First call
            await tool_discovery_service.discover_tools()
            assert mock_get.call_count == 1

            # Wait for cache to expire
            import asyncio

            await asyncio.sleep(0.2)

            # Second call - should hit API again
            await tool_discovery_service.discover_tools()
            assert mock_get.call_count == 2

    @pytest.mark.asyncio
    async def test_force_refresh(self, tool_discovery_service, mock_discovery_response):
        """Test forcing cache refresh."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_discovery_response)
            mock_get.return_value.__aenter__.return_value = mock_response

            # First call
            await tool_discovery_service.discover_tools()
            assert mock_get.call_count == 1

            # Force refresh
            await tool_discovery_service.discover_tools(force_refresh=True)
            assert mock_get.call_count == 2


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_transform_tool_without_required_fields(self):
        """Test transforming tool without required fields."""
        mcp_tool = {
            "name": "simple_tool",
            "description": "Simple tool",
            "inputSchema": {
                "type": "object",
                "properties": {"param": {"type": "string"}},
                # No required field
            },
        }

        result = transform_mcp_to_openai_format(mcp_tool)

        assert result["function"]["parameters"].get("required", []) == []

    def test_transform_tool_with_empty_properties(self):
        """Test transforming tool with no parameters."""
        mcp_tool = {
            "name": "no_param_tool",
            "description": "Tool with no parameters",
            "inputSchema": {"type": "object", "properties": {}},
        }

        result = transform_mcp_to_openai_format(mcp_tool)

        assert result["function"]["parameters"]["properties"] == {}

    @pytest.mark.asyncio
    async def test_discover_tools_with_missing_server_mapping(self, tool_discovery_service):
        """Test discovery when tool references non-existent server."""
        invalid_response = {
            "servers": [{"name": "server1", "endpoint": "/server1/mcp"}],
            "tools": [
                {
                    "name": "orphan_tool",
                    "description": "Tool without server",
                    "server": "non_existent_server",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ],
            "metadata": {},
        }

        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=invalid_response)
            mock_get.return_value.__aenter__.return_value = mock_response

            result = await tool_discovery_service.discover_tools()

            # Should handle gracefully - tool endpoint will be None
            assert result is not None
            assert result.get_tool_endpoint("orphan_tool") is None
