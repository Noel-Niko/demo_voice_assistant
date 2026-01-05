"""
Unit tests for tool restrictions system.
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Now import normally
from llm.tool_restrictions import ToolRestrictions  # noqa: E402


class TestToolRestrictionsConfiguration:
    """Test tool restriction configuration and environment variable handling."""

    def test_order_tools_disabled_by_default(self):
        """Order tools should be disabled when ENABLE_ORDER_TOOLS is not set."""
        with patch.dict(os.environ, {}, clear=True):
            assert not ToolRestrictions.is_order_tool_enabled()

    def test_order_tools_enabled_true(self):
        """Order tools should be enabled when ENABLE_ORDER_TOOLS=true."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "true"}):
            assert ToolRestrictions.is_order_tool_enabled()

    def test_order_tools_enabled_1(self):
        """Order tools should be enabled when ENABLE_ORDER_TOOLS=1."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "1"}):
            assert ToolRestrictions.is_order_tool_enabled()

    def test_order_tools_enabled_yes(self):
        """Order tools should be enabled when ENABLE_ORDER_TOOLS=yes."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "yes"}):
            assert ToolRestrictions.is_order_tool_enabled()

    def test_order_tools_enabled_on(self):
        """Order tools should be enabled when ENABLE_ORDER_TOOLS=on."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "on"}):
            assert ToolRestrictions.is_order_tool_enabled()

    def test_order_tools_disabled_false(self):
        """Order tools should be disabled when ENABLE_ORDER_TOOLS=false."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            assert not ToolRestrictions.is_order_tool_enabled()

    def test_order_tools_disabled_0(self):
        """Order tools should be disabled when ENABLE_ORDER_TOOLS=0."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "0"}):
            assert not ToolRestrictions.is_order_tool_enabled()

    def test_order_tools_case_insensitive(self):
        """Environment variable should be case insensitive."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "TRUE"}):
            assert ToolRestrictions.is_order_tool_enabled()

        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "False"}):
            assert not ToolRestrictions.is_order_tool_enabled()


class TestToolRestrictionByName:
    """Test tool restriction based on tool name."""

    def test_get_order_info_restricted_when_disabled(self):
        """get_order_info should be restricted when order tools are disabled."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            assert ToolRestrictions.is_tool_restricted("get_order_info")

    def test_get_order_info_allowed_when_enabled(self):
        """get_order_info should be allowed when order tools are enabled."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "true"}):
            assert not ToolRestrictions.is_tool_restricted("get_order_info")

    def test_non_order_tool_always_allowed(self):
        """Non-order tools should always be allowed."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            assert not ToolRestrictions.is_tool_restricted("get_product_docs")
            assert not ToolRestrictions.is_tool_restricted("parse_customer_query")
            assert not ToolRestrictions.is_tool_restricted("get_category_description")


class TestToolRestrictionByServer:
    """Test tool restriction based on server name."""

    def test_order_server_restricted_when_disabled(self):
        """Tools from order_server should be restricted when order tools are disabled."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            assert ToolRestrictions.is_tool_restricted(
                tool_name="some_tool", server_name="order_server"
            )

    def test_order_server_allowed_when_enabled(self):
        """Tools from order_server should be allowed when order tools are enabled."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "true"}):
            assert not ToolRestrictions.is_tool_restricted(
                tool_name="some_tool", server_name="order_server"
            )

    def test_non_order_server_always_allowed(self):
        """Tools from non-order servers should always be allowed."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            assert not ToolRestrictions.is_tool_restricted(
                tool_name="some_tool", server_name="product_retrieval"
            )


class TestToolRestrictionByEndpoint:
    """Test tool restriction based on endpoint path."""

    def test_order_endpoint_restricted_when_disabled(self):
        """Tools with /order/ endpoint should be restricted when order tools are disabled."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            assert ToolRestrictions.is_tool_restricted(tool_name="some_tool", endpoint="/order/mcp")

    def test_order_endpoint_allowed_when_enabled(self):
        """Tools with /order/ endpoint should be allowed when order tools are enabled."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "true"}):
            assert not ToolRestrictions.is_tool_restricted(
                tool_name="some_tool", endpoint="/order/mcp"
            )

    def test_non_order_endpoint_always_allowed(self):
        """Tools with non-order endpoints should always be allowed."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            assert not ToolRestrictions.is_tool_restricted(
                tool_name="some_tool", endpoint="/product_retrieval/mcp"
            )


class TestFilterTools:
    """Test filtering tools from a list."""

    def test_filter_order_tools_when_disabled(self):
        """Should filter out order tools when disabled."""
        tools = [
            {"name": "get_order_info", "server": "order_server"},
            {"name": "get_product_docs", "server": "product_retrieval"},
        ]

        servers = [
            {"name": "order_server", "endpoint": "/order/mcp"},
            {"name": "product_retrieval", "endpoint": "/product_retrieval/mcp"},
        ]

        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            filtered = ToolRestrictions.filter_tools(tools, servers)

            assert len(filtered) == 1
            assert filtered[0]["name"] == "get_product_docs"

    def test_filter_no_tools_when_enabled(self):
        """Should not filter any tools when order tools are enabled."""
        tools = [
            {"name": "get_order_info", "server": "order_server"},
            {"name": "get_product_docs", "server": "product_retrieval"},
        ]

        servers = [
            {"name": "order_server", "endpoint": "/order/mcp"},
            {"name": "product_retrieval", "endpoint": "/product_retrieval/mcp"},
        ]

        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "true"}):
            filtered = ToolRestrictions.filter_tools(tools, servers)

            assert len(filtered) == 2

    def test_filter_by_endpoint_pattern(self):
        """Should filter tools by endpoint pattern."""
        tools = [
            {"name": "tool1", "server": "order_server"},
            {"name": "tool2", "server": "product_server"},
        ]

        servers = [
            {"name": "order_server", "endpoint": "/order/mcp"},
            {"name": "product_server", "endpoint": "/product/mcp"},
        ]

        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            filtered = ToolRestrictions.filter_tools(tools, servers)

            assert len(filtered) == 1
            assert filtered[0]["name"] == "tool2"

    def test_filter_empty_list(self):
        """Should handle empty tool list."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            filtered = ToolRestrictions.filter_tools([])
            assert filtered == []

    def test_filter_without_servers(self):
        """Should filter by tool name even without server info."""
        tools = [
            {"name": "get_order_info"},
            {"name": "get_product_docs"},
        ]

        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            filtered = ToolRestrictions.filter_tools(tools)

            assert len(filtered) == 1
            assert filtered[0]["name"] == "get_product_docs"


class TestRestrictionSummary:
    """Test restriction summary generation."""

    def test_summary_when_disabled(self):
        """Should return correct summary when order tools are disabled."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            summary = ToolRestrictions.get_restriction_summary()

            assert summary["order_tools_enabled"] is False
            assert "get_order_info" in summary["restricted_order_tools"]
            assert "/order/mcp" in summary["restricted_order_endpoints"]
            assert "order_server" in summary["restricted_order_servers"]

    def test_summary_when_enabled(self):
        """Should return correct summary when order tools are enabled."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "true"}):
            summary = ToolRestrictions.get_restriction_summary()

            assert summary["order_tools_enabled"] is True
            # Still lists what would be restricted, but they're not active
            assert "get_order_info" in summary["restricted_order_tools"]


class TestMultipleCriteria:
    """Test restriction with multiple criteria."""

    def test_restriction_by_name_overrides_server(self):
        """Tool name in restricted list should be blocked regardless of server."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            # Even if server is not order_server, tool name should trigger restriction
            assert ToolRestrictions.is_tool_restricted(
                tool_name="get_order_info", server_name="some_other_server"
            )

    def test_restriction_by_any_criteria(self):
        """Tool should be restricted if ANY criteria matches."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            # Restricted by tool name
            assert ToolRestrictions.is_tool_restricted("get_order_info")

            # Restricted by server
            assert ToolRestrictions.is_tool_restricted(
                tool_name="unknown_tool", server_name="order_server"
            )

            # Restricted by endpoint
            assert ToolRestrictions.is_tool_restricted(
                tool_name="unknown_tool", endpoint="/order/mcp"
            )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_values(self):
        """Should handle None values gracefully."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            assert not ToolRestrictions.is_tool_restricted(
                tool_name="get_product_docs", server_name=None, endpoint=None
            )

    def test_empty_strings(self):
        """Should handle empty strings gracefully."""
        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            assert not ToolRestrictions.is_tool_restricted(
                tool_name="get_product_docs", server_name="", endpoint=""
            )

    def test_malformed_tool_dict(self):
        """Should handle malformed tool dictionaries."""
        tools = [
            {},  # Missing name
            {"name": ""},  # Empty name
            {"name": "get_product_docs"},  # Valid
        ]

        with patch.dict(os.environ, {"ENABLE_ORDER_TOOLS": "false"}):
            filtered = ToolRestrictions.filter_tools(tools)
            # Should not crash, should filter based on what's available
            assert isinstance(filtered, list)
