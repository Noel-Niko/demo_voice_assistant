# Dynamic Tool Discovery Implementation

## Overview

The Voice Assistant now dynamically discovers all available tools from the MCP (Model Context Protocol) servers at application startup, eliminating the need for hardcoded tool definitions. This enables the system to automatically use all available tools across multiple MCP server endpoints.

## Architecture

### Components

1. **ToolDiscoveryService** (`src/llm/tool_discovery.py`)
   - Calls `/tools/discovery/all` endpoint at startup
   - Parses MCP tool schemas and server metadata
   - Transforms MCP `inputSchema` format to OpenAI function calling format
   - Implements caching with configurable TTL (default: 1 hour)
   - Handles errors gracefully with fallback to hardcoded tools

2. **MCPToolManager** (`src/llm/tool_manager.py`)
   - Updated to accept dynamically discovered tools
   - Stores per-tool endpoint mapping (`_tool_endpoints`)
   - Routes tool calls to correct MCP server endpoint
   - Falls back to hardcoded tools if discovery fails

3. **Application Startup** (`src/gateway/routes.py`)
   - Runs tool discovery during FastAPI lifespan initialization
   - Stores discovery status and discovered tools in `app.state`
   - Passes discovered tools to LLMIntegrationManager

4. **UI Integration** (`src/gateway/routes.py` HTML/JS)
   - Displays tool discovery status panel
   - Shows success/failure/progress indicators
   - Auto-hides after completion

## Discovery Flow

```
Application Startup
    ↓
ToolDiscoveryService.discover_tools()
    ↓
GET /tools/discovery/all
    ↓
Parse Response:
  - servers: [{name, endpoint, description}]
  - tools: [{name, description, server, inputSchema}]
  - metadata: {version, total_tools}
    ↓
Transform MCP → OpenAI Format:
  - inputSchema → parameters
  - Wrap in OpenAI function structure
    ↓
Build Tool-to-Endpoint Mapping:
  - tool_name → /service/mcp endpoint
    ↓
Pass to MCPToolManager
    ↓
Tools Available for LLM
```

## MCP Tool Schema Format

### Discovery Response Format
```json
{
  "servers": [
    {
      "name": "product_retrieval",
      "endpoint": "/product_retrieval/mcp",
      "description": "Product search and retrieval tools"
    },
    {
      "name": "parse_query",
      "endpoint": "/parse_query/mcp",
      "description": "Query parsing tools"
    },
    {
      "name": "assortment_api",
      "endpoint": "/assortment_api/mcp",
      "description": "Category and assortment tools"
    }
  ],
  "tools": [
    {
      "name": "get_product_docs",
      "description": "Search for products using semantic search",
      "server": "product_retrieval",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "description": "Natural language search query"
          },
          "vector_store": {
            "type": "string",
            "enum": ["Product"],
            "description": "Vector store to search"
          }
        },
        "required": ["query", "vector_store"]
      }
    }
  ],
  "metadata": {
    "version": "1.0.0",
    "total_tools": 15
  }
}
```

### Transformed OpenAI Format
```json
{
  "type": "function",
  "function": {
    "name": "get_product_docs",
    "description": "Search for products using semantic search",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Natural language search query"
        },
        "vector_store": {
          "type": "string",
          "enum": ["Product"],
          "description": "Vector store to search"
        }
      },
      "required": ["query", "vector_store"]
    }
  }
}
```

## Tool Execution Flow

### JSON-RPC 2.0 Request
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "parse_customer_query",
    "arguments": {
      "query": "alternate tape for item 40TU33"
    }
  },
  "id": 1735824000000
}
```

### SSE Response Format
```
event: message
data: {"method":"notifications/message","params":{"level":"info","data":{"msg":"MCP: parse_customer_query started","extra":null}},"jsonrpc":"2.0"}

event: message
data: {"method":"notifications/message","params":{"level":"info","data":{"msg":"MCP: parse_customer_query completed","extra":null}},"jsonrpc":"2.0"}

event: message
data: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"{...}"}],"structuredContent":{...},"isError":false}}
```

## Configuration

### Environment Variables

```bash
# MCP Server Configuration
MCP_BASE_URL=https://grainger-mcp-servers.svc.ue2.prod.mlops.prod.aws.grainger.com
MCP_AUTH_TOKEN=your_bearer_token_here
# Alternative: TOKEN=your_bearer_token_here

# Optional: Cache TTL (default: 3600 seconds)
TOOL_DISCOVERY_CACHE_TTL=3600
```

### Discovery Endpoint

**URL:** `GET /tools/discovery/all`

**Headers:**
- `Authorization: Bearer $TOKEN`
- `Accept: application/json`

**Alternative Endpoints:**
- `/tools/discovery` - Basic discovery without full schemas
- `/tools/discovery/role/{role}` - Filtered by role (e.g., product_search)

## Fallback Behavior

If tool discovery fails (network error, authentication failure, invalid response), the system automatically falls back to hardcoded tools:

1. `get_product_docs` → `/product_retrieval/mcp`
2. `get_raw_docs` → `/product_retrieval/mcp`

**Fallback Triggers:**
- No `MCP_AUTH_TOKEN` or `TOKEN` in environment
- Discovery endpoint returns non-200 status
- Discovery endpoint returns invalid JSON
- Network timeout or connection error

## UI Status Reporting

### Tool Discovery Panel

The UI displays a status panel during tool discovery:

**States:**
- **Discovering** (🔍): Blue panel, "Discovering tools..."
- **Success** (✅): Green panel, "Tool discovery complete: X tools from Y servers (Z.Zs)"
- **Failed** (⚠️): Red panel, "Tool discovery failed: Using fallback hardcoded tools"

**Auto-hide:**
- Success: 5 seconds
- Failed: 8 seconds

### WebSocket Messages

**Request:**
```json
{
  "type": "tool_discovery_request"
}
```

**Response:**
```json
{
  "type": "tool_discovery_status",
  "status": "completed",
  "total_tools": 15,
  "total_servers": 3,
  "duration": 2.34,
  "error": null
}
```

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
# Test ToolDiscoveryService
python -m pytest tests/llm/test_tool_discovery.py -v

# All 17 tests should pass:
# - Schema transformation (4 tests)
# - Discovery service (7 tests)
# - Caching (3 tests)
# - Edge cases (3 tests)
```

### Integration Testing

Test with actual MCP endpoint:

```bash
# Set your token
export TOKEN="your_bearer_token_here"

# Start the server
python src/gateway/webrtc_server.py

# Check discovery status
curl http://localhost:8000/tool-discovery-status

# Expected response:
{
  "status": "success",
  "discovery": {
    "status": "completed",
    "total_tools": 15,
    "total_servers": 3,
    "duration": 2.34,
    "error": null
  }
}
```

### Manual Testing

1. Open browser to `http://localhost:8000`
2. Look for tool discovery panel at top of page
3. Should show: "✅ Tool discovery complete: X tools from Y servers (Z.Zs)"
4. Panel auto-hides after 5 seconds
5. Check browser console for discovery logs

## Monitoring & Logging

### Server Logs

```
INFO - Starting tool discovery...
INFO - ✅ Tool discovery complete: 15 tools from 3 servers in 2.34s
INFO - MCP Tool Manager initialized with 15 tools
INFO - LLMIntegrationManager initialized on app.state
```

### Error Logs

```
WARNING - ⚠️ Tool discovery failed, using fallback hardcoded tools
ERROR - Discovery endpoint returned 401: Unauthorized
INFO - Initialized 2 fallback hardcoded tools
```

## Performance

- **Discovery Time:** ~2-3 seconds at startup
- **Cache Duration:** 1 hour (configurable)
- **Cache Refresh:** Automatic on expiry or manual via `force_refresh=True`
- **Startup Impact:** Minimal - runs asynchronously during lifespan initialization

## Known Endpoints

Based on actual MCP server deployment:

| Endpoint | Tools | Description |
|----------|-------|-------------|
| `/product_retrieval/mcp` | `get_product_docs`, `get_raw_docs` | Product search and retrieval |
| `/parse_query/mcp` | `parse_customer_query` | Query parsing and entity extraction |
| `/assortment_api/mcp` | `get_category_description` | Category and assortment information |

## Future Enhancements

1. **Hot Reload:** Refresh tools without server restart
2. **Tool Metrics:** Track tool usage and performance
3. **Selective Discovery:** Filter tools by role/category
4. **Health Checks:** Periodic validation of tool availability
5. **Tool Versioning:** Handle tool schema changes gracefully

## Troubleshooting

### Issue: "Tool discovery failed"

**Causes:**
- Missing or invalid `MCP_AUTH_TOKEN`/`TOKEN`
- Network connectivity to MCP servers
- Discovery endpoint unavailable

**Solutions:**
1. Check environment variables: `echo $TOKEN`
2. Test endpoint manually: `curl -H "Authorization: Bearer $TOKEN" https://grainger-mcp-servers.../tools/discovery/all`
3. Check server logs for detailed error messages
4. System will use fallback hardcoded tools

### Issue: "Tool not found"

**Causes:**
- Tool not returned by discovery endpoint
- Tool name mismatch

**Solutions:**
1. Check discovery status: `GET /tool-discovery-status`
2. Verify tool exists in discovery response
3. Check tool name spelling matches exactly

### Issue: "Wrong endpoint for tool"

**Causes:**
- Tool-to-endpoint mapping incorrect
- Server metadata missing in discovery response

**Solutions:**
1. Check discovery response includes `server` field for each tool
2. Verify `servers` array includes matching server name
3. Check logs for endpoint mapping: "Loaded X dynamically discovered tools"

## API Reference

### ToolDiscoveryService

```python
from src.llm.tool_discovery import ToolDiscoveryService

service = ToolDiscoveryService(
    base_url="https://grainger-mcp-servers...",
    auth_token="your_token",
    cache_ttl_seconds=3600
)

# Discover tools
discovered = await service.discover_tools()

# Force refresh
discovered = await service.discover_tools(force_refresh=True)

# Get cached tools
cached = service.get_cached_tools()

# Clear cache
service.clear_cache()
```

### MCPToolManager

```python
from src.llm.tool_manager import MCPToolManager

# With discovered tools
manager = MCPToolManager(
    base_url="https://grainger-mcp-servers...",
    auth_token="your_token",
    discovered_tools={
        "tools": [...],  # OpenAI format
        "tool_endpoints": {"tool_name": "/endpoint"}
    }
)

# Without discovered tools (uses fallback)
manager = MCPToolManager(
    base_url="https://grainger-mcp-servers...",
    auth_token="your_token"
)
```

## Security Considerations

1. **Token Storage:** Store `MCP_AUTH_TOKEN` in `.env` file (gitignored)
2. **Token Rotation:** Update token in environment when rotated
3. **HTTPS Only:** Discovery endpoint uses HTTPS
4. **Bearer Authentication:** Standard OAuth 2.0 Bearer token format
5. **No Token Logging:** Token never logged or exposed in UI

## Maintenance

### Updating Fallback Tools

If discovery is unavailable, update fallback tools in `src/llm/tool_manager.py`:

```python
def _initialize_tool_definitions(self):
    """Initialize fallback hardcoded tool definitions."""
    self._available_tools.update({
        "new_tool": {
            "type": "function",
            "function": {
                "name": "new_tool",
                "description": "...",
                "parameters": {...}
            }
        }
    })
    
    self._tool_endpoints = {
        "new_tool": "/new_service/mcp"
    }
```

### Cache Management

```python
# Clear cache programmatically
from src.llm.tool_discovery import ToolDiscoveryService

service = ToolDiscoveryService(...)
service.clear_cache()

# Adjust TTL
service.cache_ttl_seconds = 7200  # 2 hours
```

---

**Implementation Date:** January 2, 2026  
**Version:** 1.0.0  
**Status:** Production Ready  
**Test Coverage:** 17/17 unit tests passing
