import pytest

pytestmark = pytest.mark.asyncio


async def test_llm_integration_manager_reuses_session_and_updates_websocket():
    import tempfile
    from unittest.mock import AsyncMock

    from src.gateway.llm_integration import LLMIntegration, LLMIntegrationManager
    from src.llm.langgraph_workflow import LangGraphWorkflow

    async def _noop_initialize(self, asr_provider=None) -> bool:
        self.is_initialized = True
        return True

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        workflow = await LangGraphWorkflow.create(database_path=tmp.name)
    mgr = LLMIntegrationManager(workflow=workflow)

    LLMIntegration.initialize = _noop_initialize

    ws1 = AsyncMock()
    ws2 = AsyncMock()

    integration1 = await mgr.get_or_create_session("session-1", ws1)
    assert integration1.websocket is ws1

    integration2 = await mgr.get_or_create_session("session-1", ws2)
    assert integration2 is integration1
    assert integration2.websocket is ws2

    await mgr.shutdown()
