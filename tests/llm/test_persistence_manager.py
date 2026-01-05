"""
Unit tests for persistence manager and SQLite checkpointer functionality.
"""

import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.llm.persistence_manager import PersistenceManager

pytestmark = pytest.mark.asyncio


class TestPersistenceManager:
    """Test cases for PersistenceManager class."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            yield tmp.name
        # Cleanup is handled by tempfile

    @pytest_asyncio.fixture
    async def persistence_manager(self, temp_db_path):
        """Create a PersistenceManager instance for testing."""
        manager = await PersistenceManager.create(database_path=temp_db_path)
        try:
            yield manager
        finally:
            await manager.aclose()

    async def test_persistence_manager_initialization(self, temp_db_path):
        """Test PersistenceManager initialization."""
        manager = await PersistenceManager.create(database_path=temp_db_path)
        try:
            assert manager.database_path == temp_db_path
            assert manager.checkpointer is not None
        finally:
            await manager.aclose()

    async def test_get_session_config(self, persistence_manager):
        """Test getting session configuration."""
        session_id = "test_session_123"
        config = persistence_manager.get_session_config(session_id)

        assert config["configurable"]["thread_id"] == session_id
        assert "thread_id" in config["configurable"]

    async def test_get_session_config_with_checkpoint(self, persistence_manager):
        """Test getting session configuration with checkpoint."""
        session_id = "test_session_123"
        checkpoint_id = "checkpoint_456"
        config = persistence_manager.get_session_config(session_id, checkpoint_id)

        assert config["configurable"]["thread_id"] == session_id
        assert config["configurable"]["checkpoint_id"] == checkpoint_id

    async def test_list_sessions(self, persistence_manager):
        """Test listing all sessions."""
        mock_checkpoints = [
            MagicMock(config={"configurable": {"thread_id": "session1"}}),
            MagicMock(config={"configurable": {"thread_id": "session2"}}),
            MagicMock(config={"configurable": {"thread_id": "session3"}}),
        ]

        persistence_manager.checkpointer.alist = AsyncMock(return_value=mock_checkpoints)

        sessions = await persistence_manager.list_sessions()

        assert len(sessions) == 3
        assert "session1" in sessions
        assert "session2" in sessions
        assert "session3" in sessions

    async def test_delete_session(self, persistence_manager):
        """Test deleting a session."""
        session_id = "test_session_to_delete"

        persistence_manager.checkpointer.adelete_thread = AsyncMock()

        await persistence_manager.delete_session(session_id)

        persistence_manager.checkpointer.adelete_thread.assert_called_once_with(session_id)
