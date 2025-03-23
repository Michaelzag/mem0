import json
import pytest
import datetime
from unittest.mock import Mock, patch

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

# Skip tests if kuzu is not available
pytestmark = pytest.mark.skipif(not KUZU_AVAILABLE, reason="Kuzu not installed")

from mem0.memory.base import MemoryBase
from mem0.memory.graph_memory_kuzu import KuzuMemoryGraph
from mem0.utils.kuzu_connection import KuzuConnectionManager


@pytest.fixture
def mock_kuzu_connection_manager():
    """Mock KuzuConnectionManager for testing."""
    with patch("mem0.utils.kuzu_connection.KuzuConnectionManager") as mock_manager_cls:
        # Create a proper mock instance
        mock_instance = Mock()
        mock_connection = Mock()
        mock_instance.get_connection.return_value = mock_connection
        
        # Configure the class mock to return our instance
        mock_manager_cls.return_value = mock_instance
        
        yield mock_manager_cls, mock_instance, mock_connection


@pytest.fixture
def mock_storage():
    """Mock SQLiteManager for testing."""
    storage = Mock()
    
    # Configure get_history to return some mock history records
    mock_history = [
        {
            "id": "hist1",
            "memory_id": "mem1",
            "old_memory": json.dumps({"name": "old_entity"}),
            "new_memory": json.dumps({"name": "new_entity"}),
            "event": "update",
            "created_at": datetime.datetime.now().isoformat(),
            "updated_at": datetime.datetime.now().isoformat()
        }
    ]
    storage.get_history.return_value = mock_history
    
    return storage


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns fixed embeddings."""
    embedder = Mock()
    embedder.embed.return_value = [0.1, 0.2, 0.3]
    return embedder


@pytest.fixture
def mock_llm():
    """Mock LLM that returns fixed responses for entity extraction and relationships."""
    llm = Mock()
    
    # Setup tool call responses
    tool_call_response = {
        "tool_calls": [
            {
                "arguments": {
                    "entities": [
                        {"entity": "test_entity", "entity_type": "test_type"}
                    ]
                }
            }
        ]
    }
    
    llm.generate_response.return_value = tool_call_response
    return llm


@pytest.fixture
def mock_config():
    """Create a mock configuration object."""
    config = Mock()
    
    # Setup graph store config
    config.graph_store = Mock()
    config.graph_store.config = Mock()
    config.graph_store.config.db_path = "/tmp/kuzu_test"
    config.graph_store.config.history_db_path = ":memory:"
    
    # Setup embedder and LLM config
    config.embedder = Mock()
    config.embedder.provider = "mock"
    config.embedder.config = {}
    
    config.llm = Mock()
    config.llm.provider = "mock"
    config.llm.config = {}
    
    return config


@pytest.fixture
def kuzu_memory(mock_kuzu_connection_manager, mock_config, mock_embedder, mock_llm, mock_storage):
    """Create a KuzuMemoryGraph instance with mocked dependencies."""
    manager_cls, manager_instance, connection = mock_kuzu_connection_manager
    
    # Setup factory mocks
    with patch("mem0.utils.factory.EmbedderFactory.create", return_value=mock_embedder), \
         patch("mem0.utils.factory.LlmFactory.create", return_value=mock_llm), \
         patch.object(KuzuConnectionManager, "__new__", return_value=manager_instance):
        
        # Configure the connection mock to return appropriate results for common queries
        def mock_execute(query, params=None):
            # Query result for schemas
            if "CREATE NODE TABLE" in query or "CREATE REL TABLE" in query:
                result = Mock()
                result.has_next.return_value = False
                return result
            # Mimic a get query
            elif "n.name, n.entity_type, n.user_id, n.created" in query:
                result = Mock()
                result.has_next.side_effect = [True, False]
                result.get_next.return_value = {
                    "n.name": "test_entity",
                    "n.entity_type": "test_type",
                    "n.user_id": "test_user",
                    "n.created": 123456789
                }
                return result
            # Mimic a relationship query
            elif "r:RELATED_TO" in query and "RETURN" in query:
                result = Mock()
                result.has_next.side_effect = [True, False]
                result.get_next.return_value = {
                    "target": "related_entity",
                    "type": "test_relation"
                }
                return result
            # Default empty result
            else:
                result = Mock()
                result.has_next.return_value = False
                return result
        
        # Configure the connection mock
        connection.execute.side_effect = mock_execute
        
        # Create and yield the memory graph instance
        memory = KuzuMemoryGraph(mock_config)
        # Manually set the storage to our mock
        memory.storage = mock_storage
        yield memory


def test_memory_base_inheritance(kuzu_memory):
    """Test that KuzuMemoryGraph correctly inherits from MemoryBase."""
    assert isinstance(kuzu_memory, MemoryBase)


def test_get_method(kuzu_memory, mock_kuzu_connection_manager):
    """Test the get method of KuzuMemoryGraph."""
    # Arrange
    memory_id = "test_id"
    _, _, connection = mock_kuzu_connection_manager
    
    # Act
    result = kuzu_memory.get(memory_id)
    
    # Assert
    assert result is not None
    assert isinstance(result, dict)
    assert result["name"] == "test_entity"
    assert result["entity_type"] == "test_type"
    assert "relationships" in result
    assert len(result["relationships"]) > 0


def test_update_method(kuzu_memory, mock_kuzu_connection_manager, mock_storage):
    """Test the update method of KuzuMemoryGraph."""
    # Arrange
    memory_id = "test_id"
    update_data = {
        "name": "updated_entity",
        "entity_type": "updated_type",
        "relationships": [
            {"target": "related_entity", "type": "new_relation"}
        ]
    }
    
    # Configure get to return a mock memory before update
    kuzu_memory.get = Mock(return_value={
        "id": memory_id,
        "name": "test_entity", 
        "entity_type": "test_type",
        "user_id": "test_user",
        "relationships": []
    })
    
    # Act
    result = kuzu_memory.update(memory_id, update_data)
    
    # Assert
    assert mock_storage.add_history.called
    # The connection manager's transaction methods should be called
    manager = kuzu_memory.connection_manager
    assert manager.begin_transaction.called
    assert manager.commit.called


def test_delete_method(kuzu_memory, mock_kuzu_connection_manager, mock_storage):
    """Test the delete method of KuzuMemoryGraph."""
    # Arrange
    memory_id = "test_id"
    
    # Configure get to return a mock memory before deletion
    kuzu_memory.get = Mock(return_value={
        "id": memory_id,
        "name": "test_entity", 
        "entity_type": "test_type",
        "user_id": "test_user",
        "relationships": []
    })
    
    # Act
    kuzu_memory.delete(memory_id)
    
    # Assert
    assert mock_storage.add_history.called
    # The connection manager's transaction methods should be called
    manager = kuzu_memory.connection_manager
    assert manager.begin_transaction.called
    assert manager.commit.called


def test_history_method(kuzu_memory, mock_storage):
    """Test the history method of KuzuMemoryGraph."""
    # Arrange
    memory_id = "test_id"
    
    # Act
    history = kuzu_memory.history(memory_id)
    
    # Assert
    mock_storage.get_history.assert_called_with(memory_id)
    assert isinstance(history, list)
    assert len(history) > 0
    assert "event" in history[0]
    assert history[0]["event"] == "update"


def test_error_handling_get(kuzu_memory, mock_kuzu_connection_manager):
    """Test error handling in the get method."""
    # Arrange
    memory_id = "test_id"
    _, _, connection = mock_kuzu_connection_manager
    connection.execute.side_effect = Exception("Test error")
    
    # Act
    result = kuzu_memory.get(memory_id)
    
    # Assert
    assert result is None


def test_error_handling_update(kuzu_memory, mock_kuzu_connection_manager):
    """Test error handling in the update method."""
    # Arrange
    memory_id = "test_id"
    update_data = {"name": "updated_entity"}
    
    # Configure get to return a mock memory before update
    kuzu_memory.get = Mock(return_value={
        "id": memory_id,
        "name": "test_entity", 
        "entity_type": "test_type",
        "user_id": "test_user"
    })
    
    # Configure the connection to raise an exception during execution
    _, _, connection = mock_kuzu_connection_manager
    connection.execute.side_effect = Exception("Test error")
    
    # Act
    result = kuzu_memory.update(memory_id, update_data)
    
    # Assert
    assert result is None
    # Verify rollback was called on error
    manager = kuzu_memory.connection_manager
    assert manager.rollback.called