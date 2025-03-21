from unittest.mock import Mock, patch, MagicMock
import pytest
import json

from mem0.memory.graph_memory_kuzu import KuzuMemoryGraph
from mem0.utils.kuzu_connection import KuzuConnectionManager


@pytest.fixture
def mock_kuzu_connection_manager():
    with patch("mem0.utils.kuzu_connection.KuzuConnectionManager") as mock_manager:
        # Setup the manager instance with mocked connection
        mock_instance = Mock()
        mock_instance.get_connection.return_value = Mock()
        mock_manager.return_value = mock_instance
        yield mock_manager


@pytest.fixture
def mock_embedder():
    with patch("mem0.utils.factory.EmbedderFactory.create") as mock_create:
        mock_embedder = Mock()
        # Mock the embed method to return a fixed embedding
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        mock_create.return_value = mock_embedder
        yield mock_embedder


@pytest.fixture
def mock_llm():
    with patch("mem0.utils.factory.LlmFactory.create") as mock_create:
        mock_llm = Mock()
        mock_create.return_value = mock_llm
        yield mock_llm


@pytest.fixture
def mock_config():
    config = Mock()
    config.graph_store.config.db_path = "./test_db"
    config.embedder.provider = "test_embedder"
    config.embedder.config = {}
    config.llm.provider = "test_llm"
    config.llm.config = {}
    config.graph_store.llm = None
    config.graph_store.custom_prompt = None
    return config


@pytest.fixture
def kuzu_memory_graph(mock_kuzu_connection_manager, mock_embedder, mock_llm, mock_config):
    # Mock the connection result for schema initialization
    mock_conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    # First call raises exception to simulate table not existing, then subsequent calls succeed
    mock_conn.execute.side_effect = [Exception(), None, None, None]
    
    return KuzuMemoryGraph(config=mock_config)


def test_memory_graph_init(kuzu_memory_graph, mock_kuzu_connection_manager, mock_config):
    """Test memory graph initialization."""
    # Verify the connection manager was created with the right path
    mock_kuzu_connection_manager.assert_called_once_with("./test_db")
    
    # Verify that the schema initialization was called
    conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    assert conn.execute.called
    
    # Verify properties were set correctly
    assert kuzu_memory_graph.db_path == "./test_db"
    assert kuzu_memory_graph.threshold == 0.7


def test_add_to_graph(kuzu_memory_graph, mock_kuzu_connection_manager, mock_embedder, mock_llm):
    """Test adding data to the graph."""
    conn_manager = mock_kuzu_connection_manager.return_value
    conn = conn_manager.get_connection.return_value
    
    # Mock LLM response for entity extraction
    mock_llm.generate_response.side_effect = [
        # First call extracts entities
        {
            "tool_calls": [
                {
                    "arguments": {
                        "entities": [
                            {"entity": "user1", "entity_type": "person"},
                            {"entity": "project1", "entity_type": "project"}
                        ]
                    }
                }
            ]
        },
        # Second call extracts relationships
        {
            "tool_calls": [
                {
                    "arguments": {
                        "entities": [
                            {
                                "source": "user1",
                                "relationship": "works_on",
                                "destination": "project1"
                            }
                        ]
                    }
                }
            ]
        }
    ]
    
    # Mock search results as empty (no existing nodes)
    search_result_mock = Mock()
    search_result_mock.has_next.return_value = False
    conn.execute.return_value = search_result_mock
    
    # Execute add method
    result = kuzu_memory_graph.add(
        data="User1 is working on Project1", 
        filters={"user_id": "test_user"}
    )
    
    # Verify LLM was called for entity extraction and relationship detection
    assert mock_llm.generate_response.call_count == 2
    
    # Verify transaction was used
    conn_manager.begin_transaction.assert_called()
    conn_manager.commit.assert_called()
    
    # Verify embedder was used to create embeddings
    assert mock_embedder.embed.called
    
    # Verify result format
    assert "added_entities" in result
    assert "deleted_entities" in result


def test_search_graph(kuzu_memory_graph, mock_kuzu_connection_manager, mock_embedder, mock_llm):
    """Test searching the graph."""
    conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    
    # Mock LLM response for entity extraction
    mock_llm.generate_response.return_value = {
        "tool_calls": [
            {
                "arguments": {
                    "entities": [
                        {"entity": "project1", "entity_type": "project"}
                    ]
                }
            }
        ]
    }
    
    # Mock search results
    mock_result = Mock()
    mock_result.has_next.side_effect = [True, False]  # Return 1 result then stop
    mock_result.get_next.return_value = {
        "source": "user1", 
        "relationship": "works_on", 
        "destination": "project1",
        "similarity": 0.95
    }
    conn.execute.return_value = mock_result
    
    # Execute search
    with patch("rank_bm25.BM25Okapi") as mock_bm25:
        # Mock BM25 reranking
        mock_bm25_instance = Mock()
        mock_bm25_instance.get_top_n.return_value = [
            ["user1", "works_on", "project1"]
        ]
        mock_bm25.return_value = mock_bm25_instance
        
        results = kuzu_memory_graph.search(
            query="Who works on Project1?", 
            filters={"user_id": "test_user"}
        )
    
    # Verify LLM was called for entity extraction
    assert mock_llm.generate_response.called
    
    # Verify embedding was created
    assert mock_embedder.embed.called
    
    # Verify search returned expected result
    assert len(results) == 1
    assert results[0]["source"] == "user1"
    assert results[0]["relationship"] == "works_on"
    assert results[0]["destination"] == "project1"


def test_delete_all(kuzu_memory_graph, mock_kuzu_connection_manager):
    """Test deleting all graph data for a user."""
    conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    
    # Execute delete_all
    kuzu_memory_graph.delete_all(filters={"user_id": "test_user"})
    
    # Verify delete queries were executed
    assert conn.execute.call_count >= 2  # At least two calls for relationships and nodes


def test_get_all(kuzu_memory_graph, mock_kuzu_connection_manager):
    """Test retrieving all graph data for a user."""
    conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    
    # Mock search results
    mock_result = Mock()
    mock_result.has_next.side_effect = [True, True, False]  # Return 2 results then stop
    mock_rows = [
        {"source": "user1", "relationship": "works_on", "target": "project1"},
        {"source": "user1", "relationship": "manages", "target": "project2"}
    ]
    mock_result.get_next.side_effect = mock_rows
    conn.execute.return_value = mock_result
    
    # Execute get_all
    results = kuzu_memory_graph.get_all(
        filters={"user_id": "test_user"},
        limit=10
    )
    
    # Verify query was executed
    assert conn.execute.called
    
    # Verify results
    assert len(results) == 2
    assert results[0]["source"] == "user1"
    assert results[0]["relationship"] == "works_on"
    assert results[0]["destination"] == "project1"  # Note the field name change
    assert results[1]["source"] == "user1"
    assert results[1]["relationship"] == "manages"
    assert results[1]["destination"] == "project2"  # Note the field name change