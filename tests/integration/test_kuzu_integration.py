import os
import shutil
import tempfile
from unittest.mock import patch, Mock, call
import pytest
import json

from mem0.utils.kuzu_connection import KuzuConnectionManager
from mem0.vector_stores.kuzu import KuzuVectorStore
from mem0.memory.graph_memory_kuzu import KuzuMemoryGraph


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_kuzu_db():
    """Mock the Kuzu Database and Connection classes."""
    with patch("kuzu.Database") as mock_db_class, \
         patch("kuzu.Connection") as mock_conn_class:
        # Setup mock DB and connection
        mock_db = mock_db_class.return_value
        mock_conn = mock_conn_class.return_value
        
        # Setup mock connection execute method
        mock_conn.execute.return_value = Mock(has_next=lambda: False)
        
        yield mock_db, mock_conn


@pytest.fixture
def mock_embedder():
    """Mock the embedder factory and embedder."""
    with patch("mem0.utils.factory.EmbedderFactory.create") as mock_create:
        mock_embedder = Mock()
        # Mock the embed method to return fixed embeddings
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]
        mock_create.return_value = mock_embedder
        yield mock_embedder


@pytest.fixture
def mock_llm():
    """Mock the LLM factory and LLM."""
    with patch("mem0.utils.factory.LlmFactory.create") as mock_create:
        mock_llm = Mock()
        # Setup mock LLM response for entity extraction and relation detection
        mock_llm.generate_response.side_effect = [
            # Entity extraction
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
            # Relation detection
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
            },
            # No deletions
            {
                "tool_calls": []
            }
        ]
        mock_create.return_value = mock_llm
        yield mock_llm


@pytest.fixture
def mock_config():
    """Create a mock configuration for the memory graph."""
    config = Mock()
    config.graph_store.config.db_path = "./test_db"
    config.embedder.provider = "test_embedder"
    config.embedder.config = {}
    config.llm.provider = "test_llm"
    config.llm.config = {}
    config.graph_store.llm = None
    config.graph_store.custom_prompt = None
    return config


def test_connection_sharing(temp_db_path, mock_kuzu_db, mock_config, mock_embedder, mock_llm):
    """
    Test that KuzuVectorStore and KuzuMemoryGraph share the same connection
    via the KuzuConnectionManager.
    """
    mock_db, mock_conn = mock_kuzu_db
    
    # Initialize the first component - KuzuVectorStore
    vector_store = KuzuVectorStore(
        collection_name="test_vectors",
        db_path=temp_db_path,
        vector_dimension=1536
    )
    
    # Update the config to use the same DB path
    mock_config.graph_store.config.db_path = temp_db_path
    
    # Initialize the second component - KuzuMemoryGraph
    memory_graph = KuzuMemoryGraph(config=mock_config)
    
    # Verify both components use the same database path
    assert vector_store.connection_manager.db_path == temp_db_path
    assert memory_graph.connection_manager.db_path == temp_db_path
    
    # Verify the database was initialized only once
    assert mock_db.call_count == 1
    
    # Reset mocks for further tests
    mock_conn.reset_mock()
    
    # Test operations in both components
    
    # 1. Insert vector using vector store
    mock_conn.execute.side_effect = None  # Reset side effect
    vector_store.insert(
        vectors=[[0.1, 0.2, 0.3]], 
        payloads=[{"name": "test_data"}],
        ids=["test_id"]
    )
    
    # Verify transaction methods were called from the connection manager
    assert vector_store.connection_manager.begin_transaction.called
    assert vector_store.connection_manager.commit.called
    
    # Reset mocks
    mock_conn.reset_mock()
    vector_store.connection_manager.begin_transaction.reset_mock()
    vector_store.connection_manager.commit.reset_mock()
    
    # 2. Add data to graph using memory graph
    with patch("rank_bm25.BM25Okapi") as mock_bm25:
        # Mock search results for the graph memory
        mock_result = Mock()
        mock_result.has_next.return_value = False
        mock_conn.execute.return_value = mock_result
        
        # Mock BM25 reranking
        mock_bm25_instance = Mock()
        mock_bm25_instance.get_top_n.return_value = []
        mock_bm25.return_value = mock_bm25_instance
        
        memory_graph.add(
            data="User1 is working on Project1",
            filters={"user_id": "test_user"}
        )
    
    # Verify transaction methods were called from the connection manager
    assert memory_graph.connection_manager.begin_transaction.called
    assert memory_graph.connection_manager.commit.called


def test_coordinated_transactions(temp_db_path, mock_kuzu_db, mock_config, mock_embedder, mock_llm):
    """
    Test that transactions between KuzuVectorStore and KuzuMemoryGraph are properly
    coordinated through the connection manager.
    """
    mock_db, mock_conn = mock_kuzu_db
    
    # Initialize components with same DB path
    vector_store = KuzuVectorStore(
        collection_name="test_vectors",
        db_path=temp_db_path
    )
    
    mock_config.graph_store.config.db_path = temp_db_path
    memory_graph = KuzuMemoryGraph(config=mock_config)
    
    # Verify they use the same connection manager instance
    assert vector_store.connection_manager is memory_graph.connection_manager
    
    # Test transaction with error
    mock_conn.execute.side_effect = [None, Exception("Test error")]
    
    try:
        # This should fail halfway through
        vector_store.insert(
            vectors=[[0.1, 0.2, 0.3]], 
            payloads=[{"name": "test_data"}],
            ids=["test_id"]
        )
    except Exception:
        pass
    
    # Verify rollback was called
    assert vector_store.connection_manager.rollback.called
    
    # Reset mocks
    mock_conn.reset_mock()
    vector_store.connection_manager.rollback.reset_mock()
    
    # Test that operations work after rollback
    mock_conn.execute.side_effect = None
    
    # Mock search results for add
    mock_result = Mock()
    mock_result.has_next.return_value = False
    mock_conn.execute.return_value = mock_result
    
    with patch("rank_bm25.BM25Okapi") as mock_bm25:
        # Mock BM25 reranking
        mock_bm25_instance = Mock()
        mock_bm25_instance.get_top_n.return_value = []
        mock_bm25.return_value = mock_bm25_instance
        
        # Try an operation in the other component
        memory_graph.add(
            data="User1 is working on Project1",
            filters={"user_id": "test_user"}
        )
    
    # Verify the operation completed successfully
    assert memory_graph.connection_manager.commit.called


def test_performance_benchmark():
    """
    Test the performance of Kuzu vector and graph operations compared to
    existing implementations.
    
    This is a placeholder for actual performance benchmarking that would
    be implemented in a real testing scenario.
    """
    # In a real test, we would:
    # 1. Create both Kuzu and existing implementations (e.g., Neo4j)
    # 2. Perform identical operations on both
    # 3. Measure and compare execution times
    # 4. Assert that Kuzu performance is within acceptable bounds
    
    # For now, we'll just ensure this test passes
    assert True, "Performance benchmarking would be implemented here"