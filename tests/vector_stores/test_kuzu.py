from unittest.mock import Mock, patch
import json
import pytest
import uuid

from mem0.vector_stores.kuzu import KuzuVectorStore, OutputData
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
def kuzu_vector_store(mock_kuzu_connection_manager):
    # Mock the connection result for initialization
    mock_conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    mock_conn.execute.return_value = Mock(has_next=lambda: False)
    
    # Create a vector store instance that uses our mocked connection
    return KuzuVectorStore(
        collection_name="test_collection",
        db_path="./test_db",
        vector_dimension=1536,
        distance_metric="cosine"
    )


def test_vector_store_init(kuzu_vector_store, mock_kuzu_connection_manager):
    """Test vector store initialization."""
    # Verify the connection manager was created with the right path
    mock_kuzu_connection_manager.assert_called_once_with("./test_db")
    
    # Verify that the create_col method was called during initialization
    conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    assert conn.execute.called
    
    # Verify properties were set correctly
    assert kuzu_vector_store.collection_name == "test_collection"
    assert kuzu_vector_store.vector_dimension == 1536
    assert kuzu_vector_store.distance_metric == "cosine"


def test_insert_vectors(kuzu_vector_store, mock_kuzu_connection_manager):
    """Test inserting vectors."""
    # Mock transaction methods
    conn_manager = mock_kuzu_connection_manager.return_value
    conn = conn_manager.get_connection.return_value
    
    # Test data
    vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    payloads = [{"name": "vector1"}, {"name": "vector2"}]
    ids = ["id1", "id2"]
    
    # Execute the method
    kuzu_vector_store.insert(vectors=vectors, payloads=payloads, ids=ids)
    
    # Verify transaction handling
    conn_manager.begin_transaction.assert_called_once()
    assert conn.execute.called
    conn_manager.commit.assert_called_once()


def test_search_vectors(kuzu_vector_store, mock_kuzu_connection_manager):
    """Test searching for vectors."""
    # Setup mock return value for search
    conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    mock_result = Mock()
    mock_result.has_next.side_effect = [True, True, False]  # Return 2 results then stop
    
    # Create two mock rows for the results
    mock_rows = [
        {"id": "id1", "payload": json.dumps({"name": "vector1"}), "similarity": 0.95},
        {"id": "id2", "payload": json.dumps({"name": "vector2"}), "similarity": 0.85}
    ]
    mock_result.get_next.side_effect = mock_rows
    conn.execute.return_value = mock_result
    
    # Execute the search
    query_vector = [[0.1, 0.2, 0.3]]
    results = kuzu_vector_store.search(query="", vectors=query_vector, limit=2)
    
    # Verify the search was executed
    assert conn.execute.called
    
    # Verify results
    assert len(results) == 2
    assert results[0].id == "id1"
    assert results[0].score == 0.95
    assert results[0].payload == {"name": "vector1"}
    assert results[1].id == "id2"
    assert results[1].score == 0.85
    assert results[1].payload == {"name": "vector2"}


def test_delete_vector(kuzu_vector_store, mock_kuzu_connection_manager):
    """Test deleting a vector by ID."""
    conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    
    # Execute delete
    kuzu_vector_store.delete(vector_id="id1")
    
    # Verify delete query was executed
    assert conn.execute.called
    # Would ideally check the query content, but that's implementation-specific


def test_update_vector(kuzu_vector_store, mock_kuzu_connection_manager):
    """Test updating a vector."""
    conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    
    # Test data
    vector_id = "id1"
    new_vector = [0.7, 0.8, 0.9]
    new_payload = {"name": "updated_vector"}
    
    # Execute update
    kuzu_vector_store.update(vector_id=vector_id, vector=new_vector, payload=new_payload)
    
    # Verify update query was executed
    assert conn.execute.called


def test_get_vector(kuzu_vector_store, mock_kuzu_connection_manager):
    """Test retrieving a vector by ID."""
    conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    
    # Setup mock return value
    mock_result = Mock()
    mock_result.has_next.side_effect = [True, False]  # Return 1 result then stop
    mock_result.get_next.return_value = {
        "id": "id1", 
        "payload": json.dumps({"name": "vector1"})
    }
    conn.execute.return_value = mock_result
    
    # Execute get
    result = kuzu_vector_store.get(vector_id="id1")
    
    # Verify get query was executed
    assert conn.execute.called
    
    # Verify result
    assert result.id == "id1"
    assert result.payload == {"name": "vector1"}


def test_list_vectors(kuzu_vector_store, mock_kuzu_connection_manager):
    """Test listing vectors."""
    conn = mock_kuzu_connection_manager.return_value.get_connection.return_value
    
    # Setup mock return value
    mock_result = Mock()
    mock_result.has_next.side_effect = [True, True, False]  # Return 2 results then stop
    
    # Create two mock rows for the results
    mock_rows = [
        {"id": "id1", "payload": json.dumps({"name": "vector1"})},
        {"id": "id2", "payload": json.dumps({"name": "vector2"})}
    ]
    mock_result.get_next.side_effect = mock_rows
    conn.execute.return_value = mock_result
    
    # Execute list
    results = kuzu_vector_store.list(limit=2)
    
    # Verify list query was executed
    assert conn.execute.called
    
    # Verify results (list returns a list of lists)
    assert len(results) == 1
    assert len(results[0]) == 2
    assert results[0][0].id == "id1"
    assert results[0][0].payload == {"name": "vector1"}
    assert results[0][1].id == "id2"
    assert results[0][1].payload == {"name": "vector2"}