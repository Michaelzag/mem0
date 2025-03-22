from unittest.mock import Mock, patch
import json
import pytest
import uuid

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

# Skip tests if kuzu is not available
pytestmark = pytest.mark.skipif(not KUZU_AVAILABLE, reason="Kuzu not installed")

from mem0.vector_stores.kuzu import KuzuVectorStore, OutputData
from mem0.utils.kuzu_connection import KuzuConnectionManager


@pytest.fixture
def mock_kuzu_connection_manager():
    # Use patch.object instead to avoid issues with super()
    with patch("mem0.utils.kuzu_connection.KuzuConnectionManager") as mock_manager_cls:
        # Create a proper mock instance that doesn't interfere with __new__
        mock_instance = Mock()
        mock_connection = Mock()
        mock_instance.get_connection.return_value = mock_connection
        
        # Configure the class mock to return our instance mock
        mock_manager_cls.return_value = mock_instance
        
        yield mock_manager_cls


def test_kuzu_vector_store_init(mock_kuzu_connection_manager):
    """Test the initialization of KuzuVectorStore."""
    # Arrange
    collection_name = "test_collection"
    db_path = "/tmp/kuzu_test"
    
    # Act
    with patch.object(KuzuConnectionManager, "__new__", return_value=mock_kuzu_connection_manager.return_value):
        store = KuzuVectorStore(collection_name=collection_name, db_path=db_path)
    
    # Assert
    assert store.collection_name == collection_name
    assert store.vector_dimension == 1536  # Default value
    assert store.distance_metric == "cosine"  # Default value


def test_kuzu_create_collection(mock_kuzu_connection_manager):
    """Test creating a collection in Kuzu vector store."""
    # Arrange
    mock_instance = mock_kuzu_connection_manager.return_value
    mock_conn = mock_instance.get_connection.return_value
    mock_conn.execute.side_effect = [Exception("Table doesn't exist"), None, None]
    
    collection_name = "test_collection"
    db_path = "/tmp/kuzu_test"
    vector_dimension = 128
    
    # Act
    with patch.object(KuzuConnectionManager, "__new__", return_value=mock_instance):
        store = KuzuVectorStore(
            collection_name=collection_name, 
            db_path=db_path,
            vector_dimension=vector_dimension
        )
    
    # Assert
    assert mock_conn.execute.call_count == 3
    # First call should be to check if table exists
    assert f"MATCH (n:{collection_name})" in mock_conn.execute.call_args_list[0][0][0]
    # Second call should be to create the table
    assert f"CREATE NODE TABLE {collection_name}" in mock_conn.execute.call_args_list[1][0][0]
    assert f"vector FLOAT[{vector_dimension}]" in mock_conn.execute.call_args_list[1][0][0]
    # Third call should be to create the vector index
    assert f"CREATE VECTOR INDEX ON {collection_name}" in mock_conn.execute.call_args_list[2][0][0]


def test_kuzu_insert(mock_kuzu_connection_manager):
    """Test inserting vectors into Kuzu vector store."""
    # Arrange
    mock_instance = mock_kuzu_connection_manager.return_value
    mock_conn = mock_instance.get_connection.return_value
    
    collection_name = "test_collection"
    db_path = "/tmp/kuzu_test"
    
    # Act
    with patch.object(KuzuConnectionManager, "__new__", return_value=mock_instance):
        store = KuzuVectorStore(collection_name=collection_name, db_path=db_path)
        
        # Setup test data
        test_vectors = [[0.1, 0.2, 0.3]]
        test_payload = [{"text": "test"}]
        test_id = ["test_id"]
        
        store.insert(vectors=test_vectors, payloads=test_payload, ids=test_id)
    
    # Assert
    mock_instance.begin_transaction.assert_called_once()
    assert mock_conn.execute.called
    mock_instance.commit.assert_called_once()


def test_kuzu_search(mock_kuzu_connection_manager):
    """Test searching vectors in Kuzu vector store."""
    # Arrange
    mock_instance = mock_kuzu_connection_manager.return_value
    mock_conn = mock_instance.get_connection.return_value
    
    # Setup mock result
    mock_result = Mock()
    mock_result.has_next.side_effect = [True, False]  # One result
    mock_result.get_next.return_value = {
        "id": "test_id",
        "payload": json.dumps({"text": "test"}),
        "similarity": 0.95
    }
    mock_conn.execute.return_value = mock_result
    
    collection_name = "test_collection"
    db_path = "/tmp/kuzu_test"
    
    # Act
    with patch.object(KuzuConnectionManager, "__new__", return_value=mock_instance):
        store = KuzuVectorStore(collection_name=collection_name, db_path=db_path)
        results = store.search(query="test", vectors=[[0.1, 0.2, 0.3]], limit=5)
    
    # Assert
    assert len(results) == 1
    assert results[0].id == "test_id"
    assert results[0].score == 0.95
    assert results[0].payload == {"text": "test"}


def test_kuzu_get(mock_kuzu_connection_manager):
    """Test getting a vector by ID from Kuzu vector store."""
    # Arrange
    mock_instance = mock_kuzu_connection_manager.return_value
    mock_conn = mock_instance.get_connection.return_value
    
    # Setup mock result
    mock_result = Mock()
    mock_result.has_next.side_effect = [True, False]  # One result
    mock_result.get_next.return_value = {
        "id": "test_id",
        "payload": json.dumps({"text": "test"})
    }
    mock_conn.execute.return_value = mock_result
    
    collection_name = "test_collection"
    db_path = "/tmp/kuzu_test"
    
    # Act
    with patch.object(KuzuConnectionManager, "__new__", return_value=mock_instance):
        store = KuzuVectorStore(collection_name=collection_name, db_path=db_path)
        result = store.get("test_id")
    
    # Assert
    assert result.id == "test_id"
    assert result.payload == {"text": "test"}


def test_kuzu_delete(mock_kuzu_connection_manager):
    """Test deleting a vector by ID from Kuzu vector store."""
    # Arrange
    mock_instance = mock_kuzu_connection_manager.return_value
    mock_conn = mock_instance.get_connection.return_value
    
    collection_name = "test_collection"
    db_path = "/tmp/kuzu_test"
    
    # Act
    with patch.object(KuzuConnectionManager, "__new__", return_value=mock_instance):
        store = KuzuVectorStore(collection_name=collection_name, db_path=db_path)
        store.delete("test_id")
    
    # Assert
    assert mock_conn.execute.called
    delete_query = mock_conn.execute.call_args[0][0]
    assert f"MATCH (n:{collection_name})" in delete_query
    assert "n.id = 'test_id'" in delete_query
    assert "DELETE n" in delete_query