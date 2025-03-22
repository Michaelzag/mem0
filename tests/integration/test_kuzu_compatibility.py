import os
import shutil
import tempfile
import pytest
import json
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from unittest.mock import patch, Mock  # Added missing import

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False

# Skip tests if kuzu is not available
pytestmark = pytest.mark.skipif(not KUZU_AVAILABLE, reason="Kuzu not installed")

from mem0.utils.factory import VectorStoreFactory, GraphMemoryFactory, EmbedderFactory, LlmFactory
from mem0.vector_stores.base import VectorStoreBase
from mem0.configs.base import MemoryConfig
# Fix import path for VectorStoreConfig
from mem0.vector_stores.configs import VectorStoreConfig
# Fix imports for Memory classes
from mem0.memory.main import Memory  # Changed from mem0.memory.base
from mem0.memory.graph_memory_kuzu import KuzuMemoryGraph
from mem0.vector_stores.kuzu import KuzuVectorStore
from mem0.utils.kuzu_connection import KuzuConnectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the test database."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns fixed embeddings."""
    class MockEmbedder:
        def embed(self, text):
            # Return a deterministic embedding based on text length
            # This ensures consistency in tests
            seed = sum(ord(c) for c in str(text))
            np.random.seed(seed)
            return list(np.random.random(1536).astype(float))
    
    return MockEmbedder()


@pytest.fixture
def mock_llm():
    """Mock LLM that returns fixed responses for entity extraction and relationships."""
    class MockLlm:
        def generate_response(self, messages=None, tools=None):
            # Return different responses based on the prompt content
            content = messages[1]["content"] if len(messages) > 1 else ""
            
            if "List of entities" in content or "Extract all the entities" in content:
                # Entity extraction
                return {
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
                }
            elif "sources" in content or "relationships" in content:
                # Relationship detection
                return {
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
            else:
                # Default empty response
                return {"tool_calls": []}
    
    return MockLlm()


@pytest.fixture
def vector_config_factory(temp_db_path):
    """Factory to create vector store configurations for different providers."""
    def _factory(provider, **kwargs):
        if provider == "kuzu":
            return VectorStoreConfig(
                provider=provider,
                config={
                    "db_path": temp_db_path,
                    "collection_name": kwargs.get("collection_name", "test_vectors")
                }
            )
        elif provider == "sqlite":
            return VectorStoreConfig(
                provider=provider,
                config={
                    "connection_string": ":memory:",
                    "collection_name": kwargs.get("collection_name", "test_vectors")
                }
            )
        else:
            # Default configuration for other providers
            return VectorStoreConfig(
                provider=provider,
                config=kwargs
            )
    
    return _factory


@pytest.fixture
def memory_config_factory(temp_db_path, mock_embedder, mock_llm):
    """Factory to create memory configurations for different providers."""
    # Patch the factory methods directly instead of using context manager
    mp = pytest.MonkeyPatch()
    mp.setattr(EmbedderFactory, "create", lambda *args, **kwargs: mock_embedder)
    mp.setattr(LlmFactory, "create", lambda *args, **kwargs: mock_llm)
    
    def _factory(provider, **kwargs):
        from mem0.configs.embedders import EmbeddersConfig
        from mem0.configs.llm import LlmConfig
        
        embedder_config = EmbeddersConfig(
            provider="mock",
            config={}
        )
        
        llm_config = LlmConfig(
            provider="mock",
            config={}
        )
        
        if provider == "kuzu":
            graph_store_config = {
                "provider": provider,
                "config": {"db_path": temp_db_path}
            }
        elif provider == "neo4j":
            graph_store_config = {
                "provider": provider,
                "config": {
                    "url": kwargs.get("url", "bolt://localhost:7687"),
                    "username": kwargs.get("username", "neo4j"),
                    "password": kwargs.get("password", "password")
                }
            }
        else:
            # Default configuration for other providers
            graph_store_config = {
                "provider": provider,
                "config": kwargs
            }
        
        # Create memory config
        memory_config = MemoryConfig(
            graph_store=graph_store_config,
            embedder=embedder_config,
            llm=llm_config
        )
        
        return memory_config
    
    yield _factory
    # Clean up patches
    mp.undo()


# Clear singleton instances before testing compatibility
@pytest.fixture(autouse=True)
def clear_singleton_instances():
    """Reset singleton instances before each test."""
    from mem0.utils.kuzu_connection import KuzuConnectionManager

def test_vector_store_factory_compatibility(vector_config_factory):
    """
    Test that the VectorStoreFactory can create different implementations
    and switch between them seamlessly.
    """
    # Use patch to ensure we don't actually create a database connection
    with patch("mem0.utils.kuzu_connection.KuzuConnectionManager") as mock_manager_cls:
        # Setup the mock
        mock_instance = Mock()
        mock_connection = Mock()
        mock_instance.get_connection.return_value = mock_connection
        mock_manager_cls.return_value = mock_instance
        
        # Configure for proper patching
        with patch.object(KuzuConnectionManager, "__new__", return_value=mock_instance):
            # Create configurations for different providers
            kuzu_config = vector_config_factory("kuzu")
            sqlite_config = vector_config_factory("sqlite")
            
            # Create instances
            kuzu_store = VectorStoreFactory.create(kuzu_config)
            sqlite_store = VectorStoreFactory.create(sqlite_config)
            
            # Verify instances are created with correct types
            assert isinstance(kuzu_store, KuzuVectorStore)
            assert isinstance(kuzu_store, VectorStoreBase)
            assert isinstance(sqlite_store, VectorStoreBase)
            
            # Test basic operations with both implementations
            test_vectors = [[0.1, 0.2, 0.3]]
            test_payload = [{"text": "test"}]
            test_id = ["test_id"]
            
            # Insert data with both implementations
            kuzu_store.insert(vectors=test_vectors, payloads=test_payload, ids=test_id)
            sqlite_store.insert(vectors=test_vectors, payloads=test_payload, ids=test_id)
            
            # Configure the mock result for get
            mock_result = Mock()
            mock_result.has_next.side_effect = [True, False]  # One result
            mock_result.get_next.return_value = {
                "id": "test_id",
                "payload": json.dumps({"text": "test"})
            }
            mock_connection.execute.return_value = mock_result
            
            # Verify data can be retrieved
            kuzu_result = kuzu_store.get(test_id[0])
            sqlite_result = sqlite_store.get(test_id[0])
            
            # Check results
            assert kuzu_result.id == test_id[0]
            assert sqlite_result.id == test_id[0]
            assert "text" in kuzu_result.payload
            assert "text" in sqlite_result.payload


def test_graph_memory_factory_compatibility(memory_config_factory):
    """
    Test that the GraphMemoryFactory can create different implementations
    and switch between them seamlessly.
    """
    # Use patch to mock KuzuConnectionManager
    with patch("mem0.utils.kuzu_connection.KuzuConnectionManager") as mock_manager_cls:
        # Setup the mock
        mock_instance = Mock()
        mock_connection = Mock()
        mock_instance.get_connection.return_value = mock_connection
        mock_manager_cls.return_value = mock_instance
        
        # Configure for proper patching
        with patch.object(KuzuConnectionManager, "__new__", return_value=mock_instance):
            # Create configurations for different providers
            kuzu_config = memory_config_factory("kuzu")
            
            # Try to create Neo4j config, but don't fail the test if Neo4j is not available
            try:
                neo4j_config = memory_config_factory("neo4j")
                neo4j_available = True
            except Exception:
                neo4j_available = False
            
            # Create Kuzu instance
            kuzu_memory = GraphMemoryFactory.create(kuzu_config)
            
            # Verify instance is created with correct type
            assert isinstance(kuzu_memory, KuzuMemoryGraph)
            
            # Configure the mock result for search
            mock_result = Mock()
            mock_result.has_next.side_effect = [True, False]  # One result
            mock_result.get_next.return_value = {
                "source": "user1", 
                "relationship": "works_on", 
                "destination": "project1",
                "similarity": 0.95
            }
            mock_connection.execute.return_value = mock_result
            
            # Test basic operations with Kuzu implementation
            test_data = "User1 is working on Project1"
            test_filters = {"user_id": "test_user"}
            
            # Add data
            kuzu_memory.add(data=test_data, filters=test_filters)
            
            # Search for data
            search_results = kuzu_memory.search(query="User1", filters=test_filters)
            
            # Verify search results
            assert len(search_results) > 0
            assert search_results[0]["source"] == "user1"
            assert search_results[0]["relationship"] == "works_on"
            assert search_results[0]["destination"] == "project1"


def test_memory_integration_compatibility(memory_config_factory):
    """
    Test that the Memory class works with different graph memory implementations.
    """
    # Use patch to mock KuzuConnectionManager
    with patch("mem0.utils.kuzu_connection.KuzuConnectionManager") as mock_manager_cls:
        # Setup the mock
        mock_instance = Mock()
        mock_connection = Mock()
        mock_instance.get_connection.return_value = mock_connection
        mock_manager_cls.return_value = mock_instance
        
        # Configure for proper patching
        with patch.object(KuzuConnectionManager, "__new__", return_value=mock_instance):
            # Create configurations for different providers
            kuzu_config = memory_config_factory("kuzu")
            
            # Create Memory instance with Kuzu as graph store
            memory = Memory(config=kuzu_config)
            
            # Configure the mock result for search
            mock_result = Mock()
            mock_result.has_next.side_effect = [True, False]  # One result
            mock_result.get_next.return_value = {
                "source": "user1", 
                "relationship": "works_on", 
                "destination": "project1",
                "similarity": 0.95
            }
            mock_connection.execute.return_value = mock_result
            
            # Test basic operations
            test_data = "User1 is working on Project1"
            test_user_id = "test_user"
            
            # Add data
            memory.add(data=test_data, user_id=test_user_id)
            
            # Search for data
            search_results = memory.search(query="User1", user_id=test_user_id)
            
            # Verify search results
            assert len(search_results) > 0