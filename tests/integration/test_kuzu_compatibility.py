import os
import shutil
import tempfile
import pytest
import json
import numpy as np
from typing import Dict, List, Optional, Any
import logging

from mem0.vector_stores.factory import VectorStoreFactory
from mem0.vector_stores.base import VectorStoreBase
from mem0.utils.factory import GraphMemoryFactory, EmbedderFactory, LlmFactory
from mem0.configs.base import Config
from mem0.configs.vector_stores import VectorStoreConfig
from mem0.configs.graph_store import GraphStoreConfig
from mem0.configs.embedders import EmbeddersConfig
from mem0.configs.llm import LlmConfig
from mem0.memory.base import Memory
from mem0.memory import MemoryGraph
from mem0.memory.graph_memory_kuzu import KuzuMemoryGraph
from mem0.vector_stores.kuzu import KuzuVectorStore

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
    # Mock the factory create methods
    with pytest.MonkeyPatch() as mp:
        mp.setattr(EmbedderFactory, "create", lambda *args, **kwargs: mock_embedder)
        mp.setattr(LlmFactory, "create", lambda *args, **kwargs: mock_llm)
        
        def _factory(provider, **kwargs):
            embedder_config = EmbeddersConfig(
                provider="mock",
                config={}
            )
            
            llm_config = LlmConfig(
                provider="mock",
                config={}
            )
            
            if provider == "kuzu":
                return Config(
                    graph_store=GraphStoreConfig(
                        provider=provider,
                        config={"db_path": temp_db_path}
                    ),
                    embedder=embedder_config,
                    llm=llm_config
                )
            elif provider == "neo4j":
                return Config(
                    graph_store=GraphStoreConfig(
                        provider=provider,
                        config={
                            "url": kwargs.get("url", "bolt://localhost:7687"),
                            "username": kwargs.get("username", "neo4j"),
                            "password": kwargs.get("password", "password")
                        }
                    ),
                    embedder=embedder_config,
                    llm=llm_config
                )
            else:
                # Default configuration for other providers
                return Config(
                    graph_store=GraphStoreConfig(
                        provider=provider,
                        config=kwargs
                    ),
                    embedder=embedder_config,
                    llm=llm_config
                )
        
        yield _factory


def test_vector_store_factory_compatibility(vector_config_factory):
    """
    Test that the VectorStoreFactory can create different implementations
    and switch between them seamlessly.
    """
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
    
    # Verify data can be retrieved
    kuzu_result = kuzu_store.get(test_id[0])
    sqlite_result = sqlite_store.get(test_id[0])
    
    # Check results (different implementations may return slightly different structures,
    # but both should have the ID and payload)
    assert kuzu_result.id == test_id[0]
    assert sqlite_result.id == test_id[0]
    assert "text" in kuzu_result.payload
    assert "text" in sqlite_result.payload


def test_graph_memory_factory_compatibility(memory_config_factory):
    """
    Test that the GraphMemoryFactory can create different implementations
    and switch between them seamlessly.
    """
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
    
    # Create Neo4j instance if available
    if neo4j_available:
        neo4j_memory = GraphMemoryFactory.create(neo4j_config)
        assert isinstance(neo4j_memory, MemoryGraph)
    
    # Test basic operations with Kuzu implementation
    test_data = "User1 is working on Project1"
    test_filters = {"user_id": "test_user"}
    
    # Add data
    kuzu_memory.add(data=test_data, filters=test_filters)
    
    # Search for data
    search_results = kuzu_memory.search(query="User1", filters=test_filters)
    
    # Verify search results
    assert len(search_results) > 0
    assert "user1" in [item["source"].lower() for item in search_results] or \
           "user1" in [item["destination"].lower() for item in search_results]


def test_memory_integration_compatibility(memory_config_factory):
    """
    Test that the Memory class works with different graph memory implementations.
    """
    # Create configurations for different providers
    kuzu_config = memory_config_factory("kuzu")
    
    # Create Memory instance with Kuzu as graph store
    memory = Memory(config=kuzu_config)
    
    # Verify Memory instance is using KuzuMemoryGraph
    assert isinstance(memory.graph_memory, KuzuMemoryGraph)
    
    # Test basic operations
    test_data = "User1 is working on Project1"
    test_user_id = "test_user"
    
    # Add data
    memory.add(data=test_data, user_id=test_user_id)
    
    # Search for data
    search_results = memory.search(query="User1", user_id=test_user_id)
    
    # Verify search results
    assert len(search_results) > 0


def test_configuration_switching(vector_config_factory, temp_db_path):
    """
    Test switching between configurations for vector stores.
    """
    # Create test data
    test_vectors = [[0.1, 0.2, 0.3]]
    test_payload = [{"text": "test_configuration_switching"}]
    test_id = ["switching_test_id"]
    
    # Create and use KuzuVectorStore
    kuzu_config = vector_config_factory("kuzu")
    kuzu_store = VectorStoreFactory.create(kuzu_config)
    kuzu_store.insert(vectors=test_vectors, payloads=test_payload, ids=test_id)
    
    # Create and use SQLite vector store
    sqlite_config = vector_config_factory("sqlite")
    sqlite_store = VectorStoreFactory.create(sqlite_config)
    sqlite_store.insert(vectors=test_vectors, payloads=test_payload, ids=test_id)
    
    # Switch back to Kuzu and verify data is still accessible
    kuzu_store2 = VectorStoreFactory.create(kuzu_config)
    kuzu_result = kuzu_store2.get(test_id[0])
    assert kuzu_result.id == test_id[0]
    assert kuzu_result.payload["text"] == "test_configuration_switching"
    
    # Switch back to SQLite and verify data is still accessible
    sqlite_store2 = VectorStoreFactory.create(sqlite_config)
    sqlite_result = sqlite_store2.get(test_id[0])
    assert sqlite_result.id == test_id[0]
    assert sqlite_result.payload["text"] == "test_configuration_switching"


def test_real_world_compatibility(temp_db_path, mock_embedder, mock_llm):
    """
    Test a real-world workflow combining vector and graph operations.
    """
    # Set up configurations
    collection_name = "real_world_test"
    
    # Create vector store configs
    kuzu_vector_config = VectorStoreConfig(
        provider="kuzu",
        config={
            "db_path": temp_db_path,
            "collection_name": collection_name,
            "vector_dimension": 1536
        }
    )
    
    # Create graph memory configs
    embedder_config = EmbeddersConfig(provider="mock", config={})
    llm_config = LlmConfig(provider="mock", config={})
    
    kuzu_graph_config = Config(
        graph_store=GraphStoreConfig(
            provider="kuzu",
            config={"db_path": temp_db_path}
        ),
        embedder=embedder_config,
        llm=llm_config
    )
    
    # Mock the factory create methods
    with pytest.MonkeyPatch() as mp:
        mp.setattr(EmbedderFactory, "create", lambda *args, **kwargs: mock_embedder)
        mp.setattr(LlmFactory, "create", lambda *args, **kwargs: mock_llm)
        
        # Create vector store
        vector_store = VectorStoreFactory.create(kuzu_vector_config)
        
        # Create graph memory
        graph_memory = GraphMemoryFactory.create(kuzu_graph_config)
        
        # Create combined Memory instance
        memory = Memory(config=kuzu_graph_config)
        
        # Test data
        user_id = "real_world_user"
        documents = [
            {"title": "Project Roadmap", "content": "Project1 has the following milestones..."},
            {"title": "Team Assignment", "content": "User1 is assigned to Project1 for development."},
            {"title": "Meeting Notes", "content": "User1 presented progress on Project1 implementation."}
        ]
        
        # Step 1: Store document vectors
        for i, doc in enumerate(documents):
            # Generate vector using mock embedder
            vector = mock_embedder.embed(doc["content"])
            
            # Store in vector store
            vector_store.insert(
                vectors=[vector],
                payloads=[{"title": doc["title"], "content": doc["content"]}],
                ids=[f"doc_{i}"]
            )
        
        # Step 2: Add relationships to graph
        for doc in documents:
            graph_memory.add(data=doc["content"], filters={"user_id": user_id})
        
        # Step 3: Search vector store
        query_vector = mock_embedder.embed("Project1 updates")
        vector_results = vector_store.search(
            query="",
            vectors=[query_vector],
            limit=3
        )
        
        # Step 4: Search graph relationships
        graph_results = graph_memory.search(
            query="User1 Project1",
            filters={"user_id": user_id}
        )
        
        # Verify combined workflow results
        assert len(vector_results) > 0
        assert len(graph_results) > 0
        
        # Step 5: Get all graph data
        all_relationships = graph_memory.get_all(filters={"user_id": user_id})
        assert len(all_relationships) > 0


def test_concurrent_operations(temp_db_path, mock_embedder, mock_llm):
    """
    Test concurrent operations using the KuzuConnectionManager.
    """
    # Set up configurations
    collection_name = "concurrent_test"
    
    # Create vector store config
    vector_config = VectorStoreConfig(
        provider="kuzu",
        config={
            "db_path": temp_db_path,
            "collection_name": collection_name,
            "vector_dimension": 1536
        }
    )
    
    # Create graph memory config
    embedder_config = EmbeddersConfig(provider="mock", config={})
    llm_config = LlmConfig(provider="mock", config={})
    
    graph_config = Config(
        graph_store=GraphStoreConfig(
            provider="kuzu",
            config={"db_path": temp_db_path}
        ),
        embedder=embedder_config,
        llm=llm_config
    )
    
    # Mock the factory create methods
    with pytest.MonkeyPatch() as mp:
        mp.setattr(EmbedderFactory, "create", lambda *args, **kwargs: mock_embedder)
        mp.setattr(LlmFactory, "create", lambda *args, **kwargs: mock_llm)
        
        # Create instances
        vector_store = VectorStoreFactory.create(vector_config)
        graph_memory = GraphMemoryFactory.create(graph_config)
        
        # Verify they share the same connection manager
        assert vector_store.connection_manager.db_path == graph_memory.connection_manager.db_path
        
        # Test 1: Parallel operations
        # Insert into vector store
        vector_store.insert(
            vectors=[[0.1, 0.2, 0.3]],
            payloads=[{"text": "concurrent operation test"}],
            ids=["concurrent_id"]
        )
        
        # Add to graph memory
        graph_memory.add(
            data="User1 is working concurrently on Project1",
            filters={"user_id": "concurrent_user"}
        )
        
        # Verify both operations succeeded
        vector_result = vector_store.get("concurrent_id")
        assert vector_result.id == "concurrent_id"
        
        graph_results = graph_memory.search(
            query="User1",
            filters={"user_id": "concurrent_user"}
        )
        assert len(graph_results) > 0
        
        # Test 2: Transaction isolation
        # Begin a transaction in vector store but don't commit
        vector_store.connection_manager.begin_transaction()
        
        # Try to use graph memory in a different transaction
        # This should work because we're using the same connection manager
        try:
            graph_memory.add(
                data="User2 is isolated from User1",
                filters={"user_id": "concurrent_user"}
            )
            transaction_isolation = True
        except Exception:
            transaction_isolation = False
        
        # Commit the pending transaction
        vector_store.connection_manager.commit()
        
        # Verify transaction isolation behavior
        # In Kuzu, the second operation should have waited for the first transaction to complete
        assert transaction_isolation, "Transaction isolation test failed"


def test_large_dataset_operations(temp_db_path, mock_embedder, mock_llm):
    """
    Test operations with larger datasets.
    """
    # Set up configurations
    collection_name = "large_dataset_test"
    
    # Create vector store config
    vector_config = VectorStoreConfig(
        provider="kuzu",
        config={
            "db_path": temp_db_path,
            "collection_name": collection_name,
            "vector_dimension": 1536
        }
    )
    
    # Create graph memory config
    embedder_config = EmbeddersConfig(provider="mock", config={})
    llm_config = LlmConfig(provider="mock", config={})
    
    graph_config = Config(
        graph_store=GraphStoreConfig(
            provider="kuzu",
            config={"db_path": temp_db_path}
        ),
        embedder=embedder_config,
        llm=llm_config
    )
    
    # Mock the factory create methods
    with pytest.MonkeyPatch() as mp:
        mp.setattr(EmbedderFactory, "create", lambda *args, **kwargs: mock_embedder)
        mp.setattr(LlmFactory, "create", lambda *args, **kwargs: mock_llm)
        
        # Create instances
        vector_store = VectorStoreFactory.create(vector_config)
        graph_memory = GraphMemoryFactory.create(graph_config)
        
        # Test with larger vector dataset (100 vectors)
        batch_size = 100
        vectors = []
        payloads = []
        ids = []
        
        for i in range(batch_size):
            vectors.append([float(i)/batch_size] * 1536)  # Simple vector
            payloads.append({"id": i, "text": f"Large dataset item {i}"})
            ids.append(f"large_id_{i}")
        
        # Insert in batches
        for i in range(0, batch_size, 10):
            end = min(i + 10, batch_size)
            vector_store.insert(
                vectors=vectors[i:end],
                payloads=payloads[i:end],
                ids=ids[i:end]
            )
        
        # Verify we can search the large dataset
        random_vector = [0.5] * 1536
        search_results = vector_store.search(
            query="",
            vectors=[random_vector],
            limit=5
        )
        assert len(search_results) == 5
        
        # Test with larger graph dataset (50 relationships)
        user_id = "large_graph_user"
        entities = ["User", "Project", "Task", "Document", "Meeting"]
        relationships = ["works_on", "contains", "assigned_to", "related_to", "scheduled"]
        
        for i in range(50):
            source = f"{np.random.choice(entities)}{i}"
            dest = f"{np.random.choice(entities)}{i+1}"
            rel = np.random.choice(relationships)
            graph_memory.add(
                data=f"{source} {rel} {dest}",
                filters={"user_id": user_id}
            )
        
        # Verify we can search the large graph
        graph_results = graph_memory.search(
            query="User",
            filters={"user_id": user_id},
            limit=10
        )
        assert len(graph_results) > 0
        
        # Verify we can retrieve all relationships
        all_relationships = graph_memory.get_all(
            filters={"user_id": user_id},
            limit=100
        )
        assert len(all_relationships) > 0


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])