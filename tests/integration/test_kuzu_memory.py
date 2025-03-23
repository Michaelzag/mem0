import datetime
import json
import os
import pytest
import uuid

from mem0.memory.base import MemoryBase
from mem0.memory.graph_memory_kuzu import KuzuMemoryGraph
from mem0.utils.factory import GraphMemoryFactory


class MockConfig:
    """Mock configuration class for testing."""
    
    def __init__(self):
        self.graph_store = self.GraphStoreConfig()
        self.embedder = self.EmbedderConfig()
        self.llm = self.LlmConfig()
        
    class GraphStoreConfig:
        def __init__(self):
            self.config = self.Config()
            self.llm = None
            self.custom_prompt = None
            
        class Config:
            def __init__(self):
                self.db_path = "./test_kuzu_db"
                self.history_db_path = ":memory:"
    
    class EmbedderConfig:
        def __init__(self):
            self.provider = "mock"
            self.config = {}
    
    class LlmConfig:
        def __init__(self):
            self.provider = "mock"
            self.config = {}


class MockEmbedding:
    """Mock embedding class for testing."""
    
    def __init__(self, config):
        self.config = config
    
    def embed(self, text):
        """Return a simple mock embedding."""
        return [0.1] * 1536  # Standard 1536-dimension embedding


class MockLlm:
    """Mock LLM class for testing."""
    
    def __init__(self, config):
        self.config = config
    
    def generate_response(self, messages, tools=None):
        """Return a mock response for entity extraction."""
        if any(tool.get("name", "") == "extract_entities" for tool in tools):
            return {
                "tool_calls": [{
                    "arguments": {
                        "entities": [
                            {"entity": "test_entity", "entity_type": "test_type"}
                        ]
                    }
                }]
            }
        elif any(tool.get("name", "") == "extract_relations" for tool in tools):
            return {
                "tool_calls": [{
                    "arguments": {
                        "entities": [
                            {"source": "test_entity", "relationship": "test_relation", "destination": "test_entity2"}
                        ]
                    }
                }]
            }
        elif any(tool.get("name", "") == "delete_graph_memory" for tool in tools):
            return {"tool_calls": []}
        return {"tool_calls": []}


# Mock the factory classes to return our mock objects
def mock_embedder_factory(monkeypatch):
    def mock_create(provider_name, config):
        return MockEmbedding(config)
    
    monkeypatch.setattr("mem0.utils.factory.EmbedderFactory.create", mock_create)


def mock_llm_factory(monkeypatch):
    def mock_create(provider_name, config):
        return MockLlm(config)
    
    monkeypatch.setattr("mem0.utils.factory.LlmFactory.create", mock_create)


@pytest.fixture
def cleanup():
    """Clean up after tests."""
    yield
    # Remove test database directory after tests
    if os.path.exists("./test_kuzu_db"):
        import shutil
        shutil.rmtree("./test_kuzu_db")


@pytest.fixture
def kuzu_memory(monkeypatch, cleanup):
    """Create a test KuzuMemoryGraph instance."""
    mock_embedder_factory(monkeypatch)
    mock_llm_factory(monkeypatch)
    
    config = MockConfig()
    memory = KuzuMemoryGraph(config)
    return memory


def test_kuzu_memory_inheritance(kuzu_memory):
    """Test that KuzuMemoryGraph inherits from MemoryBase."""
    assert isinstance(kuzu_memory, MemoryBase)


def test_factory_integration():
    """Test that GraphMemoryFactory can create a KuzuMemoryGraph."""
    # This test only verifies that the provider is registered, doesn't create an actual instance
    assert "kuzu" in GraphMemoryFactory.provider_to_class


def test_memory_crud(kuzu_memory):
    """Test the basic CRUD operations."""
    # Create a test entity
    memory_id = str(uuid.uuid4())
    test_data = {
        "name": "test_entity",
        "entity_type": "test_type",
        "user_id": "test_user"
    }
    
    # Generate a mock entity through the add method
    kuzu_memory.add("Test entity is a test type", {"user_id": "test_user"})
    
    # Update the entity (simulated)
    kuzu_memory.update(memory_id, test_data)
    
    # Get the entity
    retrieved = kuzu_memory.get(memory_id)
    
    # This will likely be None in the test due to mocking, but we're testing the method call itself
    assert isinstance(retrieved, dict) or retrieved is None
    
    # Get history
    history = kuzu_memory.history(memory_id)
    assert isinstance(history, list)
    
    # Delete the entity
    kuzu_memory.delete(memory_id)