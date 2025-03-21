import time
import uuid
import logging
import numpy as np
import pytest
import statistics
import json
from typing import Dict, List, Optional

from mem0.vector_stores.factory import VectorStoreFactory
from mem0.utils.factory import GraphMemoryFactory
from mem0.configs.base import Config
from mem0.configs.vector_stores import VectorStoreConfig
from mem0.configs.graph_store import GraphStoreConfig
from mem0.configs.embedders import EmbeddersConfig
from mem0.configs.llm import LlmConfig
from mem0.utils.kuzu_connection import KuzuConnectionManager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkResult:
    """Class to store and report benchmark results."""
    
    def __init__(self, name: str, implementation: str):
        """
        Initialize benchmark result.
        
        Args:
            name: Name of the benchmark
            implementation: Implementation being benchmarked
        """
        self.name = name
        self.implementation = implementation
        self.times = []
        self.operations = 0
        
    def add_time(self, time_taken: float, operations: int = 1):
        """
        Add a time measurement.
        
        Args:
            time_taken: Time taken in seconds
            operations: Number of operations performed
        """
        self.times.append(time_taken)
        self.operations += operations
        
    def summary(self) -> Dict:
        """
        Generate summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.times:
            return {
                "name": self.name,
                "implementation": self.implementation,
                "min": 0,
                "max": 0,
                "mean": 0,
                "median": 0,
                "total": 0,
                "operations": 0,
                "ops_per_sec": 0
            }
            
        total_time = sum(self.times)
        ops_per_sec = self.operations / total_time if total_time > 0 else 0
        
        return {
            "name": self.name,
            "implementation": self.implementation,
            "min": min(self.times),
            "max": max(self.times),
            "mean": statistics.mean(self.times),
            "median": statistics.median(self.times),
            "total": total_time,
            "operations": self.operations,
            "ops_per_sec": ops_per_sec
        }


class BenchmarkComparer:
    """Class to compare benchmark results between implementations."""
    
    def __init__(self, baseline_name: str):
        """
        Initialize benchmark comparer.
        
        Args:
            baseline_name: Name of the baseline implementation to compare against
        """
        self.baseline_name = baseline_name
        self.results = {}
        
    def add_result(self, result: BenchmarkResult):
        """
        Add a benchmark result.
        
        Args:
            result: BenchmarkResult to add
        """
        key = f"{result.name}:{result.implementation}"
        self.results[key] = result
        
    def compare(self, test_name: str, other_implementation: str) -> Dict:
        """
        Compare results between implementations.
        
        Args:
            test_name: Name of the test to compare
            other_implementation: Name of the implementation to compare against baseline
            
        Returns:
            Dictionary with comparison results
        """
        baseline_key = f"{test_name}:{self.baseline_name}"
        other_key = f"{test_name}:{other_implementation}"
        
        if baseline_key not in self.results or other_key not in self.results:
            return {
                "test": test_name,
                "baseline": self.baseline_name,
                "other": other_implementation,
                "comparison": "missing data",
                "speedup": 0
            }
            
        baseline = self.results[baseline_key].summary()
        other = self.results[other_key].summary()
        
        if baseline["ops_per_sec"] == 0:
            speedup = 0
        else:
            speedup = other["ops_per_sec"] / baseline["ops_per_sec"]
            
        return {
            "test": test_name,
            "baseline": self.baseline_name,
            "baseline_ops_per_sec": baseline["ops_per_sec"],
            "other": other_implementation,
            "other_ops_per_sec": other["ops_per_sec"],
            "speedup": speedup
        }


def generate_random_vector(dim: int = 1536) -> List[float]:
    """
    Generate a random vector of specified dimension.
    
    Args:
        dim: Dimension of the vector
        
    Returns:
        Random vector
    """
    return list(np.random.random(dim).astype(float))


def generate_vector_data(count: int, dim: int = 1536) -> tuple:
    """
    Generate random vector data for benchmarking.
    
    Args:
        count: Number of vectors to generate
        dim: Dimension of each vector
        
    Returns:
        Tuple of (vectors, payloads, ids)
    """
    vectors = [generate_random_vector(dim) for _ in range(count)]
    payloads = [{"text": f"test_data_{i}", "metadata": {"number": i}} for i in range(count)]
    ids = [str(uuid.uuid4()) for _ in range(count)]
    
    return vectors, payloads, ids


def generate_graph_data(count: int) -> List[str]:
    """
    Generate random graph data for benchmarking.
    
    Args:
        count: Number of data points to generate
        
    Returns:
        List of strings with entity relationships
    """
    entities = ["User", "Project", "Task", "Document", "Meeting"]
    relationships = ["owns", "works_on", "assigned_to", "related_to", "scheduled"]
    
    data = []
    for i in range(count):
        source = f"{np.random.choice(entities)}{i}"
        dest = f"{np.random.choice(entities)}{i+1}"
        rel = np.random.choice(relationships)
        data.append(f"{source} {rel} {dest}")
        
    return data


@pytest.fixture
def vector_store_implementations():
    """
    Create vector store implementations for benchmarking.
    
    Returns:
        Dictionary of vector store implementations
    """
    implementations = {}
    
    # Temporary paths for databases
    kuzu_path = f"./temp_kuzu_bench_{uuid.uuid4()}"
    
    # Create Kuzu implementation
    kuzu_config = VectorStoreConfig(
        provider="kuzu",
        config={"db_path": kuzu_path, "collection_name": "bench_vectors"}
    )
    implementations["kuzu"] = VectorStoreFactory.create(kuzu_config)
    
    # Create other implementation for comparison
    # For example, SQLite as an embedded alternative
    sqlite_config = VectorStoreConfig(
        provider="sqlite",
        config={"connection_string": ":memory:", "collection_name": "bench_vectors"}
    )
    
    try:
        implementations["sqlite"] = VectorStoreFactory.create(sqlite_config)
    except Exception as e:
        logger.warning(f"Could not create SQLite vector store: {e}")
    
    # Add more implementations as needed
    
    yield implementations
    
    # Cleanup
    for impl_name, impl in implementations.items():
        try:
            if hasattr(impl, "delete_col"):
                impl.delete_col()
        except Exception as e:
            logger.warning(f"Error cleaning up {impl_name}: {e}")


@pytest.fixture
def graph_memory_implementations():
    """
    Create graph memory implementations for benchmarking.
    
    Returns:
        Dictionary of graph memory implementations
    """
    implementations = {}
    
    # Create basic configuration
    embedder_config = EmbeddersConfig(
        provider="mock",
        config={}
    )
    
    llm_config = LlmConfig(
        provider="mock",
        config={}
    )
    
    # Kuzu configuration
    kuzu_path = f"./temp_kuzu_graph_bench_{uuid.uuid4()}"
    kuzu_config = Config(
        graph_store=GraphStoreConfig(
            provider="kuzu",
            config={"db_path": kuzu_path}
        ),
        embedder=embedder_config,
        llm=llm_config
    )
    
    # Create Kuzu implementation
    try:
        implementations["kuzu"] = GraphMemoryFactory.create(kuzu_config)
    except Exception as e:
        logger.warning(f"Could not create Kuzu graph memory: {e}")
    
    # Create Neo4j implementation for comparison if available
    try:
        neo4j_config = Config(
            graph_store=GraphStoreConfig(
                provider="neo4j",
                config={
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password"
                }
            ),
            embedder=embedder_config,
            llm=llm_config
        )
        implementations["neo4j"] = GraphMemoryFactory.create(neo4j_config)
    except Exception as e:
        logger.warning(f"Could not create Neo4j graph memory: {e}")
    
    yield implementations
    
    # Cleanup
    for impl_name, impl in implementations.items():
        try:
            impl.delete_all({"user_id": "benchmark_user"})
        except Exception as e:
            logger.warning(f"Error cleaning up {impl_name}: {e}")


def test_vector_store_insert_performance(vector_store_implementations):
    """
    Benchmark vector store insert performance.
    """
    comparer = BenchmarkComparer(baseline_name="sqlite")
    
    # Define test parameters
    batch_sizes = [10, 100, 1000]
    iterations = 3
    
    for impl_name, impl in vector_store_implementations.items():
        logger.info(f"Benchmarking insert for {impl_name}")
        
        for batch_size in batch_sizes:
            test_name = f"insert_batch_{batch_size}"
            result = BenchmarkResult(test_name, impl_name)
            
            for _ in range(iterations):
                vectors, payloads, ids = generate_vector_data(batch_size)
                
                start_time = time.time()
                impl.insert(vectors=vectors, payloads=payloads, ids=ids)
                end_time = time.time()
                
                result.add_time(end_time - start_time, batch_size)
            
            comparer.add_result(result)
            
            summary = result.summary()
            logger.info(f"{impl_name} {test_name}: {summary['ops_per_sec']:.2f} ops/sec")
    
    # Report comparisons
    for batch_size in batch_sizes:
        test_name = f"insert_batch_{batch_size}"
        comparison = comparer.compare(test_name, "kuzu")
        logger.info(f"Comparison - {test_name}: Kuzu is {comparison['speedup']:.2f}x faster than baseline")
        
    return comparer


def test_vector_store_search_performance(vector_store_implementations):
    """
    Benchmark vector store search performance.
    """
    comparer = BenchmarkComparer(baseline_name="sqlite")
    
    # Define test parameters
    data_size = 1000
    search_sizes = [1, 10, 50]
    iterations = 5
    
    # First populate the stores with test data
    vectors, payloads, ids = generate_vector_data(data_size)
    for impl_name, impl in vector_store_implementations.items():
        impl.insert(vectors=vectors, payloads=payloads, ids=ids)
    
    # Now benchmark search
    for impl_name, impl in vector_store_implementations.items():
        logger.info(f"Benchmarking search for {impl_name}")
        
        for search_size in search_sizes:
            test_name = f"search_limit_{search_size}"
            result = BenchmarkResult(test_name, impl_name)
            
            for _ in range(iterations):
                query_vector = generate_random_vector()
                
                start_time = time.time()
                impl.search(query="", vectors=[query_vector], limit=search_size)
                end_time = time.time()
                
                result.add_time(end_time - start_time, 1)
            
            comparer.add_result(result)
            
            summary = result.summary()
            logger.info(f"{impl_name} {test_name}: {summary['ops_per_sec']:.2f} ops/sec")
    
    # Report comparisons
    for search_size in search_sizes:
        test_name = f"search_limit_{search_size}"
        comparison = comparer.compare(test_name, "kuzu")
        logger.info(f"Comparison - {test_name}: Kuzu is {comparison['speedup']:.2f}x faster than baseline")
        
    return comparer


def test_vector_store_update_performance(vector_store_implementations):
    """
    Benchmark vector store update performance.
    """
    comparer = BenchmarkComparer(baseline_name="sqlite")
    
    # Define test parameters
    data_size = 500
    update_counts = [1, 10, 50]
    iterations = 3
    
    # First populate the stores with test data
    vectors, payloads, ids = generate_vector_data(data_size)
    for impl_name, impl in vector_store_implementations.items():
        impl.insert(vectors=vectors, payloads=payloads, ids=ids)
    
    # Now benchmark update
    for impl_name, impl in vector_store_implementations.items():
        logger.info(f"Benchmarking update for {impl_name}")
        
        for update_count in update_counts:
            test_name = f"update_count_{update_count}"
            result = BenchmarkResult(test_name, impl_name)
            
            for _ in range(iterations):
                # Select random IDs to update
                update_ids = ids[:update_count]
                
                start_time = time.time()
                for vector_id in update_ids:
                    new_vector = generate_random_vector()
                    new_payload = {"text": f"updated_{vector_id}", "updated": True}
                    impl.update(vector_id=vector_id, vector=new_vector, payload=new_payload)
                end_time = time.time()
                
                result.add_time(end_time - start_time, update_count)
            
            comparer.add_result(result)
            
            summary = result.summary()
            logger.info(f"{impl_name} {test_name}: {summary['ops_per_sec']:.2f} ops/sec")
    
    # Report comparisons
    for update_count in update_counts:
        test_name = f"update_count_{update_count}"
        comparison = comparer.compare(test_name, "kuzu")
        logger.info(f"Comparison - {test_name}: Kuzu is {comparison['speedup']:.2f}x faster than baseline")
        
    return comparer


def test_vector_store_delete_performance(vector_store_implementations):
    """
    Benchmark vector store delete performance.
    """
    comparer = BenchmarkComparer(baseline_name="sqlite")
    
    # Define test parameters
    delete_counts = [1, 10, 50]
    iterations = 3
    
    for delete_count in delete_counts:
        # Create fresh data for each delete count
        data_size = 100 * delete_count
        vectors, payloads, ids = generate_vector_data(data_size)
        
        for impl_name, impl in vector_store_implementations.items():
            # Insert data for this implementation
            impl.insert(vectors=vectors, payloads=payloads, ids=ids)
            
            test_name = f"delete_count_{delete_count}"
            result = BenchmarkResult(test_name, impl_name)
            
            for _ in range(iterations):
                # Select random IDs to delete
                delete_batch = ids[:delete_count]
                
                start_time = time.time()
                for vector_id in delete_batch:
                    impl.delete(vector_id=vector_id)
                end_time = time.time()
                
                result.add_time(end_time - start_time, delete_count)
                
                # Remove the deleted IDs from our list for next iteration
                ids = ids[delete_count:]
            
            comparer.add_result(result)
            
            summary = result.summary()
            logger.info(f"{impl_name} {test_name}: {summary['ops_per_sec']:.2f} ops/sec")
    
    # Report comparisons
    for delete_count in delete_counts:
        test_name = f"delete_count_{delete_count}"
        comparison = comparer.compare(test_name, "kuzu")
        logger.info(f"Comparison - {test_name}: Kuzu is {comparison['speedup']:.2f}x faster than baseline")
        
    return comparer


def test_graph_memory_add_performance(graph_memory_implementations):
    """
    Benchmark graph memory add performance.
    """
    comparer = BenchmarkComparer(baseline_name="neo4j")
    
    # Define test parameters
    batch_sizes = [1, 5, 10]
    iterations = 3
    
    for impl_name, impl in graph_memory_implementations.items():
        logger.info(f"Benchmarking add for {impl_name}")
        
        for batch_size in batch_sizes:
            test_name = f"add_batch_{batch_size}"
            result = BenchmarkResult(test_name, impl_name)
            
            for _ in range(iterations):
                data_points = generate_graph_data(batch_size)
                
                start_time = time.time()
                for data in data_points:
                    impl.add(data=data, filters={"user_id": "benchmark_user"})
                end_time = time.time()
                
                result.add_time(end_time - start_time, batch_size)
            
            comparer.add_result(result)
            
            summary = result.summary()
            logger.info(f"{impl_name} {test_name}: {summary['ops_per_sec']:.2f} ops/sec")
    
    # Report comparisons
    for batch_size in batch_sizes:
        test_name = f"add_batch_{batch_size}"
        
        # Check if Neo4j implementation is available
        if "neo4j" in graph_memory_implementations:
            comparison = comparer.compare(test_name, "kuzu")
            logger.info(f"Comparison - {test_name}: Kuzu is {comparison['speedup']:.2f}x faster than baseline")
        else:
            logger.warning("Neo4j implementation not available for comparison")
        
    return comparer


def test_graph_memory_search_performance(graph_memory_implementations):
    """
    Benchmark graph memory search performance.
    """
    comparer = BenchmarkComparer(baseline_name="neo4j")
    
    # Define test parameters
    data_size = 20
    iterations = 5
    search_queries = ["User works_on Project", "Task assigned_to User", "Document related_to Meeting"]
    
    # First populate the memories with test data
    data_points = generate_graph_data(data_size)
    for impl_name, impl in graph_memory_implementations.items():
        for data in data_points:
            impl.add(data=data, filters={"user_id": "benchmark_user"})
    
    # Now benchmark search
    for impl_name, impl in graph_memory_implementations.items():
        logger.info(f"Benchmarking search for {impl_name}")
        
        for query in search_queries:
            test_name = f"search_query_{query.replace(' ', '_')}"
            result = BenchmarkResult(test_name, impl_name)
            
            for _ in range(iterations):
                start_time = time.time()
                impl.search(query=query, filters={"user_id": "benchmark_user"})
                end_time = time.time()
                
                result.add_time(end_time - start_time, 1)
            
            comparer.add_result(result)
            
            summary = result.summary()
            logger.info(f"{impl_name} {test_name}: {summary['ops_per_sec']:.2f} ops/sec")
    
    # Report comparisons
    for query in search_queries:
        test_name = f"search_query_{query.replace(' ', '_')}"
        
        # Check if Neo4j implementation is available
        if "neo4j" in graph_memory_implementations:
            comparison = comparer.compare(test_name, "kuzu")
            logger.info(f"Comparison - {test_name}: Kuzu is {comparison['speedup']:.2f}x faster than baseline")
        else:
            logger.warning("Neo4j implementation not available for comparison")
        
    return comparer


def test_graph_memory_traversal_performance(graph_memory_implementations):
    """
    Benchmark graph memory relationship traversal performance.
    """
    comparer = BenchmarkComparer(baseline_name="neo4j")
    
    # Define test parameters
    traversal_depths = [1, 2, 3]
    iterations = 3
    
    # Create a more deeply connected graph for traversal testing
    for impl_name, impl in graph_memory_implementations.items():
        # Clear existing data
        impl.delete_all({"user_id": "benchmark_user"})
        
        # Create interconnected entities (e.g., User1 -> Project1 -> Task1 -> Document1)
        for i in range(10):
            impl.add(data=f"User{i} owns Project{i}", filters={"user_id": "benchmark_user"})
            impl.add(data=f"Project{i} contains Task{i}", filters={"user_id": "benchmark_user"})
            impl.add(data=f"Task{i} requires Document{i}", filters={"user_id": "benchmark_user"})
            impl.add(data=f"Document{i} created_by User{i+1 if i < 9 else 0}", filters={"user_id": "benchmark_user"})
    
    # Now benchmark traversal (using get_all as a proxy for traversal)
    for impl_name, impl in graph_memory_implementations.items():
        logger.info(f"Benchmarking traversal for {impl_name}")
        
        for depth in traversal_depths:
            test_name = f"traversal_depth_{depth}"
            result = BenchmarkResult(test_name, impl_name)
            
            for _ in range(iterations):
                start_time = time.time()
                # Use get_all with increasing limits as a proxy for deeper traversals
                impl.get_all(filters={"user_id": "benchmark_user"}, limit=10*depth)
                end_time = time.time()
                
                result.add_time(end_time - start_time, 1)
            
            comparer.add_result(result)
            
            summary = result.summary()
            logger.info(f"{impl_name} {test_name}: {summary['ops_per_sec']:.2f} ops/sec")
    
    # Report comparisons
    for depth in traversal_depths:
        test_name = f"traversal_depth_{depth}"
        
        # Check if Neo4j implementation is available
        if "neo4j" in graph_memory_implementations:
            comparison = comparer.compare(test_name, "kuzu")
            logger.info(f"Comparison - {test_name}: Kuzu is {comparison['speedup']:.2f}x faster than baseline")
        else:
            logger.warning("Neo4j implementation not available for comparison")
        
    return comparer


def run_all_benchmarks():
    """
    Run all benchmarks and return results.
    """
    logger.info("Running all benchmarks...")
    
    vector_store_implementations = pytest.fixture(vector_store_implementations)()
    graph_memory_implementations = pytest.fixture(graph_memory_implementations)()
    
    results = {
        "vector_insert": test_vector_store_insert_performance(vector_store_implementations),
        "vector_search": test_vector_store_search_performance(vector_store_implementations),
        "vector_update": test_vector_store_update_performance(vector_store_implementations),
        "vector_delete": test_vector_store_delete_performance(vector_store_implementations),
        "graph_add": test_graph_memory_add_performance(graph_memory_implementations),
        "graph_search": test_graph_memory_search_performance(graph_memory_implementations),
        "graph_traversal": test_graph_memory_traversal_performance(graph_memory_implementations)
    }
    
    logger.info("Benchmarks complete!")
    
    return results


if __name__ == "__main__":
    """
    Run all benchmarks when script is executed directly.
    """
    run_all_benchmarks()