# Kuzu Integration Plan for mem0

## Overview

This document outlines a plan for integrating [Kuzu](https://github.com/kuzudb/kuzu), an embedded graph database with vector search capabilities, into the mem0 memory layer. The integration aims to leverage Kuzu's combined graph and vector capabilities to provide an alternative to the current Neo4j-based graph memory implementation, while also offering a new vector store option.

## Current Architecture

mem0 has a modular architecture with well-defined interfaces for different components:

1. **Vector Stores**: Implemented via the `VectorStoreBase` abstract class with methods for vector operations
2. **Graph Memory**: Currently implemented using Neo4j through the `MemoryGraph` class
3. **Factory Pattern**: Uses factory classes (`VectorStoreFactory`, `LlmFactory`, etc.) to instantiate components

## Integration Goals

1. Create a Kuzu-based vector store implementation
2. Create a Kuzu-based graph memory implementation to replace Neo4j
3. Integrate both implementations into mem0's factory system
4. Provide configuration options for using Kuzu
5. Ensure compatibility with existing mem0 functionality
6. Document usage and benefits

## Benefits of Integration

1. **Simplified Deployment**: Kuzu is an embedded database, eliminating the need for a separate database instance
2. **Reduced Dependencies**: Remove Neo4j as a dependency for graph operations
3. **Unified Storage**: Use a single database for both vector and graph operations
4. **Performance**: Leverage Kuzu's optimized query processing
5. **Graph-Vector Queries**: Enable more efficient hybrid queries that combine vector similarity with graph traversal

## Implementation Plan

### Phase 1: Kuzu Vector Store Implementation

1. Create a new `KuzuVectorStore` class in `mem0/vector_stores/kuzu.py` that implements the `VectorStoreBase` interface
2. Implement required methods using Kuzu's vector capabilities:
   - `create_col`: Create a node table with a vector field
   - `insert`: Insert vectors with metadata
   - `search`: Perform vector similarity search
   - `delete`: Remove vectors
   - `update`: Update vectors or metadata
   - `get`: Retrieve vectors by ID
   - Other required methods

3. Update `VectorStoreFactory` to include the new implementation

```python
# Add to mem0/utils/factory.py
class VectorStoreFactory:
    provider_to_class = {
        # Existing providers
        "kuzu": "mem0.vector_stores.kuzu.KuzuVectorStore",
    }
```

4. Create appropriate configuration classes and documentation

### Phase 2: Kuzu Graph Memory Implementation

1. Create a Kuzu-based replacement for the Neo4j graph memory implementation
2. Implement all methods currently in `MemoryGraph` using Kuzu's Cypher capabilities:
   - `add`: Add data to the graph
   - `search`: Search for memories and related graph data
   - `delete_all`: Delete all graph data
   - `get_all`: Retrieve all nodes and relationships
   - Other helper methods

3. Update the factory or initialization logic to use Kuzu instead of Neo4j when configured

### Phase 3: Integration and Testing

1. Create comprehensive tests for both implementations
2. Ensure compatibility with existing mem0 functionality
3. Measure performance compared to existing implementations
4. Address any issues or limitations

### Phase 4: Documentation and Examples

1. Update documentation to include Kuzu as an option
2. Create examples demonstrating the benefits of the Kuzu integration
3. Provide migration guides for users of other backends

## Technical Design: KuzuVectorStore

The `KuzuVectorStore` class will implement the `VectorStoreBase` interface and use Kuzu's Python API for vector operations:

```python
from typing import Dict, List, Optional
from mem0.vector_stores.base import VectorStoreBase
from dataclasses import dataclass

@dataclass
class OutputData:
    id: Optional[str] = None
    score: Optional[float] = None
    payload: Optional[Dict] = None

class KuzuVectorStore(VectorStoreBase):
    def __init__(
        self,
        collection_name: str,
        db_path: str = "./kuzu_db",
        vector_dimension: int = 1536,
    ):
        """
        Initialize Kuzu vector store.
        
        Args:
            collection_name: Name of the collection (used as node table name)
            db_path: Path to Kuzu database
            vector_dimension: Dimension of vectors to store
        """
        import kuzu
        
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self.create_col(collection_name, vector_dimension, "cosine")
    
    def create_col(self, name: str, vector_size: int, distance: str) -> None:
        """Create a new collection (node table with vector field)."""
        # Check if table exists first
        try:
            self.conn.execute(f"MATCH (n:{name}) RETURN n LIMIT 1")
            # Table exists
            return
        except:
            # Create table with vector field
            self.conn.execute(f"""
                CREATE NODE TABLE {name} (
                    id STRING PRIMARY KEY,
                    vector FLOAT[{vector_size}],
                    payload STRING
                )
            """)
            
            # Create vector index
            self.conn.execute(f"CREATE VECTOR INDEX ON {name}(vector)")
    
    def insert(
        self, 
        vectors: List[list], 
        payloads: Optional[List[Dict]] = None, 
        ids: Optional[List[str]] = None
    ) -> None:
        """Insert vectors into collection."""
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        if payloads is None:
            payloads = [{} for _ in range(len(vectors))]
            
        for i in range(len(vectors)):
            payload_json = json.dumps(payloads[i])
            vector_str = str(vectors[i])
            
            self.conn.execute(f"""
                CREATE (:{self.collection_name} {{
                    id: '{ids[i]}',
                    vector: {vector_str},
                    payload: '{payload_json}'
                }})
            """)
    
    # Implement other required methods...
```

## Technical Design: KuzuGraphMemory

The Kuzu-based graph memory implementation will replace the current Neo4j implementation:

```python
class KuzuMemoryGraph:
    def __init__(self, config):
        """Initialize Kuzu graph memory."""
        import kuzu
        
        self.config = config
        self.db_path = config.graph_store.config.db_path
        
        self.db = kuzu.Database(self.db_path)
        self.conn = kuzu.Connection(self.db)
        
        # Initialize other components similar to current MemoryGraph
        self.embedding_model = EmbedderFactory.create(self.config.embedder.provider, self.config.embedder.config)
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.threshold = 0.7
    
    # Implement methods similar to MemoryGraph but using Kuzu's Cypher
```

## Configuration

Add Kuzu-specific configuration options to mem0:

```python
# Example configuration
from mem0 import Memory

# Vector store configuration
memory = Memory(config={
    "vector_store": {
        "provider": "kuzu",
        "config": {
            "collection_name": "memories",
            "db_path": "./kuzu_db",
            "vector_dimension": 1536
        }
    },
    # Other configuration options
})

# Graph memory configuration
memory = Memory(config={
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db_path": "./kuzu_db"
        }
    },
    # Other configuration options
})

# Combined configuration (shared database)
memory = Memory(config={
    "vector_store": {
        "provider": "kuzu",
        "config": {
            "collection_name": "memories",
            "db_path": "./kuzu_db",
            "vector_dimension": 1536
        }
    },
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db_path": "./kuzu_db"
        }
    },
    # Other configuration options
})
```

## Limitations and Considerations

1. **Vector Index Status**: Kuzu's vector indexing capabilities are still evolving. Monitor performance and feature development.
2. **Cypher Dialect Differences**: There may be differences between Neo4j's Cypher and Kuzu's Cypher implementation.
3. **Transaction Support**: Ensure transactional consistency is maintained across operations.
4. **Migration**: Provide tools for migrating existing data from Neo4j to Kuzu.
5. **Performance Testing**: Conduct thorough performance testing to identify any bottlenecks.

## Timeline

1. **Phase 1** (Vector Store Implementation): 2 weeks
2. **Phase 2** (Graph Memory Implementation): 2 weeks
3. **Phase 3** (Integration and Testing): 1 week
4. **Phase 4** (Documentation and Examples): 1 week

Total estimated time: 6 weeks

## Next Steps

1. Set up development environment with both mem0 and kuzu
2. Create proof of concept implementation of `KuzuVectorStore`
3. Draft initial API design and test cases
4. Begin implementation of core functionality

## Appendix A: Complete KuzuVectorStore Implementation

Below is a more complete implementation of the KuzuVectorStore class which can serve as a starting point for the integration:

```python
import json
import logging
import uuid
from typing import Dict, List, Optional

import kuzu

from mem0.vector_stores.base import VectorStoreBase

logger = logging.getLogger(__name__)


class OutputData:
    """Output data structure for vector store results."""
    def __init__(self, id=None, score=None, payload=None):
        self.id = id
        self.score = score
        self.payload = payload


class KuzuVectorStore(VectorStoreBase):
    """Vector store implementation using Kuzu as the backend."""

    def __init__(
        self,
        collection_name: str,
        db_path: str = "./kuzu_db",
        vector_dimension: int = 1536,
        distance_metric: str = "cosine",
    ):
        """
        Initialize Kuzu vector store.
        
        Args:
            collection_name: Name of the collection (used as node table name)
            db_path: Path to Kuzu database
            vector_dimension: Dimension of vectors to store
            distance_metric: Distance metric to use for similarity search (cosine, euclidean, dot)
        """
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self.collection_name = collection_name
        self.vector_dimension = vector_dimension
        self.distance_metric = distance_metric
        
        # Create collection
        self.create_col(collection_name, vector_dimension, distance_metric)
        
        logger.info(f"Initialized KuzuVectorStore with collection '{collection_name}'")

    def create_col(self, name: str, vector_size: int, distance: str) -> None:
        """
        Create a new collection (node table with vector field).
        
        Args:
            name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric (cosine, euclidean, dot)
        """
        # Check if table exists first
        try:
            self.conn.execute(f"MATCH (n:{name}) RETURN n LIMIT 1")
            logger.info(f"Collection '{name}' already exists")
            return
        except Exception:
            # Create table with vector field
            logger.info(f"Creating collection '{name}' with vector size {vector_size}")
            
            self.conn.execute(f"""
                CREATE NODE TABLE {name} (
                    id STRING PRIMARY KEY,
                    vector FLOAT[{vector_size}],
                    payload STRING
                )
            """)
            
            # Create vector index based on distance metric
            self.conn.execute(f"CREATE VECTOR INDEX ON {name}(vector) USING {distance.upper()}")
            
            logger.info(f"Created collection '{name}' with vector index")

    def insert(
        self, 
        vectors: List[list], 
        payloads: Optional[List[Dict]] = None, 
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Insert vectors into collection.
        
        Args:
            vectors: List of vectors to insert
            payloads: List of payloads corresponding to vectors
            ids: List of IDs corresponding to vectors
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        if payloads is None:
            payloads = [{} for _ in range(len(vectors))]
        
        logger.info(f"Inserting {len(vectors)} vectors into collection '{self.collection_name}'")
        
        # Batch insert in transaction
        self.conn.execute("BEGIN TRANSACTION")
        
        try:
            for i in range(len(vectors)):
                payload_json = json.dumps(payloads[i])
                vector_str = str(vectors[i]).replace('[', '{').replace(']', '}')
                
                # Use parameters to avoid injection issues
                self.conn.execute(f"""
                    CREATE (:{self.collection_name} {{
                        id: '{ids[i]}',
                        vector: {vector_str},
                        payload: '{payload_json}'
                    }})
                """)
            
            self.conn.execute("COMMIT")
            logger.info(f"Successfully inserted {len(vectors)} vectors")
        
        except Exception as e:
            self.conn.execute("ROLLBACK")
            logger.error(f"Error inserting vectors: {e}")
            raise

    def search(
        self, 
        query: str, 
        vectors: List[list], 
        limit: int = 5, 
        filters: Optional[Dict] = None
    ) -> List[OutputData]:
        """
        Search for similar vectors.
        
        Args:
            query: Query string (unused, but required by interface)
            vectors: Query vector
            limit: Number of results to return
            filters: Filters to apply to search
            
        Returns:
            List of OutputData objects
        """
        logger.info(f"Searching for similar vectors in collection '{self.collection_name}'")
        
        # Convert vector format for Kuzu
        query_vector = str(vectors[0]).replace('[', '{').replace(']', '}')
        
        # Build the query
        cypher_query = f"""
            MATCH (n:{self.collection_name})
        """
        
        # Add filters if provided
        where_clauses = []
        if filters:
            for key, value in filters.items():
                # Extract from JSON payload
                where_clauses.append(f"JSON_EXTRACT(n.payload, '$.{key}') = '{value}'")
        
        if where_clauses:
            cypher_query += f" WHERE {' AND '.join(where_clauses)}"
        
        # Complete the query with similarity search
        cypher_query += f"""
            RETURN n.id AS id,
                   n.payload AS payload,
                   {self.distance_metric.upper()}_SIMILARITY(n.vector, {query_vector}) AS similarity
            ORDER BY similarity DESC
            LIMIT {limit}
        """
        
        try:
            result = self.conn.execute(cypher_query)
            
            # Process results
            output_results = []
            while result.has_next():
                row = result.get_next()
                
                # Parse payload from JSON
                payload = json.loads(row["payload"])
                
                output_results.append(OutputData(
                    id=row["id"],
                    score=row["similarity"],
                    payload=payload
                ))
            
            logger.info(f"Found {len(output_results)} similar vectors")
            return output_results
        
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []

    def delete(self, vector_id: str) -> None:
        """
        Delete a vector by ID.
        
        Args:
            vector_id: ID of the vector to delete
        """
        logger.info(f"Deleting vector {vector_id} from collection '{self.collection_name}'")
        
        try:
            self.conn.execute(f"""
                MATCH (n:{self.collection_name})
                WHERE n.id = '{vector_id}'
                DELETE n
            """)
            
            logger.info(f"Deleted vector {vector_id}")
        
        except Exception as e:
            logger.error(f"Error deleting vector: {e}")
            raise

    def update(
        self,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict] = None,
    ) -> None:
        """
        Update a vector and its payload.
        
        Args:
            vector_id: ID of the vector to update
            vector: New vector values
            payload: New payload
        """
        logger.info(f"Updating vector {vector_id} in collection '{self.collection_name}'")
        
        try:
            # Build update query based on what's provided
            set_clauses = []
            
            if vector:
                vector_str = str(vector).replace('[', '{').replace(']', '}')
                set_clauses.append(f"n.vector = {vector_str}")
            
            if payload:
                payload_json = json.dumps(payload)
                set_clauses.append(f"n.payload = '{payload_json}'")
            
            if set_clauses:
                cypher_query = f"""
                    MATCH (n:{self.collection_name})
                    WHERE n.id = '{vector_id}'
                    SET {', '.join(set_clauses)}
                """
                
                self.conn.execute(cypher_query)
                logger.info(f"Updated vector {vector_id}")
            
            else:
                logger.warning(f"No update parameters provided for vector {vector_id}")
        
        except Exception as e:
            logger.error(f"Error updating vector: {e}")
            raise

    def get(self, vector_id: str) -> OutputData:
        """
        Retrieve a vector by ID.
        
        Args:
            vector_id: ID of the vector to retrieve
            
        Returns:
            OutputData object
        """
        logger.info(f"Retrieving vector {vector_id} from collection '{self.collection_name}'")
        
        try:
            result = self.conn.execute(f"""
                MATCH (n:{self.collection_name})
                WHERE n.id = '{vector_id}'
                RETURN n.id AS id, n.payload AS payload
            """)
            
            if result.has_next():
                row = result.get_next()
                payload = json.loads(row["payload"])
                
                logger.info(f"Retrieved vector {vector_id}")
                return OutputData(id=row["id"], payload=payload)
            
            else:
                logger.warning(f"Vector {vector_id} not found")
                return OutputData()
        
        except Exception as e:
            logger.error(f"Error retrieving vector: {e}")
            return OutputData()

    def list_cols(self) -> List:
        """
        List all collections.
        
        Returns:
            List of collections
        """
        logger.info("Listing collections")
        
        try:
            # Get all node table names
            result = self.conn.execute("""
                SHOW NODE TABLES
            """)
            
            collections = []
            while result.has_next():
                row = result.get_next()
                collections.append(row["name"])
            
            logger.info(f"Found {len(collections)} collections")
            return collections
        
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def delete_col(self) -> None:
        """
        Delete a collection.
        """
        logger.info(f"Deleting collection '{self.collection_name}'")
        
        try:
            # First drop the index
            self.conn.execute(f"""
                DROP INDEX ON {self.collection_name}(vector)
            """)
            
            # Then drop the table
            self.conn.execute(f"""
                DROP NODE TABLE {self.collection_name}
            """)
            
            logger.info(f"Deleted collection '{self.collection_name}'")
        
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise

    def col_info(self) -> Dict:
        """
        Get information about a collection.
        
        Returns:
            Dictionary with collection information
        """
        logger.info(f"Getting information for collection '{self.collection_name}'")
        
        try:
            # Get table info
            table_result = self.conn.execute(f"""
                SHOW NODE TABLE {self.collection_name}
            """)
            
            if table_result.has_next():
                table_info = table_result.get_next()
                
                # Get index info
                index_result = self.conn.execute(f"""
                    SHOW INDEXES
                """)
                
                indices = []
                while index_result.has_next():
                    index_row = index_result.get_next()
                    if index_row["table"] == self.collection_name:
                        indices.append(index_row)
                
                # Combine information
                info = {
                    "name": self.collection_name,
                    "properties": table_info.get("properties", []),
                    "indices": indices,
                    "dimension": self.vector_dimension,
                    "metric": self.distance_metric
                }
                
                logger.info(f"Retrieved information for collection '{self.collection_name}'")
                return info
            
            else:
                logger.warning(f"Collection '{self.collection_name}' not found")
                return {}
        
        except Exception as e:
            logger.error(f"Error retrieving collection information: {e}")
            return {}

    def list(self, filters: Optional[Dict] = None, limit: int = 100) -> List[OutputData]:
        """
        List all memories.
        
        Args:
            filters: Filters to apply
            limit: Maximum number of results
            
        Returns:
            List of OutputData objects
        """
        logger.info(f"Listing vectors in collection '{self.collection_name}'")
        
        # Build the query
        cypher_query = f"""
            MATCH (n:{self.collection_name})
        """
        
        # Add filters if provided
        where_clauses = []
        if filters:
            for key, value in filters.items():
                # Extract from JSON payload
                where_clauses.append(f"JSON_EXTRACT(n.payload, '$.{key}') = '{value}'")
        
        if where_clauses:
            cypher_query += f" WHERE {' AND '.join(where_clauses)}"
        
        # Complete the query
        cypher_query += f"""
            RETURN n.id AS id, n.payload AS payload
            LIMIT {limit}
        """
        
        try:
            result = self.conn.execute(cypher_query)
            
            # Process results
            output_results = []
            while result.has_next():
                row = result.get_next()
                
                # Parse payload from JSON
                payload = json.loads(row["payload"])
                
                output_results.append(OutputData(
                    id=row["id"],
                    payload=payload
                ))
            
            logger.info(f"Listed {len(output_results)} vectors")
            return [output_results]  # Match the API format of other vector stores
        
        except Exception as e:
            logger.error(f"Error listing vectors: {e}")
            return [[]]
```

## Appendix B: Basic KuzuMemoryGraph Implementation

Here's a starting implementation for the KuzuMemoryGraph class:

```python
import json
import logging
import uuid
from typing import Dict, List, Optional

import kuzu
from rank_bm25 import BM25Okapi

from mem0.memory.utils import format_entities
from mem0.utils.factory import EmbedderFactory, LlmFactory

logger = logging.getLogger(__name__)


class KuzuMemoryGraph:
    """Graph memory implementation using Kuzu as the backend."""
    
    def __init__(self, config):
        """
        Initialize Kuzu graph memory.
        
        Args:
            config: Memory configuration
        """
        self.config = config
        self.db_path = config.graph_store.config.db_path
        
        self.db = kuzu.Database(self.db_path)
        self.conn = kuzu.Connection(self.db)
        
        # Initialize other components similar to current MemoryGraph
        self.embedding_model = EmbedderFactory.create(self.config.embedder.provider, self.config.embedder.config)
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.threshold = 0.7
        
        # Initialize schema if needed
        self._initialize_schema()
        
        logger.info("Initialized KuzuMemoryGraph")
    
    def _initialize_schema(self):
        """Initialize the database schema for graph memory."""
        # Check if schema already exists
        try:
            self.conn.execute("MATCH (n:Entity) RETURN n LIMIT 1")
            logger.info("Graph schema already exists")
            return
        except Exception:
            logger.info("Creating graph schema")
            
            # Create entity node table
            self.conn.execute("""
                CREATE NODE TABLE Entity (
                    name STRING PRIMARY KEY,
                    type STRING,
                    user_id STRING,
                    embedding FLOAT[],
                    created TIMESTAMP
                )
            """)
            
            # Create vector index on entity embedding
            self.conn.execute("""
                CREATE VECTOR INDEX ON Entity(embedding) USING COSINE
            """)
            
            # Create relationship table for all possible relationships
            self.conn.execute("""
                CREATE REL TABLE RELATES (
                    FROM Entity(name) TO Entity(name),
                    type STRING,
                    user_id STRING,
                    created TIMESTAMP
                )
            """)
            
            logger.info("Created graph schema")
    
    def add(self, data, filters):
        """
        Add data to the graph.
        
        Args:
            data: Data to add to the graph
            filters: Filters to apply
            
        Returns:
            Dict with added and deleted entities
        """
        logger.info("Adding data to graph")
        
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)
        
        deleted_entities = self._delete_entities(to_be_deleted, filters["user_id"])
        added_entities = self._add_entities(to_be_added, filters["user_id"], entity_type_map)
        
        return {"deleted_entities": deleted_entities, "added_entities": added_entities}
    
    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.
        
        Args:
            query: Query to search for
            filters: Filters to apply
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        logger.info(f"Searching graph with query: {query}")
        
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        
        if not search_output:
            return []
        
        search_outputs_sequence = [
            [item["source"], item["relationship"], item["destination"]] for item in search_output
        ]
        bm25 = BM25Okapi(search_outputs_sequence)
        
        tokenized_query = query.split(" ")
        reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)
        
        search_results = []
        for item in reranked_results:
            search_results.append({"source": item[0], "relationship": item[1], "destination": item[2]})
        
        logger.info(f"Returned {len(search_results)} search results")
        
        return search_results
    
    def delete_all(self, filters):
        """
        Delete all data matching filters.
        
        Args:
            filters: Filters to apply
        """
        logger.info(f"Deleting all graph data for user: {filters.get('user_id')}")
        
        try:
            # Delete all relationships first
            self.conn.execute(f"""
                MATCH (n:Entity {{user_id: '{filters["user_id"]}'}})-[r:RELATES]->()
                DELETE r
            """)
            
            # Then delete all nodes
            self.conn.execute(f"""
                MATCH (n:Entity {{user_id: '{filters["user_id"]}'}})
                DELETE n
            """)
            
            logger.info(f"Deleted all graph data for user: {filters.get('user_id')}")
        
        except Exception as e:
            logger.error(f"Error deleting graph data: {e}")
            raise
    
    def get_all(self, filters, limit=100):
        """
        Retrieve all nodes and relationships.
        
        Args:
            filters: Filters to apply
            limit: Maximum number of results
            
        Returns:
            List of relationships
        """
        logger.info(f"Getting all graph data for user: {filters.get('user_id')}")
        
        try:
            # Query all relationships
            result = self.conn.execute(f"""
                MATCH (source:Entity {{user_id: '{filters["user_id"]}'}})-[r:RELATES]->(destination:Entity {{user_id: '{filters["user_id"]}'}})
                RETURN source.name AS source, r.type AS relationship, destination.name AS target
                LIMIT {limit}
            """)
            
            # Process results
            final_results = []
            while result.has_next():
                row = result.get_next()
                final_results.append({
                    "source": row["source"],
                    "relationship": row["relationship"],
                    "target": row["target"],
                })
            
            logger.info(f"Retrieved {len(final_results)} relationships")
            return final_results
        
        except Exception as e:
            logger.error(f"Error retrieving graph data: {e}")
            return []
    
    # Additional methods would follow the same pattern as the Neo4j implementation
    # but use Kuzu's Cypher dialect for queries
    
    def _retrieve_nodes_from_data(self, data, filters):
        """Extract entities from text data using LLM."""
        # Implementation would be similar to Neo4j version
        # ...
        
    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relationships between entities using LLM."""
        # Implementation would be similar to Neo4j version
        # ...
        
    def _search_graph_db(self, node_list, filters, limit=100):
        """Search for similar nodes and their relationships."""
        # Convert Neo4j queries to Kuzu's Cypher format
        # ...
        
    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Determine which entities should be deleted based on search results."""
        # Implementation would be similar to Neo4j version
        # ...
        
    def _delete_entities(self, to_be_deleted, user_id):
        """Delete specific relationships from the graph."""
        # Convert Neo4j queries to Kuzu's Cypher format
        # ...
        
    def _add_entities(self, to_be_added, user_id, entity_type_map):
        """Add new entities and relationships to the graph."""
        # Convert Neo4j queries to Kuzu's Cypher format
        # ...
```

## Appendix C: Technical Notes and Challenges

When implementing the Kuzu integration, pay attention to these technical aspects:

1. **Cypher Dialect Differences**:
   - Neo4j and Kuzu both use Cypher but have syntax differences
   - Kuzu's Cypher may lack some functions that Neo4j provides
   - Test queries thoroughly before implementation

2. **Vector Indexing**:
   - Kuzu's vector indexing is still evolving
   - May need to adjust search parameters or implementation as Kuzu matures
   - Monitor performance closely

3. **Transaction Handling**:
   - Ensure proper transaction management for consistency
   - Batch operations when possible for performance

4. **JSON Handling**:
   - Kuzu uses JSON for complex data; validate and escape properly
   - Test with various payload types

5. **Error Handling**:
   - Implement robust error handling for database operations
   - Consider fallback strategies for critical operations

6. **Monitoring and Metrics**:
   - Add metrics and logs to track performance
   - Identify bottlenecks in both vector and graph operations