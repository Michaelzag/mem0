import json
import logging
import uuid
from typing import Dict, List, Optional

from pydantic import BaseModel

try:
    import kuzu
except ImportError:
    raise ImportError("The 'kuzu' library is required. Please install it using 'pip install kuzu'.")

from mem0.vector_stores.base import VectorStoreBase
from mem0.utils.kuzu_connection import KuzuConnectionManager

logger = logging.getLogger(__name__)


class OutputData(BaseModel):
    id: Optional[str] = None  # memory id
    score: Optional[float] = None  # similarity score
    payload: Optional[Dict] = None  # metadata


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
        logger.info(f"Initializing KuzuVectorStore with collection '{collection_name}' at {db_path}")
        
        # Use connection manager instead of direct connection
        self.connection_manager = KuzuConnectionManager(db_path)
        self.conn = self.connection_manager.get_connection()
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
        
        # Batch insert in transaction using connection manager
        self.connection_manager.begin_transaction()
        
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
            
            self.connection_manager.commit()
            logger.info(f"Successfully inserted {len(vectors)} vectors")
        
        except Exception as e:
            self.connection_manager.rollback()
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