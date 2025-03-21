import json
import logging
from typing import Dict, List, Optional

try:
    import kuzu
except ImportError:
    raise ImportError("The 'kuzu' library is required. Please install it using 'pip install kuzu'.")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 is not installed. Please install it using pip install rank-bm25")

from mem0.memory.utils import format_entities
from mem0.graphs.tools import (DELETE_MEMORY_STRUCT_TOOL_GRAPH,
                               DELETE_MEMORY_TOOL_GRAPH,
                               EXTRACT_ENTITIES_STRUCT_TOOL,
                               EXTRACT_ENTITIES_TOOL, RELATIONS_STRUCT_TOOL,
                               RELATIONS_TOOL)
from mem0.graphs.utils import EXTRACT_RELATIONS_PROMPT, get_delete_messages
from mem0.utils.factory import EmbedderFactory, LlmFactory
from mem0.utils.kuzu_connection import KuzuConnectionManager

logger = logging.getLogger(__name__)


class KuzuMemoryGraph:
    """Graph memory implementation using Kuzu as the backend."""
    
    def __init__(self, config):
        """
        Initialize Kuzu memory graph.
        
        Args:
            config: Configuration object with necessary parameters
        """
        self.config = config
        
        # Initialize Kuzu database connection using connection manager
        self.db_path = getattr(self.config.graph_store.config, "db_path", "./kuzu_db")
        logger.info(f"Initializing KuzuMemoryGraph at {self.db_path}")
        
        # Use connection manager instead of direct connection
        self.connection_manager = KuzuConnectionManager(self.db_path)
        self.conn = self.connection_manager.get_connection()
        
        # Initialize embedder and LLM models (same as Neo4j implementation)
        self.embedding_model = EmbedderFactory.create(self.config.embedder.provider, self.config.embedder.config)
        
        self.llm_provider = "openai_structured"
        if self.config.llm.provider:
            self.llm_provider = self.config.llm.provider
        if self.config.graph_store.llm:
            self.llm_provider = self.config.graph_store.llm.provider
            
        self.llm = LlmFactory.create(self.llm_provider, self.config.llm.config)
        self.user_id = None
        self.threshold = 0.7
        
        # Initialize the graph schema if needed
        self._initialize_graph_schema()
        
        logger.info("KuzuMemoryGraph initialized successfully")
        
    def _initialize_graph_schema(self):
        """Initialize the graph schema if it doesn't exist."""
        # We need to ensure node tables exist for different entity types
        # For now, we'll create a generic node table if it doesn't exist
        try:
            # Check if the node table exists
            self.conn.execute("MATCH (n:Entity) RETURN n LIMIT 1")
            logger.info("Entity node table already exists")
        except Exception:
            # Create the node table with required fields
            logger.info("Creating Entity node table")
            self.conn.execute("""
                CREATE NODE TABLE Entity (
                    name STRING PRIMARY KEY,
                    user_id STRING,
                    entity_type STRING,
                    embedding FLOAT[],
                    created TIMESTAMP
                )
            """)
            
            # Create vector index for similarity search
            self.conn.execute("CREATE VECTOR INDEX ON Entity(embedding) USING COSINE")
            
            logger.info("Created Entity node table with vector index")
            
        # Check if relationship table exists
        try:
            self.conn.execute("MATCH ()-[r:RELATED_TO]->() RETURN r LIMIT 1")
            logger.info("Relationship table already exists")
        except Exception:
            # Create relationship table
            logger.info("Creating relationship table")
            self.conn.execute("""
                CREATE REL TABLE RELATED_TO (
                    FROM Entity TO Entity,
                    relationship STRING,
                    created TIMESTAMP
                )
            """)
            logger.info("Created relationship table")

    def add(self, data, filters):
        """
        Adds data to the graph.

        Args:
            data (str): The data to add to the graph.
            filters (dict): A dictionary containing filters to be applied during the addition.
            
        Returns:
            dict: Information about added and deleted entities
        """
        logger.info(f"Adding data to graph for user {filters.get('user_id')}")
        
        entity_type_map = self._retrieve_nodes_from_data(data, filters)
        to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
        to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)

        deleted_entities = self._delete_entities(to_be_deleted, filters["user_id"])
        added_entities = self._add_entities(to_be_added, filters["user_id"], entity_type_map)

        logger.info(f"Added {len(added_entities)} entities and deleted {len(deleted_entities)} entities")
        return {"deleted_entities": deleted_entities, "added_entities": added_entities}

    def search(self, query, filters, limit=100):
        """
        Search for memories and related graph data.

        Args:
            query (str): Query to search for.
            filters (dict): A dictionary containing filters to be applied during the search.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.

        Returns:
            list: A list of search results with source, relationship, and destination.
        """
        logger.info(f"Searching graph for query: {query} with filters {filters}")
        
        entity_type_map = self._retrieve_nodes_from_data(query, filters)
        search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)

        if not search_output:
            logger.info("No search results found")
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
        Delete all nodes and relationships for a user.
        
        Args:
            filters (dict): A dictionary containing filters to be applied during deletion.
        """
        logger.info(f"Deleting all graph data for user {filters.get('user_id')}")
        
        try:
            # Delete all relationships first
            self.conn.execute("""
                MATCH (n:Entity {user_id: $user_id})-[r:RELATED_TO]->(m:Entity)
                DELETE r
            """, {"user_id": filters["user_id"]})
            
            # Then delete all nodes
            self.conn.execute("""
                MATCH (n:Entity {user_id: $user_id})
                DELETE n
            """, {"user_id": filters["user_id"]})
            
            logger.info(f"Successfully deleted all graph data for user {filters.get('user_id')}")
        except Exception as e:
            logger.error(f"Error deleting graph data: {e}")
            raise

    def get_all(self, filters, limit=100):
        """
        Retrieves all nodes and relationships from the graph database based on optional filtering criteria.

        Args:
            filters (dict): A dictionary containing filters to be applied during the retrieval.
            limit (int): The maximum number of nodes and relationships to retrieve. Defaults to 100.
            
        Returns:
            list: A list of dictionaries, each containing source, relationship, and target information.
        """
        logger.info(f"Retrieving all graph data for user {filters.get('user_id')} with limit {limit}")
        
        try:
            # Query to get all relationships for the user
            result = self.conn.execute("""
                MATCH (n:Entity {user_id: $user_id})-[r:RELATED_TO]->(m:Entity {user_id: $user_id})
                RETURN n.name AS source, r.relationship AS relationship, m.name AS target
                LIMIT $limit
            """, {"user_id": filters["user_id"], "limit": limit})
            
            final_results = []
            while result.has_next():
                row = result.get_next()
                final_results.append({
                    "source": row["source"],
                    "relationship": row["relationship"],
                    "destination": row["target"]  # Using 'destination' to match the Neo4j implementation return format
                })
                
            logger.info(f"Retrieved {len(final_results)} relationships")
            return final_results
            
        except Exception as e:
            logger.error(f"Error retrieving graph data: {e}")
            return []

    def _retrieve_nodes_from_data(self, data, filters):
        """Extracts all the entities mentioned in the query."""
        _tools = [EXTRACT_ENTITIES_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [EXTRACT_ENTITIES_STRUCT_TOOL]
        search_results = self.llm.generate_response(
            messages=[
                {
                    "role": "system",
                    "content": f"You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use {filters['user_id']} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.",
                },
                {"role": "user", "content": data},
            ],
            tools=_tools,
        )

        entity_type_map = {}

        try:
            for item in search_results["tool_calls"][0]["arguments"]["entities"]:
                entity_type_map[item["entity"]] = item["entity_type"]
        except Exception as e:
            logger.error(f"Error in search tool: {e}")

        entity_type_map = {k.lower().replace(" ", "_"): v.lower().replace(" ", "_") for k, v in entity_type_map.items()}
        logger.debug(f"Entity type map: {entity_type_map}")
        return entity_type_map

    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        """Establish relations among the extracted nodes."""
        if self.config.graph_store.custom_prompt:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["user_id"]).replace(
                        "CUSTOM_PROMPT", f"4. {self.config.graph_store.custom_prompt}"
                    ),
                },
                {"role": "user", "content": data},
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["user_id"]),
                },
                {"role": "user", "content": f"List of entities: {list(entity_type_map.keys())}. \n\nText: {data}"},
            ]

        _tools = [RELATIONS_TOOL]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [RELATIONS_STRUCT_TOOL]

        extracted_entities = self.llm.generate_response(
            messages=messages,
            tools=_tools,
        )

        if extracted_entities["tool_calls"]:
            extracted_entities = extracted_entities["tool_calls"][0]["arguments"]["entities"]
        else:
            extracted_entities = []

        extracted_entities = self._remove_spaces_from_entities(extracted_entities)
        logger.debug(f"Extracted entities: {extracted_entities}")
        return extracted_entities

    def _search_graph_db(self, node_list, filters, limit=100):
        """Search similar nodes and their respective incoming and outgoing relations."""
        result_relations = []

        for node in node_list:
            n_embedding = self.embedding_model.embed(node)
            
            # Kuzu version of the vector similarity search
            cypher_query = """
            MATCH (n:Entity)
            WHERE n.user_id = $user_id
            WITH n, COSINE_SIMILARITY(n.embedding, $embedding) AS similarity
            WHERE similarity >= $threshold
            MATCH (n)-[r:RELATED_TO]->(m:Entity)
            RETURN n.name AS source, ID(n) AS source_id, r.relationship AS relationship, 
                   ID(r) AS relation_id, m.name AS destination, ID(m) AS destination_id, similarity
            UNION
            MATCH (n:Entity)
            WHERE n.user_id = $user_id
            WITH n, COSINE_SIMILARITY(n.embedding, $embedding) AS similarity
            WHERE similarity >= $threshold
            MATCH (m:Entity)-[r:RELATED_TO]->(n)
            RETURN m.name AS source, ID(m) AS source_id, r.relationship AS relationship, 
                   ID(r) AS relation_id, n.name AS destination, ID(n) AS destination_id, similarity
            ORDER BY similarity DESC
            LIMIT $limit
            """
            
            # Execute the query
            try:
                result = self.conn.execute(cypher_query, {
                    "embedding": n_embedding,
                    "threshold": self.threshold,
                    "user_id": filters["user_id"],
                    "limit": limit
                })
                
                # Process results
                while result.has_next():
                    row = result.get_next()
                    result_relations.append(row)
                    
            except Exception as e:
                logger.error(f"Error searching graph database: {e}")
        
        return result_relations

    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        """Get the entities to be deleted from the search output."""
        search_output_string = format_entities(search_output)
        system_prompt, user_prompt = get_delete_messages(search_output_string, data, filters["user_id"])

        _tools = [DELETE_MEMORY_TOOL_GRAPH]
        if self.llm_provider in ["azure_openai_structured", "openai_structured"]:
            _tools = [
                DELETE_MEMORY_STRUCT_TOOL_GRAPH,
            ]

        memory_updates = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=_tools,
        )
        to_be_deleted = []
        for item in memory_updates["tool_calls"]:
            if item["name"] == "delete_graph_memory":
                to_be_deleted.append(item["arguments"])
        # in case if it is not in the correct format
        to_be_deleted = self._remove_spaces_from_entities(to_be_deleted)
        logger.debug(f"Deleted relationships: {to_be_deleted}")
        return to_be_deleted

    def _delete_entities(self, to_be_deleted, user_id):
        """Delete the entities from the graph."""
        results = []
        for item in to_be_deleted:
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # Delete the specific relationship between nodes
            try:
                result = self.conn.execute("""
                MATCH (n:Entity {name: $source_name, user_id: $user_id})
                -[r:RELATED_TO]->(m:Entity {name: $dest_name, user_id: $user_id})
                WHERE r.relationship = $relationship
                DELETE r
                RETURN n.name AS source, m.name AS target, r.relationship AS relationship
                """, {
                    "source_name": source,
                    "dest_name": destination,
                    "relationship": relationship,
                    "user_id": user_id
                })
                
                # Process results
                while result.has_next():
                    row = result.get_next()
                    results.append(row)
                    
            except Exception as e:
                logger.error(f"Error deleting relationship: {e}")
        
        return results

    def _add_entities(self, to_be_added, user_id, entity_type_map):
        """Add the new entities to the graph. Merge the nodes if they already exist."""
        results = []
        
        for item in to_be_added:
            # entities
            source = item["source"]
            destination = item["destination"]
            relationship = item["relationship"]

            # types
            source_type = entity_type_map.get(source, "unknown")
            destination_type = entity_type_map.get(destination, "unknown")

            # embeddings
            source_embedding = self.embedding_model.embed(source)
            dest_embedding = self.embedding_model.embed(destination)

            # search for the nodes with the closest embeddings
            source_node_search_result = self._search_source_node(source_embedding, user_id, threshold=0.9)
            destination_node_search_result = self._search_destination_node(dest_embedding, user_id, threshold=0.9)

            try:
                # Begin transaction for atomic operations using connection manager
                self.connection_manager.begin_transaction()
                
                # Determine appropriate approach based on search results
                if not destination_node_search_result and source_node_search_result:
                    # Source exists, need to create destination
                    self.conn.execute("""
                    MATCH (source:Entity)
                    WHERE ID(source) = $source_id
                    CREATE (destination:Entity {
                        name: $destination_name,
                        user_id: $user_id,
                        entity_type: $destination_type,
                        embedding: $destination_embedding,
                        created: TIMESTAMP()
                    })
                    CREATE (source)-[:RELATED_TO {
                        relationship: $relationship,
                        created: TIMESTAMP()
                    }]->(destination)
                    RETURN source.name AS source, destination.name AS target, $relationship AS relationship
                    """, {
                        "source_id": source_node_search_result[0]["ID(source_candidate)"],
                        "destination_name": destination,
                        "destination_type": destination_type,
                        "destination_embedding": dest_embedding,
                        "relationship": relationship,
                        "user_id": user_id
                    })
                    
                elif destination_node_search_result and not source_node_search_result:
                    # Destination exists, need to create source
                    self.conn.execute("""
                    MATCH (destination:Entity)
                    WHERE ID(destination) = $destination_id
                    CREATE (source:Entity {
                        name: $source_name,
                        user_id: $user_id,
                        entity_type: $source_type,
                        embedding: $source_embedding,
                        created: TIMESTAMP()
                    })
                    CREATE (source)-[:RELATED_TO {
                        relationship: $relationship,
                        created: TIMESTAMP()
                    }]->(destination)
                    RETURN source.name AS source, destination.name AS target, $relationship AS relationship
                    """, {
                        "destination_id": destination_node_search_result[0]["ID(destination_candidate)"],
                        "source_name": source,
                        "source_type": source_type,
                        "source_embedding": source_embedding,
                        "relationship": relationship,
                        "user_id": user_id
                    })
                    
                elif source_node_search_result and destination_node_search_result:
                    # Both nodes exist, just create relationship
                    self.conn.execute("""
                    MATCH (source:Entity)
                    WHERE ID(source) = $source_id
                    MATCH (destination:Entity)
                    WHERE ID(destination) = $destination_id
                    CREATE (source)-[:RELATED_TO {
                        relationship: $relationship,
                        created: TIMESTAMP()
                    }]->(destination)
                    RETURN source.name AS source, destination.name AS target, $relationship AS relationship
                    """, {
                        "source_id": source_node_search_result[0]["ID(source_candidate)"],
                        "destination_id": destination_node_search_result[0]["ID(destination_candidate)"],
                        "relationship": relationship,
                        "user_id": user_id
                    })
                    
                else:
                    # Neither node exists, create both
                    self.conn.execute("""
                    CREATE (source:Entity {
                        name: $source_name,
                        user_id: $user_id,
                        entity_type: $source_type,
                        embedding: $source_embedding,
                        created: TIMESTAMP()
                    })
                    CREATE (destination:Entity {
                        name: $destination_name,
                        user_id: $user_id,
                        entity_type: $destination_type,
                        embedding: $destination_embedding,
                        created: TIMESTAMP()
                    })
                    CREATE (source)-[:RELATED_TO {
                        relationship: $relationship,
                        created: TIMESTAMP()
                    }]->(destination)
                    RETURN source.name AS source, destination.name AS target, $relationship AS relationship
                    """, {
                        "source_name": source,
                        "source_type": source_type,
                        "destination_name": destination,
                        "destination_type": destination_type,
                        "source_embedding": source_embedding,
                        "destination_embedding": dest_embedding,
                        "relationship": relationship,
                        "user_id": user_id
                    })
                
                # Commit the transaction using connection manager
                self.connection_manager.commit()
                
                # Add a result entry
                results.append({
                    "source": source,
                    "relationship": relationship,
                    "target": destination
                })
                
            except Exception as e:
                # Rollback in case of error using connection manager
                self.connection_manager.rollback()
                logger.error(f"Error adding entities: {e}")
        
        return results

    def _remove_spaces_from_entities(self, entity_list):
        for item in entity_list:
            item["source"] = item["source"].lower().replace(" ", "_")
            item["relationship"] = item["relationship"].lower().replace(" ", "_")
            item["destination"] = item["destination"].lower().replace(" ", "_")
        return entity_list

    def _search_source_node(self, source_embedding, user_id, threshold=0.9):
        """Search for a source node with similar embedding."""
        try:
            result = self.conn.execute("""
            MATCH (source_candidate:Entity)
            WHERE source_candidate.user_id = $user_id
            WITH source_candidate, COSINE_SIMILARITY(source_candidate.embedding, $embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN ID(source_candidate), similarity
            ORDER BY similarity DESC
            LIMIT 1
            """, {
                "embedding": source_embedding,
                "user_id": user_id,
                "threshold": threshold
            })
            
            results = []
            while result.has_next():
                row = result.get_next()
                results.append(row)
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching for source node: {e}")
            return []

    def _search_destination_node(self, destination_embedding, user_id, threshold=0.9):
        """Search for a destination node with similar embedding."""
        try:
            result = self.conn.execute("""
            MATCH (destination_candidate:Entity)
            WHERE destination_candidate.user_id = $user_id
            WITH destination_candidate, COSINE_SIMILARITY(destination_candidate.embedding, $embedding) AS similarity
            WHERE similarity >= $threshold
            RETURN ID(destination_candidate), similarity
            ORDER BY similarity DESC
            LIMIT 1
            """, {
                "embedding": destination_embedding,
                "user_id": user_id,
                "threshold": threshold
            })
            
            results = []
            while result.has_next():
                row = result.get_next()
                results.append(row)
                
            return results
            
        except Exception as e:
            logger.error(f"Error searching for destination node: {e}")
            return []