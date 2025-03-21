import concurrent
import hashlib
import json
import logging
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict

import pytz
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.prompts import get_update_memory_messages
from mem0.memory.base import MemoryBase
from mem0.memory.setup import setup_config
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import (
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    remove_code_blocks,
)
from mem0.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory

# Setup user config
setup_config()

logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        self.embedding_model = EmbedderFactory.create(self.config.embedder.provider, self.config.embedder.config)
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version

        self.enable_graph = False

        if self.config.graph_store.config:
            from mem0.utils.factory import GraphMemoryFactory
            
            # Use the provider from config or default to neo4j
            provider = getattr(self.config.graph_store, "provider", "neo4j")
            self.graph = GraphMemoryFactory.create(provider, self.config)
            self.enable_graph = True

        capture_event("mem0.init", self)

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def add(
        self,
        messages,
        user_id=None,
        agent_id=None,
        run_id=None,
        metadata=None,
        filters=None,
        infer=True,
        prompt=None,
    ):
        """
        Create a new memory.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            filters (dict, optional): Filters to apply to the search. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            prompt (str, optional): Prompt to use for memory deduction. Defaults to None.

        Returns:
            dict: A dictionary containing the result of the memory addition operation.
            result: dict of affected events with each dict has the following key:
              'memories': affected memories
              'graph': affected graph memories

              'memories' and 'graph' is a dict, each with following subkeys:
                'add': added memory
                'update': updated memory
                'delete': deleted memory


        """
        if metadata is None:
            metadata = {}

        filters = filters or {}
        if user_id:
            filters["user_id"] = metadata["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = metadata["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = metadata["run_id"] = run_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, metadata, filters, infer)
            future2 = executor.submit(self._add_to_graph, messages, filters)

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        if self.api_version == "v1.0":
            warnings.warn(
                "The current add API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return vector_store_result

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}
