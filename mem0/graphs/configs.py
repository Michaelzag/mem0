from typing import Optional, Union, Dict, Any

from pydantic import BaseModel, Field, field_validator, model_validator

from mem0.llms.configs import LlmConfig

# Try to import Kuzu at module level but don't fail if not available
try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False


class Neo4jConfig(BaseModel):
    url: Optional[str] = Field(None, description="Host address for the graph database")
    username: Optional[str] = Field(None, description="Username for the graph database")
    password: Optional[str] = Field(None, description="Password for the graph database")

    @model_validator(mode="before")
    def check_host_port_or_path(cls, values):
        url, username, password = (
            values.get("url"),
            values.get("username"),
            values.get("password"),
        )
        if not url or not username or not password:
            raise ValueError("Please provide 'url', 'username' and 'password'.")
        return values


class KuzuGraphConfig(BaseModel):
    """Configuration for Kuzu graph database."""
    
    # Store reference to kuzu module to avoid field detection
    kuzu_module: Any = None
    
    db_path: str = Field(
        "./kuzu_db", description="Path to the Kuzu database directory"
    )


class GraphStoreConfig(BaseModel):
    provider: str = Field(description="Provider of the data store (e.g., 'neo4j', 'kuzu')", default="neo4j")
    config: Union[Neo4jConfig, KuzuGraphConfig, Dict[str, Any]] = Field(
        description="Configuration for the specific data store",
        default=None
    )
    llm: Optional[LlmConfig] = Field(description="LLM configuration for querying the graph store", default=None)
    custom_prompt: Optional[str] = Field(
        description="Custom prompt to fetch entities from the given text", default=None
    )

    @field_validator("config")
    def validate_config(cls, v, values):
        provider = values.data.get("provider")
        if provider == "neo4j":
            return Neo4jConfig(**v.model_dump())
        elif provider == "kuzu":
            return KuzuGraphConfig(**v.model_dump())
        else:
            raise ValueError(f"Unsupported graph store provider: {provider}")
