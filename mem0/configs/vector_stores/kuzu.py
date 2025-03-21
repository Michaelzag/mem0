from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class KuzuConfig(BaseModel):
    """Configuration for Kuzu vector store."""
    
    try:
        import kuzu
    except ImportError:
        raise ImportError("The 'kuzu' library is required. Please install it using 'pip install kuzu'.")

    collection_name: str = Field(
        "memories", description="Name for the Kuzu node table used as collection"
    )
    db_path: str = Field(
        "./kuzu_db", description="Path to the Kuzu database directory"
    )
    vector_dimension: int = Field(
        1536, description="Dimension of vectors to store"
    )
    distance_metric: str = Field(
        "cosine", description="Distance metric to use for similarity search (cosine, euclidean, dot)"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    @model_validator(mode="before")
    def validate_distance_metric(cls, values):
        distance_metric = values.get("distance_metric")
        if distance_metric and distance_metric not in ["cosine", "euclidean", "dot"]:
            raise ValueError(
                "Distance metric must be one of 'cosine', 'euclidean', or 'dot'"
            )
        return values