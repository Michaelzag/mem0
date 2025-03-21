# Kuzu Vector Store Example

This example demonstrates how to use Kuzu as a vector store in mem0, showcasing its key features and practical usage patterns.

## Basic Setup

First, let's set up a simple Kuzu vector store:

```python
import os
from mem0 import Memory

# Set your API key
os.environ["OPENAI_API_KEY"] = "sk-your-api-key"

# Configure mem0 with Kuzu vector store
config = {
    "vector_store": {
        "provider": "kuzu",
        "config": {
            "collection_name": "product_knowledge",
            "db_path": "./product_db",
            "vector_dimension": 1536,
            "distance_metric": "cosine"
        }
    }
}

# Initialize Memory with Kuzu vector store
memory = Memory.from_config(config)
```

## Adding Data to Kuzu Vector Store

Now let's add some product information:

```python
# Sample product data
product_data = [
    "The Ultra Notebook Pro features 32GB RAM, an Intel i9 processor, and a 4K OLED display.",
    "The Lightweight Tablet S8 weighs only 350g and has a battery life of 12 hours.",
    "The Gaming Desktop GTX has NVIDIA RTX 4080 graphics and liquid cooling for intense gaming sessions.",
    "The Smart Speaker Max has 360-degree sound and advanced voice recognition technology.",
    "The Wireless Earbuds Pro come with noise cancellation and 8 hours of battery life per charge."
]

# Add each product description
for i, data in enumerate(product_data):
    memory.add(
        data,
        user_id="product-catalog",
        metadata={"product_id": f"PROD-{i+100}", "category": "electronics"}
    )

print("Added product information to Kuzu vector store")
```

## Searching for Similar Products

Let's search for products that match a customer query:

```python
# Search for products similar to a query
query = "I need a powerful computer for gaming"

# Get similar products
search_results = memory.search(
    query,
    user_id="product-catalog",
    limit=2
)

print("\nSearch Results:")
for i, result in enumerate(search_results):
    print(f"Result {i+1}:")
    print(f"- Content: {result.content}")
    print(f"- Similarity: {result.similarity:.4f}")
    print(f"- Product ID: {result.metadata.get('product_id')}")
    print()
```

Output:
```
Search Results:
Result 1:
- Content: The Gaming Desktop GTX has NVIDIA RTX 4080 graphics and liquid cooling for intense gaming sessions.
- Similarity: 0.8923
- Product ID: PROD-102

Result 2:
- Content: The Ultra Notebook Pro features 32GB RAM, an Intel i9 processor, and a 4K OLED display.
- Similarity: 0.7645
- Product ID: PROD-100
```

## Filtering Search Results

Kuzu allows filtering search results based on metadata:

```python
# Search with filters
filtered_results = memory.search(
    "lightweight device",
    user_id="product-catalog",
    filters={"product_id": "PROD-101"},  # Only return specific product
    limit=5
)

print("\nFiltered Results:")
for i, result in enumerate(filtered_results):
    print(f"Result {i+1}:")
    print(f"- Content: {result.content}")
    print(f"- Product ID: {result.metadata.get('product_id')}")
```

## Advanced Operations

### Updating Vector Data

You can update existing vector data:

```python
# Update a product description
memory.vector_store.update(
    "PROD-100",  # Assuming this is the vector ID
    payload={"product_id": "PROD-100", "category": "electronics", "price": 1299.99}
)
```

### Deleting Vectors

Remove vectors when they're no longer needed:

```python
# Delete a product from the vector store
memory.vector_store.delete("PROD-104")  # Remove the earbuds
```

## Performance Considerations

Kuzu vector store performance can be optimized with these techniques:

1. **Batch Processing**: When adding multiple vectors, use batch operations:
   ```python
   vectors = [...list of embeddings...]
   payloads = [...list of metadata...]
   ids = [...list of ids...]
   memory.vector_store.insert(vectors, payloads, ids)
   ```

2. **Transaction Management**: For multiple operations, use transaction control:
   ```python
   from mem0.utils.kuzu_connection import KuzuConnectionManager
   
   # Get connection manager instance
   manager = KuzuConnectionManager("./product_db")
   
   # Begin transaction
   manager.begin_transaction()
   
   try:
       # Multiple operations
       memory.vector_store.insert(vectors1, payloads1, ids1)
       memory.vector_store.insert(vectors2, payloads2, ids2)
       
       # Commit if successful
       manager.commit()
   except Exception as e:
       # Rollback on error
       manager.rollback()
       print(f"Error: {e}")
   ```

3. **Vector Dimension**: Choose appropriate vector dimension for your embeddings model

4. **Distance Metric**: Select the most appropriate distance metric for your use case:
   - `cosine`: Best for most text embeddings (default)
   - `euclidean`: Better for some spatial data
   - `dot`: May work better for certain specialized models

## Complete Example Code

Here's the complete example, including all imports and error handling:

```python
import os
import logging
from mem0 import Memory

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set API key
os.environ["OPENAI_API_KEY"] = "sk-your-api-key"

# Configure memory with Kuzu
config = {
    "vector_store": {
        "provider": "kuzu",
        "config": {
            "collection_name": "product_knowledge",
            "db_path": "./product_db",
            "vector_dimension": 1536,
            "distance_metric": "cosine"
        }
    }
}

try:
    # Initialize memory
    memory = Memory.from_config(config)
    
    # Sample product data
    product_data = [
        "The Ultra Notebook Pro features 32GB RAM, an Intel i9 processor, and a 4K OLED display.",
        "The Lightweight Tablet S8 weighs only 350g and has a battery life of 12 hours.",
        "The Gaming Desktop GTX has NVIDIA RTX 4080 graphics and liquid cooling for intense gaming sessions.",
        "The Smart Speaker Max has 360-degree sound and advanced voice recognition technology.",
        "The Wireless Earbuds Pro come with noise cancellation and 8 hours of battery life per charge."
    ]
    
    # Add data
    for i, data in enumerate(product_data):
        memory.add(
            data,
            user_id="product-catalog",
            metadata={"product_id": f"PROD-{i+100}", "category": "electronics"}
        )
    
    # Search example
    query = "I need a powerful computer for gaming"
    search_results = memory.search(
        query,
        user_id="product-catalog",
        limit=2
    )
    
    # Display results
    print("\nSearch Results:")
    for i, result in enumerate(search_results):
        print(f"Result {i+1}:")
        print(f"- Content: {result.content}")
        print(f"- Similarity: {result.similarity:.4f}")
        print(f"- Product ID: {result.metadata.get('product_id')}")
        print()
    
except Exception as e:
    logging.error(f"Error in Kuzu vector store example: {e}")
```

This example demonstrates the basic usage of Kuzu as a vector store in mem0. Kuzu offers an efficient embedded database solution, eliminating the need for external dependencies while providing powerful vector search capabilities.