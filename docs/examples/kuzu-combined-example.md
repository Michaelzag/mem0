# Kuzu Combined Vector and Graph Memory Example

This example demonstrates how to use Kuzu's combined vector and graph capabilities in a single application, showcasing the benefits of using a unified database for both memory types.

## Setup

First, let's configure mem0 to use Kuzu for both vector storage and graph memory:

```python
import os
from mem0 import Memory

# Set your API key
os.environ["OPENAI_API_KEY"] = "sk-your-api-key"

# Configure mem0 with Kuzu for both vector store and graph memory
config = {
    "vector_store": {
        "provider": "kuzu",
        "config": {
            "collection_name": "support_knowledge",
            "db_path": "./support_db",
            "vector_dimension": 1536,
            "distance_metric": "cosine"
        }
    },
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db_path": "./support_db"  # Same path enables shared connection
        }
    }
}

# Initialize Memory with shared Kuzu database
memory = Memory.from_config(config)
```

Notice we use the same `db_path` for both vector store and graph memory. This enables mem0 to use a shared connection manager, improving performance and resource utilization.

## Adding Knowledge Base Data

Let's populate our customer support knowledge base with product information:

```python
# Sample product information
product_data = [
    {
        "content": "The SmartHome Hub connects all your smart devices using WiFi and Bluetooth. " +
                  "It supports voice commands and has a touchscreen interface.",
        "metadata": {"product_id": "SH-100", "category": "smart home"}
    },
    {
        "content": "The SmartHome Sensor detects motion, temperature, and humidity. " +
                  "It runs on batteries that last approximately 12 months.",
        "metadata": {"product_id": "SH-101", "category": "smart home"}
    },
    {
        "content": "The SmartHome Lock enables keyless entry to your home via the mobile app. " +
                  "It features fingerprint recognition and temporary access codes.",
        "metadata": {"product_id": "SH-102", "category": "smart home"}
    }
]

# Add product information to memory
for item in product_data:
    memory.add(
        item["content"],
        user_id="support",
        metadata=item["metadata"]
    )

print("Added product information to vector store")
```

## Building a Knowledge Graph

Now let's add relationship information about common issues and solutions. This will be stored in the graph memory:

```python
# Common issues and solutions with relationships to products
issues_data = [
    "SmartHome Hub fails to connect to some devices after a power outage",
    "SmartHome Sensor stops detecting motion after battery level drops below 20%",
    "SmartHome Lock occasionally shows 'connection lost' error in the mobile app",
    "SmartHome Hub touchscreen becomes unresponsive after extended use",
    "SmartHome Sensor reports incorrect temperature readings in humid environments",
    "SmartHome Lock fingerprint reader fails to recognize registered users"
]

# Add issues to graph memory
for issue in issues_data:
    memory.add(
        issue,
        user_id="support",
        metadata={"type": "issue"}
    )

# Solutions with relationships to issues and products
solutions_data = [
    "Restart the SmartHome Hub by holding the power button for 10 seconds. This will reset the connection protocol without deleting your device configuration.",
    "Replace the batteries in the SmartHome Sensor when they reach 20% to prevent detection issues. The app sends a notification when battery levels are low.",
    "Ensure your SmartHome Lock is within range of your WiFi router. Moving your router closer or adding a WiFi extender can resolve connection issues.",
    "Clean the SmartHome Hub touchscreen with a microfiber cloth. If the issue persists, perform a factory reset by pressing the recessed button on the back.",
    "Calibrate your SmartHome Sensor through the app settings menu by selecting the device and choosing 'Calibrate Sensors' option.",
    "Clean the SmartHome Lock fingerprint reader with a dry microfiber cloth and re-register your fingerprints in the mobile app."
]

# Add solutions to graph memory
for solution in solutions_data:
    memory.add(
        solution,
        user_id="support",
        metadata={"type": "solution"}
    )

print("Added issues and solutions to graph memory")
```

## Querying the Combined Memory System

Now let's see how we can leverage both vector and graph memory to answer customer questions:

```python
def support_assistant(query):
    """Simulate a support assistant using both vector and graph memory"""
    print(f"\nCustomer question: {query}")
    
    # Step 1: Find relevant information using vector search
    vector_results = memory.search(
        query,
        user_id="support",
        limit=2
    )
    
    print("\nRelevant product information (vector search):")
    for i, result in enumerate(vector_results):
        print(f"Result {i+1}: {result.content}")
        print(f"Product ID: {result.metadata.get('product_id')}")
    
    # Step 2: Find related issues and solutions using graph memory
    graph_results = memory.graph.search(
        query,
        filters={"user_id": "support"},
        limit=5
    )
    
    # Extract entity names from graph results
    entities = set()
    for result in graph_results:
        entities.add(result["source"])
        entities.add(result["destination"])
    
    # Find solutions that relate to our question
    solutions = [entity for entity in entities if "solution" in entity.lower() or 
                                               "restart" in entity.lower() or
                                               "replace" in entity.lower() or
                                               "clean" in entity.lower()]
    
    print("\nRecommended solutions (graph search):")
    for i, solution in enumerate(solutions[:2]):
        print(f"Solution {i+1}: {solution}")
    
    return vector_results, graph_results

# Test with customer questions
queries = [
    "My SmartHome Hub won't connect to my new smart light bulbs",
    "The touchscreen on my hub is not working properly",
    "My sensor is showing the wrong temperature",
]

for query in queries:
    support_assistant(query)
    print("\n" + "-"*50)
```

## Visualizing the Knowledge Graph

Let's visualize the graph structure we've created:

```python
# Get all relationships from the graph
relationships = memory.graph.get_all(
    filters={"user_id": "support"},
    limit=100
)

print("\nKnowledge Graph Structure:")
print(f"Found {len(relationships)} relationships")

# Display a sample of relationships
for i, rel in enumerate(relationships[:5]):
    print(f"{i+1}. {rel['source']} --[{rel['relationship']}]--> {rel['destination']}")
```

## Benefits of the Combined Approach

This example demonstrates several key benefits of using Kuzu for both vector and graph memory:

1. **Simplified Infrastructure**: A single embedded database handles both vector similarity search and graph operations
2. **Shared Connection**: The `KuzuConnectionManager` automatically shares a database connection between vector and graph components
3. **Complementary Capabilities**: Vector search finds semantically similar content, while graph traversal reveals relationships
4. **Consistent Performance**: All operations work against the same database without network overhead
5. **Reduced Resource Usage**: Single database process requires less memory and CPU than multiple systems

## Performance Considerations

When using the combined approach:

1. **Transaction Coordination**: Both vector and graph operations can participate in the same transactions
2. **Memory Efficiency**: A single database process uses less memory than separate vector and graph databases
3. **Query Optimization**: Structure complex operations to minimize database round-trips

## Complete Example Code

Here's the complete example with error handling:

```python
import os
import logging
from mem0 import Memory

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set API key
os.environ["OPENAI_API_KEY"] = "sk-your-api-key"

try:
    # Configure memory with shared Kuzu database
    config = {
        "vector_store": {
            "provider": "kuzu",
            "config": {
                "collection_name": "support_knowledge",
                "db_path": "./support_db",
                "vector_dimension": 1536,
                "distance_metric": "cosine"
            }
        },
        "graph_store": {
            "provider": "kuzu",
            "config": {
                "db_path": "./support_db"  # Same path for shared connection
            }
        }
    }
    
    # Initialize memory
    memory = Memory.from_config(config)
    
    # Add product information (for vector store)
    product_data = [
        {
            "content": "The SmartHome Hub connects all your smart devices using WiFi and Bluetooth. " +
                      "It supports voice commands and has a touchscreen interface.",
            "metadata": {"product_id": "SH-100", "category": "smart home"}
        },
        {
            "content": "The SmartHome Sensor detects motion, temperature, and humidity. " +
                      "It runs on batteries that last approximately 12 months.",
            "metadata": {"product_id": "SH-101", "category": "smart home"}
        },
        {
            "content": "The SmartHome Lock enables keyless entry to your home via the mobile app. " +
                      "It features fingerprint recognition and temporary access codes.",
            "metadata": {"product_id": "SH-102", "category": "smart home"}
        }
    ]
    
    for item in product_data:
        memory.add(
            item["content"],
            user_id="support",
            metadata=item["metadata"]
        )
    
    # Add issues and solutions (for graph memory)
    issues_data = [
        "SmartHome Hub fails to connect to some devices after a power outage",
        "SmartHome Sensor battery low warning",
        "SmartHome Lock connection lost error"
    ]
    
    solutions_data = [
        "Restart the SmartHome Hub by holding the power button for 10 seconds",
        "Replace the batteries in the SmartHome Sensor",
        "Ensure your SmartHome Lock is within range of your WiFi router"
    ]
    
    for issue in issues_data:
        memory.add(
            issue,
            user_id="support",
            metadata={"type": "issue"}
        )
    
    for solution in solutions_data:
        memory.add(
            solution,
            user_id="support",
            metadata={"type": "solution"}
        )
    
    # Define a query function using both memory types
    def support_assistant(query):
        # Vector search for product information
        vector_results = memory.search(
            query,
            user_id="support",
            limit=2
        )
        
        # Graph search for related issues and solutions
        graph_results = memory.graph.search(
            query,
            filters={"user_id": "support"},
            limit=5
        )
        
        return vector_results, graph_results
    
    # Test with a customer question
    query = "My SmartHome Hub won't connect to my devices"
    vector_results, graph_results = support_assistant(query)
    
    # Display results
    print("\nVector search results:")
    for i, result in enumerate(vector_results):
        print(f"Result {i+1}: {result.content[:100]}...")
    
    print("\nGraph search results:")
    for i, rel in enumerate(graph_results[:3]):
        print(f"Relationship {i+1}: {rel['source']} --[{rel['relationship']}]--> {rel['destination']}")
    
except Exception as e:
    logging.error(f"Error in Kuzu combined example: {e}")
```

This example demonstrates how to leverage Kuzu's combined vector and graph capabilities within mem0, providing a powerful and efficient memory system for AI applications.