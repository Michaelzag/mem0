# Migrating from Neo4j to Kuzu

This guide provides a comprehensive roadmap for migrating your mem0 application from Neo4j to Kuzu for graph memory storage.

## Why Migrate to Kuzu?

Kuzu offers several advantages over Neo4j for many mem0 applications:

1. **Simplified Deployment**: Kuzu is an embedded database that runs within your application process, eliminating the need to manage a separate Neo4j server
2. **Combined Vector and Graph Capabilities**: Kuzu provides native vector operations, allowing you to use a single database for both vector storage and graph memory
3. **Reduced Resource Requirements**: Kuzu has a smaller memory footprint and lower system requirements
4. **Streamlined Development Workflow**: No need to install, configure, and maintain a separate database server
5. **Easy Integration**: Seamless integration with other mem0 components

## Configuration Changes

### Basic Configuration

Migrating from Neo4j to Kuzu requires changing your Memory configuration:

**Neo4j Configuration:**
```python
config = {
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password"
        }
    }
}
```

**Kuzu Configuration:**
```python
config = {
    "graph_store": {
        "provider": "kuzu",
        "config": {
            "db_path": "./kuzu_db"
        }
    }
}
```

### Combined Vector Store and Graph Memory

One of Kuzu's key advantages is the ability to share a single database for both vector storage and graph operations:

```python
config = {
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
            "db_path": "./kuzu_db"  # Same database path for shared connection
        }
    }
}
```

When using the same `db_path` for both vector and graph stores, mem0 automatically manages a shared database connection through the `KuzuConnectionManager`, ensuring efficient resource usage.

## Data Migration Strategies

### Option 1: Gradual Migration (Recommended)

A phased approach allows for safe migration with minimal downtime:

1. **Setup Kuzu alongside Neo4j**:
   ```python
   from mem0 import Memory
   
   # Create a Memory instance with Kuzu
   kuzu_config = {
       "graph_store": {
           "provider": "kuzu",
           "config": {
               "db_path": "./kuzu_db"
           }
       }
   }
   kuzu_memory = Memory.from_config(kuzu_config)
   
   # Existing Neo4j Memory instance
   neo4j_memory = existing_memory_instance
   
   # Fetch all data from Neo4j
   all_neo4j_data = neo4j_memory.graph.get_all({"user_id": user_id})
   
   # Process and add to Kuzu
   for relationship in all_neo4j_data:
       # Convert to appropriate format if needed
       kuzu_memory.add(
           data=f"{relationship['source']} {relationship['relationship']} {relationship['destination']}",
           filters={"user_id": user_id}
       )
   ```

2. **Validate data integrity** by comparing results from both systems
3. **Switch to Kuzu** once validation is complete

### Option 2: Export/Import Approach

For smaller datasets where downtime is acceptable:

1. **Export Neo4j data** to a structured format (JSON/CSV)
2. **Process the exported data** to match Kuzu's expected input format
3. **Import into Kuzu** using mem0's add operations

## API Differences and Compatibility

The mem0 Memory API abstracts most differences between Neo4j and Kuzu, but there are some implementation details to be aware of:

### Key Compatibility Notes

| Feature | Neo4j | Kuzu | Migration Notes |
|---------|-------|------|----------------|
| Schema | Label-based node types | Generic Entity nodes with type property | Kuzu uses a simpler schema with all entities in a single node table |
| Querying | Full Cypher support | Subset of Cypher with some syntax differences | Complex custom queries may need adaptation |
| Performance | Optimized for large graphs | Efficient for embedded use cases | Kuzu performs best for moderate-sized graphs |
| Transactions | Client-side transaction control | Connection manager handles transactions | No code changes needed when using mem0 API |

### Error Handling

Kuzu may produce different error messages than Neo4j. Review your error handling to ensure it accounts for Kuzu-specific errors.

## Performance Considerations

Based on benchmark testing, Kuzu shows:

- **Faster startup time**: ~200ms vs ~2s for Neo4j
- **Comparable query performance** for common operations
- **Lower memory usage**: ~30-50% less than Neo4j for similar workloads
- **Efficient vector operations** when using combined vector/graph store

For very large graphs (millions of nodes/relationships), Neo4j may still offer performance advantages due to its distributed architecture.

## Troubleshooting Common Issues

1. **Schema differences**: Kuzu uses a simpler schema model than Neo4j. If you encounter issues, check that your entities and relationships conform to Kuzu's expected structure.

2. **Cypher compatibility**: If using custom Cypher queries, review for Kuzu compatibility. Some advanced Neo4j Cypher features may need adaptation.

3. **Connection management**: Kuzu handles connections differently than Neo4j. Let the `KuzuConnectionManager` handle connection lifecycle to avoid resource leaks.

4. **Transaction handling**: Kuzu transactions work differently than Neo4j. Use the connection manager's transaction methods for best results.

## Conclusion

Migrating from Neo4j to Kuzu offers significant simplification for many mem0 applications. By following this guide, you can transition smoothly while maintaining application functionality.

For more detailed information about using Kuzu with mem0, refer to:
- [Kuzu Vector Store Documentation](../components/vectordbs/dbs/kuzu.mdx)
- [Memory Types Documentation](../core-concepts/memory-types.mdx)
- [Kuzu Vector Store Example](./kuzu-vector-store-example.md)
- [Combined Kuzu Example](./kuzu-combined-example.md)