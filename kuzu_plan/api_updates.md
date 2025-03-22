# Kuzu Integration API Updates

## API Changes in Kuzu v0.8.2

The primary API change in Kuzu v0.8.2 that affects our integration is the connection creation method. Previously, connections were created using the constructor directly, but now they should be created using a method on the Database object.

### Before (pre-v0.8.2):
```python
from kuzu import Database, Connection

db = Database(db_path)
conn = Connection(db)  # Direct constructor usage
```

### After (v0.8.2+):
```python
from kuzu import Database

db = Database(db_path)
conn = db.create_connection()  # Using method on Database object
```

## Implementation Updates

We've updated the KuzuConnectionManager class to use the new API:

```python
# Before
self.db = kuzu.Database(db_path)
self.conn = kuzu.Connection(self.db)

# After
self.db = kuzu.Database(db_path)
self.conn = self.db.create_connection()
```

This change is backward incompatible with older versions of Kuzu, so users must update to Kuzu v0.8.2 or later to use this integration.

## Testing Strategy

To ensure compatibility with the new API, we've added:

1. **Conditional imports**: All Kuzu-related test files now check for Kuzu availability
   ```python
   try:
       import kuzu
       KUZU_AVAILABLE = True
   except ImportError:
       KUZU_AVAILABLE = False
   
   pytestmark = pytest.mark.skipif(not KUZU_AVAILABLE, reason="Kuzu not installed")
   ```

2. **API version tests**: The integration tests explicitly verify compatibility with v0.8.2 API
   ```python
   @pytest.mark.parametrize("api_version", ["0.8.2"])
   def test_connection_manager_api_compatibility(api_version, mock_kuzu_db):
       # Test code verifying that db.create_connection() is called
   ```

3. **Mock usage**: All tests use mocks that match the new API pattern

## Dependency Management

To use Kuzu with mem0, users should:

1. Install the core mem0 package
2. Install Kuzu v0.8.2 or later (`pip install kuzu>=0.8.2`)

When installed together, mem0 will automatically use the appropriate API for Kuzu v0.8.2.

## Migration Guidance

Users moving from older versions should:

1. Upgrade their Kuzu installation: `pip install --upgrade kuzu>=0.8.2`
2. Make sure to update any custom code that directly creates Kuzu connections

No data migration is needed as the database format remains compatible.