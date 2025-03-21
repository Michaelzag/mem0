import logging
from typing import Dict, Optional

try:
    import kuzu
except ImportError:
    raise ImportError("The 'kuzu' library is required. Please install it using 'pip install kuzu'.")

logger = logging.getLogger(__name__)


class KuzuConnectionManager:
    """
    Singleton manager for Kuzu database connections.
    
    This class ensures that only one connection is created per database path,
    enabling proper connection sharing and transaction coordination between
    different components using the same Kuzu database.
    """
    
    _instances: Dict[str, 'KuzuConnectionManager'] = {}
    
    def __new__(cls, db_path: str):
        """
        Create or return existing connection manager for the given database path.
        
        Args:
            db_path: Path to the Kuzu database directory
            
        Returns:
            KuzuConnectionManager instance for the specified database path
        """
        if db_path not in cls._instances:
            logger.info(f"Creating new KuzuConnectionManager for database at {db_path}")
            cls._instances[db_path] = super(KuzuConnectionManager, cls).__new__(cls)
            cls._instances[db_path]._initialized = False
        return cls._instances[db_path]
    
    def __init__(self, db_path: str):
        """
        Initialize the Kuzu connection manager.
        
        Args:
            db_path: Path to the Kuzu database directory
        """
        if not hasattr(self, "_initialized") or not self._initialized:
            logger.info(f"Initializing Kuzu database connection at {db_path}")
            self.db_path = db_path
            self.db = kuzu.Database(db_path)
            self.conn = kuzu.Connection(self.db)
            self._transaction_active = False
            self._initialized = True
            
    def get_connection(self) -> kuzu.Connection:
        """
        Get the shared Kuzu connection.
        
        Returns:
            kuzu.Connection: Shared connection instance
        """
        return self.conn
    
    def begin_transaction(self) -> None:
        """
        Begin a new transaction if one is not already active.
        """
        if not self._transaction_active:
            logger.debug("Beginning Kuzu transaction")
            self.conn.execute("BEGIN TRANSACTION")
            self._transaction_active = True
        else:
            logger.warning("Transaction already active, skipping begin_transaction")
    
    def commit(self) -> None:
        """
        Commit the current transaction if one is active.
        """
        if self._transaction_active:
            logger.debug("Committing Kuzu transaction")
            self.conn.execute("COMMIT")
            self._transaction_active = False
        else:
            logger.warning("No active transaction to commit")
    
    def rollback(self) -> None:
        """
        Rollback the current transaction if one is active.
        """
        if self._transaction_active:
            logger.debug("Rolling back Kuzu transaction")
            self.conn.execute("ROLLBACK")
            self._transaction_active = False
        else:
            logger.warning("No active transaction to rollback")