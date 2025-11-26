import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class SqliteTool:
    """
    A robust tool for interacting with a SQLite database.

    This class provides methods to get table schemas and execute SQL queries,
    with proper error handling and context management. It is designed to be
    used as a context manager to ensure database connections are handled correctly.

    Usage:
        with SqliteTool() as db_tool:
            schema = db_tool.get_table_schema('orders')
            results = db_tool.execute_sql('SELECT * FROM orders LIMIT 5;')
    """

    def __init__(self, db_path: Optional[Path] = None, db_name: str = "northwind.sqlite"):
        # Construct a robust path to the database
        if db_path:
            self._db_path = db_path
        else:
            self._db_path = Path(__file__).parent.parent.parent / "data" / db_name
        self._conn: Optional[sqlite3.Connection] = None
        self._allowed_tables = {
            "Orders", "orders",
            "Order Details", "order_items",
            "Products", "products",
            "Customers", "customers",
            "Categories", "categories",
            "Suppliers", "suppliers"
        }

    def __enter__(self):
        """Establish the database connection."""
        if not self._db_path.exists():
            raise FileNotFoundError(f"Database file not found at {self._db_path}")
        try:
            self._conn = sqlite3.connect(self._db_path)
            logger.info(f"Successfully connected to {self._db_path}")
            self._create_views()
            return self
        except sqlite3.Error as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            logger.info("Database connection closed.")

    def _create_views(self):
        """Create lowercase compatibility views for easier querying."""
        if not self._conn:
            raise ConnectionError("Database is not connected.")
        
        script = """
            CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;
            CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM "Order Details";
            CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;
            CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;
        """
        try:
            with self._conn:
                self._conn.executescript(script)
            logger.info("Compatibility views created or already exist.")
        except sqlite3.Error as e:
            logger.error(f"Error creating views: {e}")
            raise

    def get_table_schema(self, table_name: str) -> str:
        """
        Returns the 'CREATE TABLE' statement for a given table, cleaned for the LLM.
        """
        if table_name not in self._allowed_tables:
            raise ValueError(f"Access to table '{table_name}' is not allowed.")
        if not self._conn:
            raise ConnectionError("Database is not connected.")

        try:
            with self._conn:
                cursor = self._conn.cursor()
                # Query the sqlite_master table for the original CREATE statement
                cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND tbl_name=?", (table_name,))
                result = cursor.fetchone()
                if result:
                    # Clean the schema string by removing brackets for the LLM
                    return result[0].replace('[', '').replace(']', '')
                else:
                    raise ValueError(f"Table '{table_name}' not found in the database schema.")
        except sqlite3.Error as e:
            logger.error(f"Failed to get schema for table '{table_name}': {e}")
            raise

    def get_all_schemas(self) -> str:
        """Returns the CREATE TABLE statements for all allowed canonical tables."""
        canonical_tables = sorted(["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"])
        return "\n\n".join([self.get_table_schema(table) for table in canonical_tables])

    def execute_sql(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes a SQL query and returns the results as a list of dictionaries.
        """
        if not self._conn:
            raise ConnectionError("Database is not connected.")
        
        logger.info(f"Executing SQL query: {query}")
        try:
            with self._conn:
                cursor = self._conn.cursor()
                cursor.execute(query)
                
                # Get column names from the cursor description
                columns = [description[0] for description in cursor.description]
                
                # Fetch all rows and create a list of dictionaries
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logger.error(f"SQL execution failed for query '{query}': {e}")
            # Re-raise with a more informative error message
            raise sqlite3.Error(f"SQL Error: {e}. Query: '{query}'") from e
