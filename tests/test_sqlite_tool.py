import pytest
import sqlite3
from pathlib import Path
from agent.tools.sqlite_tool import SqliteTool

# Define a temporary test database path
@pytest.fixture
def test_db_path(tmp_path):
    """Fixture to create a temporary database file for testing."""
    db_path = tmp_path / "test_northwind.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create dummy tables matching Northwind structure (simplified)
    cursor.execute("""
        CREATE TABLE Orders (
            OrderID INTEGER PRIMARY KEY,
            CustomerID TEXT,
            OrderDate TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE "Order Details" (
            OrderID INTEGER,
            ProductID INTEGER,
            UnitPrice REAL,
            Quantity INTEGER,
            Discount REAL
        );
    """)
    cursor.execute("""
        CREATE TABLE Products (
            ProductID INTEGER PRIMARY KEY,
            ProductName TEXT,
            UnitPrice REAL
        );
    """)
    cursor.execute("""
        CREATE TABLE Customers (
            CustomerID TEXT PRIMARY KEY,
            CompanyName TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE Categories (
            CategoryID INTEGER PRIMARY KEY,
            CategoryName TEXT
        );
    """)
    cursor.execute("""
        CREATE TABLE Suppliers (
            SupplierID INTEGER PRIMARY KEY,
            CompanyName TEXT
        );
    """)
    cursor.execute("""
        INSERT INTO Orders (OrderID, CustomerID, OrderDate) VALUES (1, 'ALFKI', '1997-01-01');
    """)
    cursor.execute("""
        INSERT INTO "Order Details" (OrderID, ProductID, UnitPrice, Quantity, Discount) VALUES (1, 1, 10.0, 1, 0.0);
    """)
    cursor.execute("""
        INSERT INTO Products (ProductID, ProductName, UnitPrice) VALUES (1, 'Chai', 18.0);
    """)
    cursor.execute("""
        INSERT INTO Customers (CustomerID, CompanyName) VALUES ('ALFKI', 'Alfreds Futterkiste');
    """)

    conn.commit()
    conn.close()
    return db_path

# Test the SqliteTool class
def test_sqlite_tool_connection_and_context_management(test_db_path):
    """Test that the tool connects and closes correctly as a context manager."""
    tool = SqliteTool(db_path=test_db_path)
    assert tool._conn is None  # Should not be connected until __enter__
    with tool:
        assert tool._conn is not None
        assert isinstance(tool._conn, sqlite3.Connection)
    assert tool._conn is not None # Connection object still exists, but should be closed
    # Try to execute something after exiting context, should fail
    with pytest.raises(sqlite3.ProgrammingError): # e.g. "Cannot operate on a closed database."
         tool._conn.execute("SELECT 1")


def test_sqlite_tool_get_table_schema(test_db_path):
    """Test getting schema for allowed tables."""
    with SqliteTool(db_path=test_db_path) as tool:
        orders_schema = tool.get_table_schema("Orders")
        assert "CREATE TABLE Orders" in orders_schema
        assert "OrderID INTEGER PRIMARY KEY" in orders_schema

        order_details_schema = tool.get_table_schema("Order Details")
        assert "CREATE TABLE \"Order Details\"" in order_details_schema
        assert "ProductID INTEGER" in order_details_schema

def test_sqlite_tool_get_table_schema_invalid_table(test_db_path):
    """Test getting schema for a disallowed table raises ValueError."""
    with SqliteTool(db_path=test_db_path) as tool:
        with pytest.raises(ValueError, match="Access to table 'invalid_table' is not allowed."):
            tool.get_table_schema("invalid_table")

def test_sqlite_tool_get_all_schemas(test_db_path):
    """Test getting all schemas returns a combined string."""
    with SqliteTool(db_path=test_db_path) as tool:
        all_schemas = tool.get_all_schemas()
        assert "CREATE TABLE Orders" in all_schemas
        assert "CREATE TABLE Products" in all_schemas
        assert "CREATE TABLE Customers" in all_schemas
        assert "CREATE TABLE \"Order Details\"" in all_schemas
        assert "CREATE TABLE Categories" in all_schemas
        assert "CREATE TABLE Suppliers" in all_schemas
        # Check the order of canonical tables
        assert all_schemas.index("CREATE TABLE Customers") < all_schemas.index("CREATE TABLE Orders")


def test_sqlite_tool_execute_sql_valid_query(test_db_path):
    """Test executing a valid SQL query returns correct data and headers."""
    with SqliteTool(db_path=test_db_path) as tool:
        results = tool.execute_sql("SELECT OrderID, CustomerID FROM Orders WHERE OrderID = 1;")
        assert len(results) == 1
        assert results[0] == {"OrderID": 1, "CustomerID": "ALFKI"}

        products_results = tool.execute_sql("SELECT ProductName, UnitPrice FROM Products WHERE ProductID = 1;")
        assert len(products_results) == 1
        assert products_results[0] == {"ProductName": "Chai", "UnitPrice": 18.0}

def test_sqlite_tool_execute_sql_invalid_query(test_db_path):
    """Test executing an invalid SQL query raises sqlite3.Error."""
    with SqliteTool(db_path=test_db_path) as tool:
        with pytest.raises(sqlite3.Error, match="SQL Error: .*no such column: NonExistentColumn.*"):
            tool.execute_sql("SELECT NonExistentColumn FROM Orders;")

def test_sqlite_tool_execute_sql_no_results(test_db_path):
    """Test executing a query that returns no results."""
    with SqliteTool(db_path=test_db_path) as tool:
        results = tool.execute_sql("SELECT * FROM Orders WHERE OrderID = 999;")
        assert len(results) == 0

def test_sqlite_tool_db_not_found(tmp_path):
    """Test that SqliteTool raises FileNotFoundError if DB does not exist."""
    non_existent_path = tmp_path / "non_existent.sqlite"
    tool = SqliteTool(db_path=non_existent_path)
    with pytest.raises(FileNotFoundError, match="Database file not found"):
        with tool:
            pass
