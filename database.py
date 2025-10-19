"""
SQL database operations
"""

import sqlite3
import pandas as pd
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Handle all database operations"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"Connected to database: {self.db_path}")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def save_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace'):
        """
        Save dataframe to database table

        Args:
            df: DataFrame to save
            table_name: Name of table
            if_exists: What to do if table exists ('replace', 'append', 'fail')
        """
        logger.info(f"Saving {len(df)} rows to table '{table_name}'...")

        df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)

        # Create index on timestamp for faster queries
        try:
            self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp)")
            self.conn.commit()
            logger.info(f"Created index on timestamp column")
        except Exception as e:
            logger.warning(f"Could not create index: {e}")

        logger.info(f"✓ Saved to table '{table_name}'")

    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame

        Args:
            query: SQL query string

        Returns:
            Query results as DataFrame
        """
        logger.info("Executing query...")
        result = pd.read_sql_query(query, self.conn)
        logger.info(f"✓ Query returned {len(result)} rows")
        return result

    def execute_query_file(self, filepath: str) -> List[pd.DataFrame]:
        """
        Execute all queries from a SQL file

        Args:
            filepath: Path to .sql file

        Returns:
            List of DataFrames (one per query)
        """
        logger.info(f"Executing queries from: {filepath}")

        with open(filepath, 'r') as f:
            sql_content = f.read()

        # Split on semicolons and filter empty queries
        queries = [q.strip() for q in sql_content.split(';') if q.strip() and not q.strip().startswith('--')]

        results = []
        for i, query in enumerate(queries, 1):
            try:
                logger.info(f"Executing query {i}/{len(queries)}...")
                result = self.execute_query(query)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing query {i}: {e}")
                logger.error(f"Query: {query[:100]}...")

        logger.info(f"✓ Executed {len(results)} queries successfully")
        return results

    def get_table_info(self, table_name: str) -> Dict:
        """Get information about a table"""
        info = {}

        # Row count
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        info['row_count'] = row_count

        # Column info
        cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        info['columns'] = [{'name': col[1], 'type': col[2]} for col in columns]

        return info