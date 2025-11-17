"""
Database utilities for PostgreSQL with vector search support.
"""

import os
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import numpy as np
from typing import List, Tuple, Optional, Any
import logging
import subprocess
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager for PostgreSQL with pgvector support."""

    def __init__(self):
        self.host = os.getenv("DB_HOST", "dl-postgres")
        self.port = os.getenv("DB_PORT", "5432")
        self.user = os.getenv("DB_USER", "admin")
        self.password = os.getenv("DB_PASSWORD", "PassW0rd")
        self.database = os.getenv("DB_NAME", "db")
        self.conn = None

    def connect(self):
        """Establish database connection and register vector extension."""
        try:
            self.conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            register_vector(self.conn)
            logger.info("Database connection established")
            return self.conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def __enter__(self):
        self.connect()
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class VectorStore:
    """Vector store operations for similarity search."""

    def __init__(self, db_conn: DatabaseConnection):
        self.db = db_conn

    def create_vector_table(self, table_name: str = "images_features", vector_dim: int = 1024):
        """Create a unified table with vector column for embeddings and separate metadata columns."""
        with self.db as conn:
            cur = conn.cursor()
            cur.execute(f"""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    model_name VARCHAR(50),
                    label VARCHAR(100),
                    augmentation VARCHAR(50),
                    original_image VARCHAR(255),
                    embedding vector({vector_dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info(f"Created vector table: {table_name}")

    def insert_vectors(self, table_name: str, vectors: List[Tuple[str, str, str, str, str, np.ndarray]]):
        """Insert vectors with separate metadata columns."""
        with self.db as conn:
            cur = conn.cursor()
            # Convert numpy arrays to lists for PostgreSQL
            data = [(content, model_name, label, augmentation, original_image, emb.tolist())
                   for content, model_name, label, augmentation, original_image, emb in vectors]
            execute_values(
                cur,
                f"INSERT INTO {table_name} (content, model_name, label, augmentation, original_image, embedding) VALUES %s",
                data,
                template="(%s, %s, %s, %s, %s, %s::vector)"
            )
            conn.commit()
            logger.info(f"Inserted {len(vectors)} vectors into {table_name}")

    def search_similar(self, table_name: str, query_embedding: np.ndarray,
                      limit: int = 10) -> List[Tuple[int, str, float, str, str, str, str]]:
        """Search for similar vectors using cosine distance with normalization."""
        # Normalize the query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm

        with self.db as conn:
            cur = conn.cursor()
            cur.execute(f"""
                SELECT id, content,
                       1 - (embedding <=> %s::vector) as similarity,
                       model_name, label, augmentation, original_image
                FROM {table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding.tolist(), query_embedding.tolist(), limit))

            results = cur.fetchall()
            return results

    def get_vector_count(self, table_name: str) -> int:
        """Get total count of vectors in table."""
        with self.db as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cur.fetchone()[0]


def get_db_connection() -> DatabaseConnection:
    """Factory function for database connection."""
    return DatabaseConnection()


def get_vector_store(db_conn: Optional[DatabaseConnection] = None) -> VectorStore:
    """Factory function for vector store."""
    if db_conn is None:
        db_conn = get_db_connection()
    return VectorStore(db_conn)


def backup_database(backup_dir: str = "./backup", container_name: str = "dl-postgres") -> str:
    """
    Create a backup of the PostgreSQL database using pg_dump through Docker.

    Args:
        backup_dir: Directory to store the backup file
        container_name: Name of the PostgreSQL Docker container

    Returns:
        Path to the created backup file
    """
    # Create backup directory if it doesn't exist
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)

    # Use fixed filename (overwrite existing)
    backup_filename = "wound_classifier_vectors.backup"
    backup_file_path = backup_path / backup_filename

    try:
        # Run pg_dump through Docker
        cmd = [
            "docker", "exec", container_name,
            "pg_dump", "-U", "admin", "-d", "db",
            "--no-owner", "--no-privileges", "--clean", "--if-exists",
            "--format=custom",
            "-f", f"/backup/{backup_filename}"
        ]

        logger.info(f"Creating database backup: {backup_file_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Backup failed: {result.stderr}")
            raise Exception(f"Database backup failed: {result.stderr}")

        logger.info(f"Database backup completed successfully: {backup_file_path}")
        return str(backup_file_path)

    except Exception as e:
        logger.error(f"Failed to create database backup: {e}")
        raise


def restore_database(backup_file_path: Optional[str] = None, container_name: str = "dl-postgres", backup_dir: str = "./backup") -> None:
    """
    Restore the PostgreSQL database from a backup file using pg_restore through Docker.

    Args:
        backup_file_path: Path to the backup file to restore from (if None, uses default backup file)
        container_name: Name of the PostgreSQL Docker container
        backup_dir: Directory containing backup files
    """
    if backup_file_path is None:
        backup_file_path = str(Path(backup_dir) / "wound_classifier_vectors.backup")

    backup_path = Path(backup_file_path)
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_file_path}")

    try:
        # Copy backup file to container
        backup_filename = backup_path.name
        container_backup_path = f"/backup/{backup_filename}"

        # Copy file to container
        copy_cmd = ["docker", "cp", str(backup_path), f"{container_name}:{container_backup_path}"]
        logger.info(f"Copying backup file to container: {backup_file_path}")
        copy_result = subprocess.run(copy_cmd, capture_output=True, text=True)

        if copy_result.returncode != 0:
            logger.error(f"Failed to copy backup file to container: {copy_result.stderr}")
            raise Exception(f"Failed to copy backup file: {copy_result.stderr}")

        # Restore database using pg_restore
        restore_cmd = [
            "docker", "exec", container_name,
            "pg_restore", "-U", "admin", "-d", "db",
            "--clean", "--if-exists", "--no-owner", "--no-privileges",
            "--verbose", container_backup_path
        ]

        logger.info(f"Restoring database from backup: {backup_file_path}")
        restore_result = subprocess.run(restore_cmd, capture_output=True, text=True)

        if restore_result.returncode != 0:
            logger.error(f"Restore failed: {restore_result.stderr}")
            raise Exception(f"Database restore failed: {restore_result.stderr}")

        logger.info("Database restore completed successfully")

    except Exception as e:
        logger.error(f"Failed to restore database: {e}")
        raise


def list_backups(backup_dir: str = "./backup") -> List[str]:
    """
    List all available backup files in the backup directory.

    Args:
        backup_dir: Directory containing backup files

    Returns:
        List of backup file paths sorted by modification time (newest first)
    """
    backup_path = Path(backup_dir)
    if not backup_path.exists():
        return []

    backup_files = []
    for file_path in backup_path.glob("*.backup"):
        backup_files.append(str(file_path))

    # Sort by modification time (newest first)
    backup_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return backup_files