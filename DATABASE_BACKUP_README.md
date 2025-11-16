# Database Backup and Restore

This project includes utilities for backing up and restoring the PostgreSQL database containing wound classifier vectors.

## Files

- `database_backup.py` - Command-line utility for backup/restore operations
- `core/database.py` - Contains `backup_database()` and `restore_database()` functions

## Usage

### Command Line

```bash
# Create a backup
python database_backup.py backup

# Restore from default backup file
python database_backup.py restore

# Restore from specific backup file
python database_backup.py restore --backup-file ./backup/my_backup.backup

# List available backup files
python database_backup.py list
```

### Python API

```python
from core.database import backup_database, restore_database

# Create backup
backup_file = backup_database()
print(f"Backup created: {backup_file}")

# Restore from default backup
restore_database()

# Restore from specific file
restore_database("./backup/my_backup.backup")
```

## Backup File Location

- Default backup directory: `./backup/`
- Default backup filename: `wound_classifier_vectors.backup`
- The backup uses PostgreSQL's custom format for efficient storage and restoration

## Docker Requirements

The backup and restore functions work with the Docker container `dl-postgres`. Make sure the container is running before performing backup/restore operations.

## Notes

- Backups include all vector data, table schemas, and indexes
- Restore operations will clean existing data before importing
- The backup format preserves vector data types and pgvector extensions