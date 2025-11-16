#!/usr/bin/env python3
"""
Database backup and restore utility for wound classifier vectors.
"""

import sys
import argparse
from pathlib import Path

# Add core to path
sys.path.append(str(Path(__file__).parent / 'core'))

from core.database import backup_database, restore_database, list_backups


def main():
    parser = argparse.ArgumentParser(description='Database backup and restore utility')
    parser.add_argument('action', choices=['backup', 'restore', 'list'],
                       help='Action to perform')
    parser.add_argument('--backup-file', '-f',
                       help='Backup file path for restore action (optional, uses default if not specified)')
    parser.add_argument('--backup-dir', '-d', default='./backup',
                       help='Backup directory (default: ./backup)')
    parser.add_argument('--container', '-c', default='dl-postgres',
                       help='Docker container name (default: dl-postgres)')

    args = parser.parse_args()

    try:
        if args.action == 'backup':
            print("Creating database backup...")
            backup_file = backup_database(args.backup_dir, args.container)
            print(f"✅ Backup created successfully: {backup_file}")

        elif args.action == 'restore':
            print(f"Restoring database from: {args.backup_file or 'default backup file'}")
            restore_database(args.backup_file, args.container, args.backup_dir)
            print("✅ Database restored successfully")

        elif args.action == 'list':
            print("Available backup files:")
            backups = list_backups(args.backup_dir)
            if not backups:
                print("  No backup files found")
            else:
                for i, backup in enumerate(backups, 1):
                    backup_path = Path(backup)
                    size_mb = backup_path.stat().st_size / (1024 * 1024)
                    print(f"  {i}. {backup_path.name} ({size_mb:.1f} MB)")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()