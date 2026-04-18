#!/bin/bash

TARGET_DIR="results"
# Format: YYYY-MM-DD HH:MM:SS TZ
TARGET_DATE="2026-04-17 12:00:00"

echo "Deleting files in $TARGET_DIR older than $TARGET_DATE..."

# -type f: Only look for files
# ! -newermt: NOT newer than the modified time (i.e., older than)
# -delete: removes the matched files
find "$TARGET_DIR" -type f ! -newermt "$TARGET_DATE" -delete

echo "Cleanup complete."

# Optional: If you want to clean up any directories that are now empty, uncomment the line below:
find "$TARGET_DIR" -type d -empty -delete
