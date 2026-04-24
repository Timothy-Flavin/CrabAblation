#!/bin/bash

# Target the current directory and all subdirectories
echo "Searching for TensorBoard event files..."

# -name: matches the file pattern
# -type f: ensures we only target files, not directories
# -delete: removes the matches found
find . -name "events.out.tfevents.*" -type f -delete

echo "Cleanup complete. Your repository is now a blank slate (for logs, anyway)."