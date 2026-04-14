#!/bin/sh
# Warn if total size of staged files exceeds 1GB

MAX_TOTAL_SIZE=1073741824  # 1GB in bytes
total_size=0

# Get list of staged files that are Added (A), Copied (C), or Modified (M)
files=$(git diff --cached --name-only --diff-filter=ACM)

for file in $files; do
  if [ -f "$file" ]; then
    size=$(stat -c%s "$file")
    total_size=$((total_size + size))
  fi
done

if [ "$total_size" -gt "$MAX_TOTAL_SIZE" ]; then
  echo "Warning: Total size of staged files is $((total_size / 1024 / 1024)) MB, which exceeds the 1GB limit."
  echo "Consider reviewing the files before committing."
  # Exit with non-zero status to prevent the commit
  exit 1
fi

exit 0
