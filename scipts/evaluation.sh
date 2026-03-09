#! /bin/bash

# Configuration
BASE_DIR="/yuchang/lsy_jx/AGVBench/work_dirs/classification/_starlknet_ok/scut834"
EPOCH_NUM=600
RECORD_NUM=10
KEY="head0_top1"

# Optional: Filter pattern (leave empty to process all .json files)
# Examples:
#   FILTER_PATTERN="*vgg16*.json"  # Only process files containing "vgg16"
#   FILTER_PATTERN="*vanilla*.json" # Only process files containing "vanilla"
FILTER_PATTERN=""

# Optional: Exclude pattern (files matching this will be skipped)
# Examples:
#   EXCLUDE_PATTERN="*test*.json"  # Skip test files
EXCLUDE_PATTERN=""

# Find all JSON files in BASE_DIR recursively
if [ -z "$FILTER_PATTERN" ]; then
    # Find all .json files
    JSON_FILES=$(find "$BASE_DIR" -type f -name "*.json" | sort)
else
    # Find files matching the filter pattern
    JSON_FILES=$(find "$BASE_DIR" -type f -name "$FILTER_PATTERN" | sort)
fi

# Filter out excluded files if EXCLUDE_PATTERN is set
if [ -n "$EXCLUDE_PATTERN" ]; then
    JSON_FILES=$(echo "$JSON_FILES" | grep -v "$EXCLUDE_PATTERN")
fi

# Check if any files were found
if [ -z "$JSON_FILES" ]; then
    echo "No JSON files found in $BASE_DIR"
    if [ -n "$FILTER_PATTERN" ]; then
        echo "Filter pattern: $FILTER_PATTERN"
    fi
    exit 1
fi

# Count total files
TOTAL_FILES=$(echo "$JSON_FILES" | wc -l)
echo "Found $TOTAL_FILES JSON file(s) to process"
if [ -n "$FILTER_PATTERN" ]; then
    echo "Filter pattern: $FILTER_PATTERN"
fi
if [ -n "$EXCLUDE_PATTERN" ]; then
    echo "Exclude pattern: $EXCLUDE_PATTERN"
fi
echo "=========================================="
echo ""

# Process each JSON file
COUNTER=0
while IFS= read -r json_path; do
    COUNTER=$((COUNTER + 1))
    echo "[$COUNTER/$TOTAL_FILES] Processing: $json_path"
    
    # Check if file exists
    if [ ! -f "$json_path" ]; then
        echo "  Warning: File not found, skipping..."
        echo "---"
        continue
    fi
    
    # Run the Python script
    python tools/summary/find_classification_val_median.py \
        "$json_path" "$EPOCH_NUM" "$RECORD_NUM" "$KEY"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed"
    fi
    echo "---"
done <<< "$JSON_FILES"

echo ""
echo "=========================================="
echo "Completed processing $COUNTER file(s)"