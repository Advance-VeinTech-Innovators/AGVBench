#!/bin/bash

# Configuration
BASE_DIR="/home/being/fym/AGVBench/AGVBench/work_dirs/classification/tju600"
EPOCH_NUM=600
RECORD_NUM=10
KEY="head0_top1-acc_aug_q_top1"

# FILTER_PATTERN 现在会匹配整个路径中的“文件夹名”
# 例如：FILTER_PATTERN="*ampvnet*" 会匹配任何层级中名字包含 ampvnet 的文件夹
FILTER_PATTERN="*swin*"

# 排除模式
EXCLUDE_PATTERN=""

echo "Scanning all levels of directories in: $BASE_DIR"
echo "=========================================="

# 1. 递归寻找符合条件的目录 (去掉 -maxdepth 1)
DIR_PATTERN="${FILTER_PATTERN:-*}"

# 找到所有匹配该模式的目录
# 注意：这里会搜索 BASE_DIR 下所有深度的文件夹
MATCHED_DIRS=$(find "$BASE_DIR" -type d -name "$DIR_PATTERN" | sort)

if [ -z "$MATCHED_DIRS" ]; then
    echo "No directories found matching pattern: $DIR_PATTERN"
    exit 1
fi

# 2. 在匹配到的目录中搜集当前层级的 JSON 文件
JSON_FILES=""
while IFS= read -r dir; do
    # -maxdepth 1 确保只取该文件夹下的 json，不往更深处找，防止重复处理
    found_jsons=$(find "$dir" -maxdepth 1 -type f -name "*.json")
    if [ -n "$found_jsons" ]; then
        # 逐行添加到列表中，处理路径中可能的空格
        while IFS= read -r f; do
            JSON_FILES+="$f"$'\n'
        done <<< "$found_jsons"
    fi
done <<< "$MATCHED_DIRS"

# 3. 过滤掉不需要的文件 (使用 grep -v)
if [ -n "$EXCLUDE_PATTERN" ]; then
    JSON_FILES=$(echo "$JSON_FILES" | grep -v -E "$EXCLUDE_PATTERN" | grep . )
else
    JSON_FILES=$(echo "$JSON_FILES" | grep . )
fi

# 检查最终文件列表
if [ -z "$JSON_FILES" ]; then
    echo "No JSON files found in the matched directories."
    exit 1
fi

TOTAL_FILES=$(echo "$JSON_FILES" | wc -l)
echo "Found $TOTAL_FILES JSON file(s) across all matching directories."
echo "Filter pattern (directory name): $DIR_PATTERN"
echo "------------------------------------------"

# 4. 循环处理文件
COUNTER=0
while IFS= read -r json_path; do
    [ -z "$json_path" ] && continue
    COUNTER=$((COUNTER + 1))
    echo "[$COUNTER/$TOTAL_FILES] Processing: $json_path"

    # 执行 Python 脚本
    python tools/summary/find_classification_val_median.py \
        "$json_path" "$EPOCH_NUM" "$RECORD_NUM" "$KEY"

    if [ $? -eq 0 ]; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed"
    fi
    echo "---"
done <<< "$JSON_FILES"

echo "=========================================="
echo "Completed processing $COUNTER file(s)"