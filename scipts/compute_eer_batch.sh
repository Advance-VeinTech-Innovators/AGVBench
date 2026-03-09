#!/bin/bash

# Batch script to compute EER for all checkpoints in casia200 directory
# Usage: ./compute_eer_batch.sh [options]

# Default values
SCRIPT_PATH="tools/analysis_tools/compute_eer.py"

BASE_DIR="/yuchang/lsy_jx/AGVBench/work_dirs/classification/_starlknet_ok/scut834"
CONFIG_BASE_DIR="/yuchang/lsy_jx/AGVBench/configs/classification/_starlknet_ok/scut834"
WORK_DIR="/yuchang/lsy_jx/AGVBench/work_dirs/eer"
DATASET="scut834"
NUM_CLASS=834
HEAD="head0"
GPU_ID=0
LAUNCHER="none"

# Parse command line arguments
SKIP_EXISTING=true  # Default to skip existing results to save time
DRY_RUN=false
FILTER_PATTERN=""
CHECK_NPY=true  # Also check npy directory exists

while [[ $# -gt 0 ]]; do
    case $1 in
        --base-dir)
            BASE_DIR="$2"
            shift 2
            ;;
        --config-base-dir)
            CONFIG_BASE_DIR="$2"
            shift 2
            ;;
        --work-dir)
            WORK_DIR="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --num-class)
            NUM_CLASS="$2"
            shift 2
            ;;
        --head)
            HEAD="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --launcher)
            LAUNCHER="$2"
            shift 2
            ;;
        --filter)
            FILTER_PATTERN="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --no-skip-existing)
            SKIP_EXISTING=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --base-dir DIR          Base directory for checkpoints (default: $BASE_DIR)"
            echo "  --config-base-dir DIR   Base directory for configs (default: $CONFIG_BASE_DIR)"
            echo "  --work-dir DIR          Work directory for EER results (default: $WORK_DIR)"
            echo "  --dataset NAME         Dataset name (default: $DATASET)"
            echo "  --num-class N          Number of classes (default: $NUM_CLASS)"
            echo "  --head NAME           Head name (default: $HEAD)"
            echo "  --gpu-id N            GPU ID to use (default: $GPU_ID)"
            echo "  --launcher TYPE        Launcher type: none/pytorch/slurm/mpi (default: $LAUNCHER)"
            echo "  --filter PATTERN       Filter checkpoint paths by pattern (e.g., 'starlknet_tiny')"
            echo "  --skip-existing        Skip if results already exist (default: enabled)"
            echo "  --no-skip-existing     Force recompute even if results exist"
            echo "  --dry-run              Show what would be executed without running"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Find all .pth files
echo "Searching for checkpoint files in: $BASE_DIR"
if [[ -n "$FILTER_PATTERN" ]]; then
    echo "Filter pattern: $FILTER_PATTERN"
fi

# Create array to store checkpoint files
CHECKPOINTS=()
while IFS= read -r -d '' file; do
    if [[ -z "$FILTER_PATTERN" ]] || [[ "$file" == *"$FILTER_PATTERN"* ]]; then
        CHECKPOINTS+=("$file")
    fi
done < <(find "$BASE_DIR" -name "*.pth" -type f -print0)

TOTAL=${#CHECKPOINTS[@]}
echo "Found $TOTAL checkpoint file(s)"
echo ""

if [[ $TOTAL -eq 0 ]]; then
    echo "No checkpoint files found. Exiting."
    exit 1
fi

# Process each checkpoint
SUCCESS=0
FAILED=0
SKIPPED=0

for checkpoint in "${CHECKPOINTS[@]}"; do
    echo "=========================================="
    echo "Processing: $checkpoint"
    
    # Extract model_type and config_name from checkpoint path
    # Path format: .../vera220/{model_type}/{config_name}/epoch_XXX.pth
    rel_path="${checkpoint#$BASE_DIR/}"
    path_parts=($(echo "$rel_path" | tr '/' ' '))
    
    if [[ ${#path_parts[@]} -lt 2 ]]; then
        echo "ERROR: Invalid checkpoint path structure: $checkpoint"
        ((FAILED++))
        continue
    fi
    
    model_type="${path_parts[0]}"
    config_name="${path_parts[1]}"
    epoch_file="${path_parts[2]}"
    
    # Extract base name by removing sz/bs patterns (handles both orders)
    # e.g., "r18_cutmix_sz224_bs32" -> "r18_cutmix"
    # e.g., "r18_cutmix_bs32_sz224" -> "r18_cutmix"
    base_name=$(echo "$config_name" | sed -E 's/_sz[0-9]+_bs[0-9]+//' | sed -E 's/_bs[0-9]+_sz[0-9]+//')
    
    # Construct config file path - try direct match first
    config_file="${CONFIG_BASE_DIR}/${model_type}/${config_name}.py"
    
    # If direct match fails, try to find matching config file (handles sz/bs order differences)
    if [[ ! -f "$config_file" ]]; then
        config_dir="${CONFIG_BASE_DIR}/${model_type}"
        if [[ -d "$config_dir" ]]; then
            # Search for config files that match the base name
            found_config=""
            while IFS= read -r file; do
                file_basename=$(basename "$file" .py)
                # Extract base name from file (same normalization)
                file_base=$(echo "$file_basename" | sed -E 's/_sz[0-9]+_bs[0-9]+//' | sed -E 's/_bs[0-9]+_sz[0-9]+//')
                
                # If base names match, we found it
                if [[ "$file_base" == "$base_name" ]]; then
                    found_config="$file"
                    break
                fi
            done < <(find "$config_dir" -maxdepth 1 -name "*.py" -type f)
            
            if [[ -n "$found_config" ]] && [[ -f "$found_config" ]]; then
                config_file="$found_config"
                echo "INFO: Found config file (matched base name): $config_file"
                echo "      (Original expected: ${CONFIG_BASE_DIR}/${model_type}/${config_name}.py)"
            fi
        fi
    fi
    
    # Final check if config file exists
    if [[ ! -f "$config_file" ]]; then
        echo "ERROR: Config file not found: ${CONFIG_BASE_DIR}/${model_type}/${config_name}.py"
        echo "       Searched in: ${CONFIG_BASE_DIR}/${model_type}/"
        echo "       Base name extracted: $base_name"
        ((FAILED++))
        continue
    fi
    
    # Extract epoch name
    epoch_name="${epoch_file%.pth}"
    
    # Check if results already exist (for skip-existing option)
    if [[ "$SKIP_EXISTING" == true ]]; then
        log_file="${WORK_DIR}/${DATASET}/${model_type}/eer_${epoch_name}_${config_name}.log"
        npy_dir="${WORK_DIR}/${DATASET}/${model_type}/${config_name}/${epoch_name}"
        
        # Check if log file exists
        log_exists=false
        if [[ -f "$log_file" ]]; then
            log_exists=true
        fi
        
        # Check if npy directory exists and contains files
        npy_exists=false
        if [[ "$CHECK_NPY" == true ]] && [[ -d "$npy_dir" ]]; then
            # Check if directory contains at least one .npy file
            if [[ $(find "$npy_dir" -name "*.npy" -type f | wc -l) -gt 0 ]]; then
                npy_exists=true
            fi
        fi
        
        # Skip if both log and npy exist, or if log exists (npy check is optional)
        if [[ "$log_exists" == true ]]; then
            if [[ "$CHECK_NPY" == true ]]; then
                if [[ "$npy_exists" == true ]]; then
                    echo "SKIP: Results already exist (log + npy):"
                    echo "      Log: $log_file"
                    echo "      NPY: $npy_dir"
                    ((SKIPPED++))
                    continue
                else
                    echo "WARN: Log exists but npy missing, will recompute"
                    echo "      Log: $log_file"
                    echo "      NPY: $npy_dir (missing)"
                fi
            else
                echo "SKIP: Log file already exists: $log_file"
                ((SKIPPED++))
                continue
            fi
        fi
    fi
    
    # Build command
    cmd="python $SCRIPT_PATH"
    cmd="$cmd --config $config_file"
    cmd="$cmd --checkpoint $checkpoint"
    cmd="$cmd --dataset $DATASET"
    cmd="$cmd --num_class $NUM_CLASS"
    cmd="$cmd --head $HEAD"
    cmd="$cmd --work_dir $WORK_DIR"
    cmd="$cmd --gpu-id $GPU_ID"
    cmd="$cmd --launcher $LAUNCHER"
    
    echo "Config: $config_file"
    echo "Command: $cmd"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY RUN] Would execute: $cmd"
        ((SUCCESS++))
    else
        # Execute command
        if eval "$cmd"; then
            echo "SUCCESS"
            ((SUCCESS++))
        else
            echo "FAILED"
            ((FAILED++))
        fi
    fi
    echo ""
done

# Summary
echo "=========================================="
echo "Summary:"
echo "  Total: $TOTAL"
echo "  Success: $SUCCESS"
echo "  Failed: $FAILED"
if [[ "$SKIP_EXISTING" == true ]]; then
    echo "  Skipped: $SKIPPED"
fi
echo "=========================================="

