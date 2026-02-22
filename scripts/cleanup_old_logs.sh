#!/bin/bash
# Log Cleanup Script - Remove old log files to save disk space
# Keeps recent logs, removes old ones

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DAYS_TO_KEEP=7
MAX_FILES_TO_KEEP=50
DRY_RUN=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --days)
            DAYS_TO_KEEP="$2"
            shift 2
            ;;
        --max-files)
            MAX_FILES_TO_KEEP="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--dry-run] [--days N] [--max-files N]"
            exit 1
            ;;
    esac
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo -e "${YELLOW}DRY RUN MODE - No files will be deleted${NC}"
fi

echo -e "${BLUE}Log Cleanup Configuration:${NC}"
echo "  Keep logs: Last ${DAYS_TO_KEEP} days OR latest ${MAX_FILES_TO_KEEP} files"
echo ""

# Function to cleanup timestamped logs
cleanup_timestamped_logs() {
    local dir=$1
    local pattern=$2
    local desc=$3
    
    if [[ ! -d "$dir" ]]; then
        return
    fi
    
    echo -e "${BLUE}Cleaning ${desc} in ${dir}/${NC}"
    
    # Count current files
    local total_files
    total_files=$(find "$dir" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | wc -l)
    
    if [[ $total_files -eq 0 ]]; then
        echo "  No files found"
        return
    fi
    
    echo "  Current files: ${total_files}"
    
    # Find old files (older than DAYS_TO_KEEP)
    local old_files
    old_files=$(find "$dir" -maxdepth 1 -name "$pattern" -type f -mtime +${DAYS_TO_KEEP} 2>/dev/null | wc -l)
    
    if [[ $old_files -gt 0 ]]; then
        echo -e "  ${YELLOW}Files older than ${DAYS_TO_KEEP} days: ${old_files}${NC}"
        
        if [[ $DRY_RUN -eq 1 ]]; then
            find "$dir" -maxdepth 1 -name "$pattern" -type f -mtime +${DAYS_TO_KEEP} 2>/dev/null | head -10
            if [[ $old_files -gt 10 ]]; then
                echo "  ... and $(($old_files - 10)) more"
            fi
        else
            find "$dir" -maxdepth 1 -name "$pattern" -type f -mtime +${DAYS_TO_KEEP} -delete 2>/dev/null
            echo -e "  ${GREEN}✓ Deleted ${old_files} old files${NC}"
        fi
    fi
    
    # Keep only latest MAX_FILES_TO_KEEP files
    local remaining
    remaining=$(find "$dir" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | wc -l)
    
    if [[ $remaining -gt $MAX_FILES_TO_KEEP ]]; then
        local to_delete=$((remaining - MAX_FILES_TO_KEEP))
        echo -e "  ${YELLOW}Keeping latest ${MAX_FILES_TO_KEEP}, removing ${to_delete} older files${NC}"
        
        if [[ $DRY_RUN -eq 1 ]]; then
            find "$dir" -maxdepth 1 -name "$pattern" -type f -printf '%T+ %p\n' 2>/dev/null | \
                sort | head -n $to_delete | cut -d' ' -f2- | head -5
            if [[ $to_delete -gt 5 ]]; then
                echo "  ... and $(($to_delete - 5)) more"
            fi
        else
            find "$dir" -maxdepth 1 -name "$pattern" -type f -printf '%T+ %p\n' 2>/dev/null | \
                sort | head -n $to_delete | cut -d' ' -f2- | xargs -r rm -f
            echo -e "  ${GREEN}✓ Deleted ${to_delete} excess files${NC}"
        fi
    fi
    
    # Final count
    local final_count
    final_count=$(find "$dir" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | wc -l)
    echo -e "  ${GREEN}Final count: ${final_count} files${NC}"
    echo ""
}

# Clean up various log directories
cleanup_timestamped_logs "logs/ctrader" "ctrader_*.log" "Python application logs"
cleanup_timestamped_logs "logs/ctrader" "bot_*.log" "Bot logs"
cleanup_timestamped_logs "logs" "run_*.log" "Run logs"
cleanup_timestamped_logs "logs" "training_*.log" "Training logs"
cleanup_timestamped_logs "logs/python" "*.log" "Python logs"
cleanup_timestamped_logs "logs/archived" "*.log" "Archived logs"
cleanup_timestamped_logs "logs/archive" "*.log" "Archive logs"

# Clean up old FIX session logs (but keep structure)
if [[ -d "logs/fix" ]]; then
    echo -e "${BLUE}Cleaning FIX logs...${NC}"
    for subdir in logs/fix/*/; do
        if [[ -d "$subdir" ]]; then
            local fix_files
            fix_files=$(find "$subdir" -name "*.log" -type f -mtime +${DAYS_TO_KEEP} 2>/dev/null | wc -l)
            if [[ $fix_files -gt 0 ]]; then
                echo "  $(basename "$subdir"): ${fix_files} old files"
                if [[ $DRY_RUN -eq 0 ]]; then
                    find "$subdir" -name "*.log" -type f -mtime +${DAYS_TO_KEEP} -delete 2>/dev/null
                fi
            fi
        fi
    done
    echo ""
fi

# Summary
echo -e "${GREEN}=== Cleanup Summary ===${NC}"
if [[ $DRY_RUN -eq 1 ]]; then
    echo -e "${YELLOW}DRY RUN: No files were deleted${NC}"
    echo "Run without --dry-run to actually delete files"
else
    echo "Log cleanup completed successfully"
fi

# Show disk usage
echo ""
echo -e "${BLUE}Current disk usage:${NC}"
du -sh logs/ 2>/dev/null || true
