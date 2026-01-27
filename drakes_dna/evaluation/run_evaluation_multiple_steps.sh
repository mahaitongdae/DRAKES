#!/bin/bash

# Script to run DNA model evaluation with multiple num_steps values
# Usage: ./run_evaluation_multiple_steps.sh <model_path> [output_dir]
# Note: This script must be run from the evaluation/ directory

set -e  # Exit on any error

# Check if script is run from the evaluation directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" != "evaluation" ]]; then
    echo "Error: This script must be run from the evaluation/ directory"
    echo "Current directory: $SCRIPT_DIR"
    echo "Please run: cd drakes_dna/evaluation && ./run_evaluation_multiple_steps.sh <model_path>"
    exit 1
fi

# Check if we're in the right project structure
if [[ ! -f "../diffusion_gosai_update.py" ]]; then
    echo "Error: Cannot find diffusion_gosai_update.py in parent directory"
    echo "Please ensure this script is run from the correct project structure"
    exit 1
fi

# Check if model path is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_path> [output_dir]"
    echo "Example: $0 /path/to/model.ckpt"
    echo "Example: $0 /path/to/model.ckpt /path/to/output/dir"
    exit 1
fi

MODEL_PATH="$1"
OUTPUT_DIR="${2:-./evaluation_results}"

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found at $MODEL_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Array of num_steps values to test
# You can modify this array to test different values
NUM_STEPS_VALUES=(32 64 128)

# Other evaluation parameters (modify as needed)
NUM_BATCHES=5
SAMPLES_PER_BATCH=32

echo "=========================================="
echo "DNA Model Evaluation - Multiple Steps Test"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Number of Batches: $NUM_BATCHES"
echo "Samples per Batch: $SAMPLES_PER_BATCH"
echo "Testing num_steps values: ${NUM_STEPS_VALUES[*]}"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo

# Function to run evaluation for a specific num_steps
run_evaluation() {
    local num_steps=$1
    local output_file="$OUTPUT_DIR/evaluation_results_steps_${num_steps}.npz"
    
    echo "Running evaluation with num_steps=$num_steps..."
    echo "Output will be saved to: $output_file"
    
    # Change to the drakes_dna directory to ensure proper Python path
    cd "$(dirname "$0")/.."
    
    # Set PYTHONPATH to include current directory
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    # Run the evaluation script
    python evaluation/evaluate_model.py \
        --model_path "$MODEL_PATH" \
        --num_batches "$NUM_BATCHES" \
        --samples_per_batch "$SAMPLES_PER_BATCH" \
        --num_steps "$num_steps" \
        --output "$output_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed successfully for num_steps=$num_steps"
    else
        echo "✗ Evaluation failed for num_steps=$num_steps"
    fi
    echo
    
    # Return to the original directory
    cd - > /dev/null
}

# Function to create a summary report
create_summary() {
    local summary_file="$OUTPUT_DIR/evaluation_summary.txt"
    
    echo "Creating summary report..."
    echo "Evaluation Summary - $(date)" > "$summary_file"
    echo "Model: $MODEL_PATH" >> "$summary_file"
    echo "Number of Batches: $NUM_BATCHES" >> "$summary_file"
    echo "Samples per Batch: $SAMPLES_PER_BATCH" >> "$summary_file"
    echo "==========================================" >> "$summary_file"
    echo "" >> "$summary_file"
    
    for num_steps in "${NUM_STEPS_VALUES[@]}"; do
        local result_file="$OUTPUT_DIR/evaluation_results_steps_${num_steps}.npz"
        if [ -f "$result_file" ]; then
            echo "num_steps=$num_steps: ✓ Completed - $result_file" >> "$summary_file"
        else
            echo "num_steps=$num_steps: ✗ Failed or missing" >> "$summary_file"
        fi
    done
    
    echo "" >> "$summary_file"
    echo "Summary saved to: $summary_file"
}

# Main execution
echo "Starting evaluation runs..."
echo

# Run evaluation for each num_steps value
for num_steps in "${NUM_STEPS_VALUES[@]}"; do
    run_evaluation "$num_steps"
    
    # Add a small delay between runs to avoid overwhelming the system
    sleep 2
done

# Create summary report
create_summary

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved in: $OUTPUT_DIR"
echo "Summary report: $OUTPUT_DIR/evaluation_summary.txt"
echo "=========================================="

# Optional: Display a quick comparison of results
echo
echo "Quick comparison of results:"
echo "----------------------------"
for num_steps in "${NUM_STEPS_VALUES[@]}"; do
    local result_file="$OUTPUT_DIR/evaluation_results_steps_${num_steps}.npz"
    if [ -f "$result_file" ]; then
        echo "num_steps=$num_steps: $(ls -lh "$result_file" | awk '{print $5}')"
    fi
done
