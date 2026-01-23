#!/bin/bash
# =============================================================================
# full.sh - Train meta-models with different soft prompt types
#
# Trains three soft prompt variants:
#   1. CNN soft prompts (default) - uses raw token-level losses
#   2. Average loss soft prompts - uses average loss per step
#   3. Delta probe soft prompts - uses MLP on query-relative histogram delta
#
# Usage:
#   ./scripts/train/full.sh [--submit] [--holdout_tasks] [--architecture_type TYPE]
#
# Options:
#   --submit                  Actually submit jobs to SLURM (default: dry run)
#   --holdout_tasks           Train on 53 tasks, hold out 13 for evaluation (task generalization experiment)
#
# =============================================================================

set -e

# Parse arguments
SUBMIT=false
HOLDOUT_TASKS=false
ARCHITECTURE_TYPE="encoder"

while [[ $# -gt 0 ]]; do
    case $1 in
        --submit)
            SUBMIT=true
            shift
            ;;
        --holdout_tasks)
            HOLDOUT_TASKS=true
            shift
            ;;
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


# Common settings
LOSS_TYPE="quantile"
DATA_DIR="./results/datadecide_train"

# Set output dir based on experiment type
if $HOLDOUT_TASKS; then
    OUTPUT_DIR="./results/task_ablation_final/metamodel"
else
    OUTPUT_DIR="./results/ablation/metamodel"
fi
LR="6e-4"
EPOCHS="3"
BATCH_SIZE="256"
SEEDS="0"

# Task filtering for holdout experiment
TASK_FLAG=""
if $HOLDOUT_TASKS; then
    # Held-out tasks for evaluation (13 tasks, randomly sampled with seed=42)
    # 4 standard + 9 MMLU tasks
    HELD_OUT_TASKS="csqa mmlu_college_biology mmlu_college_chemistry \
mmlu_high_school_mathematics mmlu_high_school_microeconomics mmlu_high_school_psychology \
mmlu_international_law mmlu_management mmlu_professional_psychology mmlu_world_religions \
openbookqa piqa winogrande"

    # Training tasks (53 tasks = 66 - 13 held out)
    # 5 standard tasks
    TRAIN_STANDARD="arc_challenge arc_easy boolq hellaswag socialiqa"
    # 48 MMLU tasks
    TRAIN_MMLU="mmlu_abstract_algebra mmlu_anatomy mmlu_astronomy mmlu_business_ethics \
mmlu_clinical_knowledge mmlu_college_computer_science mmlu_college_mathematics \
mmlu_college_medicine mmlu_college_physics mmlu_computer_security mmlu_conceptual_physics \
mmlu_econometrics mmlu_electrical_engineering mmlu_elementary_mathematics mmlu_formal_logic \
mmlu_global_facts mmlu_high_school_biology mmlu_high_school_chemistry \
mmlu_high_school_computer_science mmlu_high_school_european_history \
mmlu_high_school_geography mmlu_high_school_government_and_politics \
mmlu_high_school_macroeconomics mmlu_high_school_physics mmlu_high_school_statistics \
mmlu_high_school_us_history mmlu_high_school_world_history mmlu_human_aging \
mmlu_human_sexuality mmlu_jurisprudence mmlu_logical_fallacies mmlu_machine_learning \
mmlu_marketing mmlu_medical_genetics mmlu_miscellaneous mmlu_moral_disputes \
mmlu_moral_scenarios mmlu_nutrition mmlu_philosophy mmlu_prehistory \
mmlu_professional_accounting mmlu_professional_law mmlu_professional_medicine \
mmlu_public_relations mmlu_security_studies mmlu_sociology mmlu_us_foreign_policy \
mmlu_virology"

    TARGET_TASKS="$TRAIN_STANDARD $TRAIN_MMLU"
    TASK_FLAG="--target_list \"$TARGET_TASKS\""
fi

echo "=============================================="
echo "Full Meta-model (CNN + Encoder) Training (5 seeds)"
echo "=============================================="
echo "Submit mode: $SUBMIT"
echo "Holdout tasks: $HOLDOUT_TASKS"
echo "Architecture type: $ARCHITECTURE_TYPE"
echo "Output dir: $OUTPUT_DIR"
if $HOLDOUT_TASKS; then
    echo "Training tasks: 53 (13 held out for evaluation)"
fi
echo "Seeds: $SEEDS"
echo ""

# Function to run or print command
run_cmd() {
    local desc="$1"
    shift
    echo "--- $desc ---"
    echo "CMD: $@"
    if $SUBMIT; then
        eval "$@"
        echo ""
    else
        echo "(dry run - use --submit to actually run)"
        echo ""
    fi
}


# for SEED in $SEEDS; do
#     run_cmd "Meta-model cnn (seed=$SEED)" \
#         "sbatch scripts/train_metaloss.slurm $LOSS_TYPE \
#             --predict_accuracy \
#             --soft_prompt_type cnn \
#             --architecture_type $ARCHITECTURE_TYPE \
#             --data_dir $DATA_DIR \
#             --output_dir $OUTPUT_DIR/cnn/seed$SEED \
#             --lr $LR \
#             --epochs $EPOCHS \
#             --batch_size $BATCH_SIZE \
#             --seed $SEED \
#             --max_encoder_tokens 256000 \
#             $TASK_FLAG"
# done

# for SEED in $SEEDS; do
#     run_cmd "Meta-model avg_loss (seed=$SEED)" \
#         "sbatch scripts/train_metaloss.slurm $LOSS_TYPE \
#             --predict_accuracy \
#             --soft_prompt_type avg_loss \
#             --architecture_type $ARCHITECTURE_TYPE \
#             --data_dir $DATA_DIR \
#             --output_dir $OUTPUT_DIR/avg_loss/seed$SEED \
#             --lr $LR \
#             --epochs $EPOCHS \
#             --batch_size $BATCH_SIZE \
#             --seed $SEED \
#             $TASK_FLAG"
# done


for SEED in $SEEDS; do
    run_cmd "Meta-model delta (seed=$SEED)" \
        "sbatch scripts/train_metaloss.slurm $LOSS_TYPE \
            --predict_accuracy \
            --soft_prompt_type delta \
            --architecture_type $ARCHITECTURE_TYPE \
            --num_bins 32 \
            --data_dir $DATA_DIR \
            --output_dir $OUTPUT_DIR/delta/seed$SEED \
            --lr $LR \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --seed $SEED \
            $TASK_FLAG"
done

echo ""
echo "=============================================="
echo "All experiments launched!"
echo "=============================================="
