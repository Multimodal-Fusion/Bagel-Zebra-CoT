#!/bin/bash -il

# explicit ordered job list
JOB_SCRIPTS=(
    # 1000 samples
    jobs-19-08-2025/bagel_frozenlake_sft_1000samples.sh
    jobs-19-08-2025/bagel_frozenlake_textual-cot_1000samples.sh
    jobs-19-08-2025/bagel_frozenlake_visual-cot_1000samples.sh
    jobs-19-08-2025/bagel_maze_small_sft_1000samples.sh
    jobs-19-08-2025/bagel_maze_small_textual-cot_1000samples.sh
    jobs-19-08-2025/bagel_maze_small_visual-cot_1000samples.sh
    jobs-19-08-2025/bagel_tetris_small_sft_1000samples.sh
    jobs-19-08-2025/bagel_tetris_small_textual-cot_1000samples.sh
    jobs-19-08-2025/bagel_tetris_small_visual-cot_1000samples.sh
    # 2000 samples
    jobs-19-08-2025/bagel_frozenlake_sft_2000samples.sh
    jobs-19-08-2025/bagel_frozenlake_textual-cot_2000samples.sh
    jobs-19-08-2025/bagel_frozenlake_visual-cot_2000samples.sh
    jobs-19-08-2025/bagel_maze_small_sft_2000samples.sh
    jobs-19-08-2025/bagel_maze_small_textual-cot_2000samples.sh
    jobs-19-08-2025/bagel_maze_small_visual-cot_2000samples.sh
    jobs-19-08-2025/bagel_tetris_small_sft_2000samples.sh
    jobs-19-08-2025/bagel_tetris_small_textual-cot_2000samples.sh
    jobs-19-08-2025/bagel_tetris_small_visual-cot_2000samples.sh
    # 5000 samples
    jobs-19-08-2025/bagel_frozenlake_sft_5000samples.sh
    jobs-19-08-2025/bagel_frozenlake_textual-cot_5000samples.sh
    jobs-19-08-2025/bagel_frozenlake_visual-cot_5000samples.sh
    jobs-19-08-2025/bagel_maze_small_sft_5000samples.sh
    jobs-19-08-2025/bagel_maze_small_textual-cot_5000samples.sh
    jobs-19-08-2025/bagel_maze_small_visual-cot_5000samples.sh
    jobs-19-08-2025/bagel_tetris_small_sft_5000samples.sh
    jobs-19-08-2025/bagel_tetris_small_textual-cot_5000samples.sh
    jobs-19-08-2025/bagel_tetris_small_visual-cot_5000samples.sh
    # 10000 samples
    jobs-19-08-2025/bagel_frozenlake_sft_10000samples.sh
    jobs-19-08-2025/bagel_frozenlake_textual-cot_10000samples.sh
    jobs-19-08-2025/bagel_frozenlake_visual-cot_10000samples.sh
    jobs-19-08-2025/bagel_maze_small_sft_10000samples.sh
    jobs-19-08-2025/bagel_maze_small_textual-cot_10000samples.sh
    jobs-19-08-2025/bagel_maze_small_visual-cot_10000samples.sh
    jobs-19-08-2025/bagel_tetris_small_sft_10000samples.sh
    jobs-19-08-2025/bagel_tetris_small_textual-cot_10000samples.sh
    jobs-19-08-2025/bagel_tetris_small_visual-cot_10000samples.sh
)

echo "Launching jobs..."
# print the number of job scripts
echo "Number of job scripts: ${#JOB_SCRIPTS[@]}"
# also display the job scripts to be launched
printf "%s\n" "${JOB_SCRIPTS[@]}"

# ask for confirmation
read -p "Are you sure you want to launch these jobs? (y/n): " CONFIRM
if [[ "$CONFIRM" != "y" ]]; then
    echo "Exiting..."
    exit 1
fi

echo "Launching jobs..."
for JOB_SCRIPT in "${JOB_SCRIPTS[@]}"; do
    sbatch "$JOB_SCRIPT"
done