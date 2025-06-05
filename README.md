# Quick Start Guide - Distributed LLM Scheduler

## Installation


# Clone the repository

git clone 


# Install dependencies
pip install -r requirements.txt

## Running the Code

### 1. Run Complete Evaluation (Recommended)

The easiest way to see everything in action:


python simulation.py


This will:
- Test all 4 scheduling algorithms
- Generate performance graphs in `evaluation_results/` folder
- Print summary statistics to console
- Save detailed results to `evaluation_results/raw_results.csv`

### 2. Test Basic Scheduling

Quick test of all schedulers:


python schedulers.py


This runs a simple 4-task example and shows which tasks complete.

### 3. Visualize DAG Structures

See different types of computational graphs:


python visu.py


This will:
- Generate visualizations of LLM DAG structures
- Show scheduling results with Gantt charts
- Save images to current directory

### 4. Extract and test Real GPT-2 Model 


python test_gpt2.py


This extracts a real GPT-2 computation graph and tests scheduling on it.


### Generated Files:

- `evaluation_results/scheduler_performance.png` - Performance comparison graphs
- `evaluation_results/raw_results.csv` - Detailed results data


