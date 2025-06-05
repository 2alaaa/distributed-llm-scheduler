
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
import time
import random
import os
from collections import defaultdict
from typing import List, Dict, Tuple
import copy
from schedulers import *


@dataclass
class TestResult:
    
    scheduler_name: str
    dag_type: str
    memory_regime: float
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    makespan: float
    avg_node_utilization: float
    param_cache_hits: int
    param_cache_misses: int
    load_balance_score: float
    execution_time: float
    completion_rate: float


class DAGGenerator:
  
    
    @staticmethod
    def generate_llm_dag(num_layers: int, layer_width: int = 1, 
                        attention_heads: int = 8, ffn_multiplier: int = 4) -> List[Task]:
       
        tasks = []
        
        # Input embedding
        task = Task("embedding", memory_required=0.5, compute_time=0.1,
                   dependencies=[], params_needed={"embedding_weights"})
        tasks.append(task)
        
        # Transformer layers
        for layer in range(num_layers):
            layer_tasks = []
            
            # Multi-head attention
            for head in range(min(attention_heads, 4)):  # Limit heads for memory constraints
                deps = ["embedding"] if layer == 0 else [f"layer_{layer-1}_output"]
                params = {f"layer_{layer}_attention_head_{head}_weights"}
                task = Task(f"layer_{layer}_attention_head_{head}", 
                          memory_required=0.2, compute_time=0.05,
                          dependencies=deps, params_needed=params)
                tasks.append(task)
                layer_tasks.append(task.id)
            
            # Attention output projection
            task = Task(f"layer_{layer}_attention_output",
                       memory_required=0.3, compute_time=0.05,
                       dependencies=layer_tasks,
                       params_needed={f"layer_{layer}_attention_output_weights"})
            tasks.append(task)
            
            # FFN
            task = Task(f"layer_{layer}_ffn",
                       memory_required=0.5, compute_time=0.1,
                       dependencies=[f"layer_{layer}_attention_output"],
                       params_needed={f"layer_{layer}_ffn_weights"})
            tasks.append(task)
            
            # Layer output
            task = Task(f"layer_{layer}_output",
                       memory_required=0.1, compute_time=0.02,
                       dependencies=[f"layer_{layer}_ffn"],
                       params_needed=set())
            tasks.append(task)
        
        # Output layer
        task = Task("output", memory_required=0.3, compute_time=0.05,
                   dependencies=[f"layer_{num_layers-1}_output"],
                   params_needed={"output_weights"})
        tasks.append(task)
        
        return tasks
    
    @staticmethod
    def generate_random_dag(num_tasks: int, max_deps: int = 3) -> List[Task]:

        tasks = []
        
        for i in range(num_tasks):
            # Determine dependencies (only from previous tasks)
            deps = []
            if i > 0:
                num_deps = min(random.randint(0, min(max_deps, i)), i)
                if num_deps > 0:
                    deps = random.sample([f"task_{j}" for j in range(i)], num_deps)
            
            # Random parameters (fewer to fit in memory)
            num_params = random.randint(1, 2)
            params = {f"param_{i}_{j}" for j in range(num_params)}
            
            task = Task(f"task_{i}",
                       memory_required=random.uniform(0.1, 0.5),
                       compute_time=random.uniform(0.05, 0.15),
                       dependencies=deps,
                       params_needed=params)
            tasks.append(task)
        
        return tasks
    
    @staticmethod
    def generate_pipeline_dag(num_stages: int, width: int = 3) -> List[Task]:

        tasks = []
        
        for stage in range(num_stages):
            stage_outputs = []
            
            for w in range(width):
                task_id = f"stage_{stage}_worker_{w}"
                
                if stage == 0:
                    deps = []
                else:
                    # Each worker depends on all workers from previous stage
                    deps = [f"stage_{stage-1}_worker_{i}" for i in range(width)]
                
                params = {f"stage_{stage}_params"}
                
                task = Task(task_id,
                           memory_required=0.3,
                           compute_time=0.1,
                           dependencies=deps,
                           params_needed=params)
                tasks.append(task)
                stage_outputs.append(task_id)
        
        # Final aggregation
        task = Task("final_output",
                   memory_required=0.2,
                   compute_time=0.05,
                   dependencies=[f"stage_{num_stages-1}_worker_{i}" for i in range(width)],
                   params_needed={"output_params"})
        tasks.append(task)
        
        return tasks


class ImprovedSchedulerEvaluator:

    
    def __init__(self, schedulers: Dict[str, type]):
        self.schedulers = schedulers
        self.results = []
        
    def create_nodes_with_memory_regime(self, total_memory_needed: float, 
                                      memory_regime: float, num_nodes: int = 4) -> List[Node]:
        """Create nodes with specified memory regime"""
        available_memory = total_memory_needed * memory_regime
        
        # Create nodes with varied capabilities
        if num_nodes == 2:
            # Two powerful nodes
            nodes = [
                Node("node_0", total_memory=available_memory * 0.6, compute_speed=1.2),
                Node("node_1", total_memory=available_memory * 0.4, compute_speed=1.0),
            ]
        elif num_nodes == 4:
            # Mix of node types
            memory_fractions = [0.35, 0.25, 0.25, 0.15]
            speeds = [1.2, 1.0, 1.0, 0.8]
            nodes = []
            for i in range(num_nodes):
                nodes.append(Node(f"node_{i}", 
                                total_memory=available_memory * memory_fractions[i],
                                compute_speed=speeds[i]))
        else:
            # Many small nodes
            memory_per_node = available_memory / num_nodes
            nodes = []
            for i in range(num_nodes):
                speed = random.uniform(0.7, 1.3)
                nodes.append(Node(f"node_{i}", 
                                total_memory=memory_per_node,
                                compute_speed=speed))
                
        return nodes
    
    def calculate_total_memory_needed(self, tasks: List[Task]) -> float:
    
        max_concurrent_memory = 0
        
        # Estimate based on task requirements
        for task in tasks:
            task_total = task.memory_required
            # Add parameter memory
            task_total += len(task.params_needed) * 0.5
            max_concurrent_memory = max(max_concurrent_memory, task_total)
        
        # Get all unique parameters
        all_params = set()
        for task in tasks:
            all_params.update(task.params_needed)
        
        # Total if all params loaded once
        total_param_memory = len(all_params) * 0.5
        
        # Estimate: max task memory + all params
        return max_concurrent_memory + total_param_memory
    
    def simulate_execution(self, scheduler: BaseScheduler, 
                          schedule: Dict[str, List[str]]) -> Tuple[float, Dict]:
        if not schedule:
            return 0.0, {'param_cache_hits': 0, 'param_cache_misses': 0, 'node_utilization': {}}
            
        node_finish_times = defaultdict(float)
        task_start_times = {}
        task_finish_times = {}
        
        # Track statistics
        stats = {
            'param_cache_hits': 0,
            'param_cache_misses': 0,
            'node_utilization': defaultdict(float)
        }
        
        # Track which params are cached on each node initially
        initial_cached_params = {}
        for node_id, node in scheduler.nodes.items():
            initial_cached_params[node_id] = set()
        
        # Process scheduled tasks
        for node_id, task_ids in schedule.items():
            if node_id not in scheduler.nodes:
                continue
                
            current_time = 0
            node = scheduler.nodes[node_id]
            cached_params = initial_cached_params[node_id].copy()
            
            for task_id in task_ids:
                if task_id not in scheduler.tasks:
                    continue
                    
                task = scheduler.tasks[task_id]
                
                # Check cache hits/misses
                for param in task.params_needed:
                    if param in cached_params:
                        stats['param_cache_hits'] += 1
                    else:
                        stats['param_cache_misses'] += 1
                        cached_params.add(param)
                
                # Calculate task duration
                task_duration = task.compute_time / node.compute_speed
                
                task_start_times[task_id] = current_time
                current_time += task_duration
                task_finish_times[task_id] = current_time
                
                stats['node_utilization'][node_id] += task_duration
            
            node_finish_times[node_id] = current_time
        
        makespan = max(node_finish_times.values()) if node_finish_times else 0
        
        # Calculate average utilization
        if makespan > 0:
            for node_id in stats['node_utilization']:
                stats['node_utilization'][node_id] /= makespan
        
        return makespan, stats
    
    def calculate_load_balance(self, scheduler: BaseScheduler, 
                             schedule: Dict[str, List[str]]) -> float:
        node_loads = []
        
        for node_id, task_ids in schedule.items():
            if node_id not in scheduler.nodes:
                continue
            node = scheduler.nodes[node_id]
            load = sum(scheduler.tasks[tid].compute_time / node.compute_speed
                      for tid in task_ids if tid in scheduler.tasks)
            node_loads.append(load)
        
        if not node_loads or max(node_loads) == 0:
            return 0
        
        avg_load = np.mean(node_loads)
        std_load = np.std(node_loads)
        
        # Coefficient of variation (inverted and normalized)
        if avg_load > 0:
            cv = std_load / avg_load
            return 1 / (1 + cv)
        return 0
    
    def run_single_test(self, scheduler_class: type, scheduler_name: str,
                       tasks: List[Task], nodes: List[Node], 
                       dag_type: str, memory_regime: float) -> TestResult:
        # Create deep copies to avoid interference
        task_copies = []
        for task in tasks:
            task_copy = Task(task.id, task.memory_required, task.compute_time,
                           task.dependencies.copy(), task.params_needed.copy())
            task_copies.append(task_copy)
            
        node_copies = []
        for node in nodes:
            node_copy = Node(node.id, node.total_memory, node.compute_speed)
            node_copies.append(node_copy)
        
        # Initialize scheduler
        scheduler = scheduler_class(node_copies)
        
        # Add tasks
        for task in task_copies:
            scheduler.add_task(task)
        
        # Time the scheduling
        start_time = time.time()
        try:
            schedule = scheduler.schedule()
        except Exception as e:
            print(f"Error in {scheduler_name}: {e}")
            schedule = {}
        execution_time = time.time() - start_time
        
        # Simulate execution
        makespan, stats = self.simulate_execution(scheduler, schedule)
        
        # Calculate metrics
        completed_tasks = len(scheduler.completed_tasks)
        failed_tasks = len(scheduler.failed_tasks)
        total_tasks = len(tasks)
        completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        avg_utilization = np.mean(list(stats['node_utilization'].values())) \
                         if stats['node_utilization'] else 0
        
        load_balance = self.calculate_load_balance(scheduler, schedule)
        
        return TestResult(
            scheduler_name=scheduler_name,
            dag_type=dag_type,
            memory_regime=memory_regime,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            makespan=makespan,
            avg_node_utilization=avg_utilization,
            param_cache_hits=stats['param_cache_hits'],
            param_cache_misses=stats['param_cache_misses'],
            load_balance_score=load_balance,
            execution_time=execution_time,
            completion_rate=completion_rate
        )
    
    def run_experiments(self, num_runs: int = 5):
        dag_configs = [
            ("LLM-Small", lambda: DAGGenerator.generate_llm_dag(4, attention_heads=4)),
            ("LLM-Medium", lambda: DAGGenerator.generate_llm_dag(8, attention_heads=4)),
            ("LLM-Large", lambda: DAGGenerator.generate_llm_dag(12, attention_heads=4)),
            ("Random-Small", lambda: DAGGenerator.generate_random_dag(30)),
            ("Random-Medium", lambda: DAGGenerator.generate_random_dag(60)),
            ("Pipeline", lambda: DAGGenerator.generate_pipeline_dag(5, width=3))
        ]
        
        memory_regimes = [1.0, 0.9, 0.8]
        node_configs = [2, 4, 8]  # Different numbers of nodes
        
        total_tests = len(dag_configs) * len(memory_regimes) * len(node_configs) * num_runs
        current_test = 0
        
        for dag_name, dag_generator in dag_configs:
            print(f"\nTesting {dag_name} DAGs...")
            
            for num_nodes in node_configs:
                print(f"  With {num_nodes} nodes:")
                
                for memory_regime in memory_regimes:
                    print(f"    Memory regime: {memory_regime*100}%", end='', flush=True)
                    
                    for run in range(num_runs):
                        current_test += 1
                        if run % 2 == 0:
                            print('.', end='', flush=True)
                        
                        # Generate DAG
                        tasks = dag_generator()
                        
                        # Calculate memory needed and create nodes
                        total_memory = self.calculate_total_memory_needed(tasks)
                        nodes = self.create_nodes_with_memory_regime(
                            total_memory, memory_regime, num_nodes)
                        
                        # Test each scheduler
                        for scheduler_name, scheduler_class in self.schedulers.items():
                            try:
                                result = self.run_single_test(
                                    scheduler_class, scheduler_name,
                                    tasks, nodes, dag_name, memory_regime)
                                result.num_nodes = num_nodes  # Add node count
                                self.results.append(result)
                            except Exception as e:
                                print(f"\n      Error with {scheduler_name}: {e}")
                    
                    print(" Done")
        
        print(f"\nCompleted {current_test} test configurations")
    
    def analyze_results(self):
        if not self.results:
            print("No results to analyze!")
            return
            
        # Convert results to DataFrame
        df = pd.DataFrame([{
            'scheduler_name': r.scheduler_name,
            'dag_type': r.dag_type,
            'memory_regime': r.memory_regime,
            'total_tasks': r.total_tasks,
            'completed_tasks': r.completed_tasks,
            'failed_tasks': r.failed_tasks,
            'makespan': r.makespan,
            'avg_node_utilization': r.avg_node_utilization,
            'param_cache_hits': r.param_cache_hits,
            'param_cache_misses': r.param_cache_misses,
            'load_balance_score': r.load_balance_score,
            'execution_time': r.execution_time,
            'completion_rate': r.completion_rate,
            'num_nodes': getattr(r, 'num_nodes', 4)
        } for r in self.results])
        
        # Create output directory
        os.makedirs("evaluation_results", exist_ok=True)
        
        # Save raw results
        df.to_csv('evaluation_results/raw_results.csv', index=False)
        
        # 1. Completion rate by memory regime
        plt.figure(figsize=(12, 8))
        
        # Average across all DAG types
        plt.subplot(2, 2, 1)
        completion_by_memory = df.groupby(['scheduler_name', 'memory_regime'])['completion_rate'].mean().reset_index()
        
        for scheduler in df['scheduler_name'].unique():
            data = completion_by_memory[completion_by_memory['scheduler_name'] == scheduler]
            plt.plot(data['memory_regime'] * 100, data['completion_rate'], 
                    marker='o', label=scheduler, linewidth=2)
        
        plt.xlabel('Memory Regime (%)')
        plt.ylabel('Completion Rate (%)')
        plt.title('Average Task Completion Rate vs Memory Constraints')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Completion rate for LLM DAGs specifically
        plt.subplot(2, 2, 2)
        llm_data = df[df['dag_type'].str.startswith('LLM')]
        llm_completion = llm_data.groupby(['scheduler_name', 'memory_regime'])['completion_rate'].mean().reset_index()
        
        for scheduler in df['scheduler_name'].unique():
            data = llm_completion[llm_completion['scheduler_name'] == scheduler]
            if not data.empty:
                plt.plot(data['memory_regime'] * 100, data['completion_rate'], 
                        marker='s', label=scheduler, linewidth=2)
        
        plt.xlabel('Memory Regime (%)')
        plt.ylabel('Completion Rate (%)')
        plt.title('LLM DAG Completion Rate vs Memory Constraints')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Makespan comparison for completed tasks
        plt.subplot(2, 2, 3)
        # Only consider cases where tasks were completed
        completed_data = df[df['completed_tasks'] > 0]
        if not completed_data.empty:
            makespan_by_dag = completed_data.groupby(['scheduler_name', 'dag_type'])['makespan'].mean().reset_index()
            pivot = makespan_by_dag.pivot(index='dag_type', columns='scheduler_name', values='makespan')
            pivot.plot(kind='bar', ax=plt.gca())
            plt.ylabel('Makespan (seconds)')
            plt.xlabel('DAG Type')
            plt.title('Average Makespan by DAG Type (Completed Tasks Only)')
            plt.xticks(rotation=45)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Load balance score
        plt.subplot(2, 2, 4)
        load_balance_data = df[df['completed_tasks'] > 0].groupby(['scheduler_name', 'memory_regime'])['load_balance_score'].mean().reset_index()
        
        for scheduler in df['scheduler_name'].unique():
            data = load_balance_data[load_balance_data['scheduler_name'] == scheduler]
            if not data.empty:
                plt.plot(data['memory_regime'] * 100, data['load_balance_score'], 
                        marker='^', label=scheduler, linewidth=2)
        
        plt.xlabel('Memory Regime (%)')
        plt.ylabel('Load Balance Score (0-1)')
        plt.title('Load Balance Quality vs Memory Constraints')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('evaluation_results/scheduler_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate summary statistics
        print("\n=== EVALUATION SUMMARY ===")
        summary = df.groupby(['scheduler_name', 'memory_regime']).agg({
            'completion_rate': 'mean',
            'makespan': 'mean',
            'avg_node_utilization': 'mean',
            'load_balance_score': 'mean',
            'execution_time': 'mean'
        }).round(3)
        
        print(summary)
        
        # Best scheduler by metric
        print("\n=== BEST SCHEDULERS BY METRIC ===")
        for memory_regime in [0.8, 0.9, 1.0]:
            print(f"\nAt {memory_regime*100}% memory:")
            regime_data = df[df['memory_regime'] == memory_regime]
            
            if not regime_data.empty:
                # Completion rate
                best_completion = regime_data.groupby('scheduler_name')['completion_rate'].mean().idxmax()
                print(f"  Best Completion Rate: {best_completion} ({regime_data.groupby('scheduler_name')['completion_rate'].mean()[best_completion]:.1f}%)")
                
                # Makespan (only for completed tasks)
                completed_regime = regime_data[regime_data['completed_tasks'] > 0]
                if not completed_regime.empty:
                    best_makespan = completed_regime.groupby('scheduler_name')['makespan'].mean().idxmin()
                    print(f"  Best Makespan: {best_makespan} ({completed_regime.groupby('scheduler_name')['makespan'].mean()[best_makespan]:.3f}s)")
                
                # Load balance
                if not completed_regime.empty:
                    best_balance = completed_regime.groupby('scheduler_name')['load_balance_score'].mean().idxmax()
                    print(f"  Best Load Balance: {best_balance} ({completed_regime.groupby('scheduler_name')['load_balance_score'].mean()[best_balance]:.3f})")
        
        # LLM-specific results
        print("\n=== LLM DAG RESULTS ===")
        llm_results = df[df['dag_type'].str.startswith('LLM')]
        llm_summary = llm_results.groupby(['scheduler_name', 'memory_regime']).agg({
            'completion_rate': 'mean',
            'makespan': 'mean',
            'param_cache_hits': 'sum',
            'param_cache_misses': 'sum'
        }).round(3)
        
        # Add cache hit rate
        llm_summary['cache_hit_rate'] = llm_summary['param_cache_hits'] / (llm_summary['param_cache_hits'] + llm_summary['param_cache_misses'])
        
        print(llm_summary[['completion_rate', 'makespan', 'cache_hit_rate']])


def main():
    print("Starting Scheduler Evaluation...")
    
    # Import schedulers here
    schedulers = {
        "DFS": DFSScheduler,
        "Greedy": GreedyScheduler,
        "Critical": CriticalPathScheduler,
        "MRU_spec": MRUScheduler,
    }
    
    # Create evaluator
    evaluator = ImprovedSchedulerEvaluator(schedulers)
    
    # Run experiments
    evaluator.run_experiments(num_runs=3)  # Reduced for faster testing
    
    # Analyze results
    evaluator.analyze_results()
    
    print("\nEvaluation complete! Check 'evaluation_results' directory for outputs.")


if __name__ == "__main__":
    main()