import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set
import random

# Minimal Task and Node classes for testing
@dataclass
class Task:
    id: str
    memory_required: float
    compute_time: float
    dependencies: List[str]
    params_needed: Set[str]

@dataclass  
class Node:
    id: str
    total_memory: float
    compute_speed: float = 1.0

# Simple DAG generators for testing
def create_simple_dag():

    tasks = [
        Task("t1", memory_required=1.0, compute_time=0.1, 
             dependencies=[], params_needed={"p1"}),
        Task("t2", memory_required=1.0, compute_time=0.1,
             dependencies=["t1"], params_needed={"p2"}),
        Task("t3", memory_required=1.0, compute_time=0.1,
             dependencies=["t1"], params_needed={"p3"}),
        Task("t4", memory_required=1.0, compute_time=0.1,
             dependencies=["t2", "t3"], params_needed={"p1", "p2"}),
    ]
    return tasks

def create_mini_llm_dag(num_layers=3):
 
    tasks = []
    
    # Embedding
    tasks.append(Task("embedding", 0.5, 0.1, [], {"embedding_weights"}))
    
    # Layers
    for i in range(num_layers):
        prev = "embedding" if i == 0 else f"layer_{i-1}_output"
        
        # Attention
        tasks.append(Task(f"layer_{i}_attention", 0.3, 0.05, 
                         [prev], {f"layer_{i}_attn_weights"}))
        
        # FFN
        tasks.append(Task(f"layer_{i}_ffn", 0.5, 0.1,
                         [f"layer_{i}_attention"], {f"layer_{i}_ffn_weights"}))
        
        # Output
        tasks.append(Task(f"layer_{i}_output", 0.1, 0.02,
                         [f"layer_{i}_ffn"], set()))
    
    # Final output
    tasks.append(Task("output", 0.3, 0.05,
                     [f"layer_{num_layers-1}_output"], {"output_weights"}))
    
    return tasks

def create_random_dag(n=10):
    """Create a random DAG"""
    tasks = []
    for i in range(n):
        deps = []
        if i > 0:
            # Random dependencies from previous tasks
            num_deps = min(random.randint(0, 2), i)
            if num_deps > 0:
                deps = random.sample([f"task_{j}" for j in range(i)], num_deps)
        
        tasks.append(Task(
            f"task_{i}",
            memory_required=random.uniform(0.2, 0.8),
            compute_time=random.uniform(0.05, 0.2),
            dependencies=deps,
            params_needed={f"param_{i}"}
        ))
    return tasks

def visualize_dag_simple(tasks: List[Task], title: str = "Task DAG"):
    """Simple DAG visualization"""
    G = nx.DiGraph()
    
    # Add nodes and edges
    for task in tasks:
        G.add_node(task.id)
        for dep in task.dependencies:
            G.add_edge(dep, task.id)
    
    plt.figure(figsize=(10, 8))
    
    # Layout
    if len(tasks) < 10:
        pos = nx.spring_layout(G, k=3, iterations=50)
    else:
        pos = nx.spring_layout(G)
    
    # Draw
    nx.draw(G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=1500,
            font_size=10,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            edge_color='gray',
            arrowstyle='->')
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_dag_detailed(tasks: List[Task], title: str = "Task DAG"):
    """Detailed DAG visualization with memory and compute info"""
    G = nx.DiGraph()
    
    # Build task map
    task_map = {task.id: task for task in tasks}
    
    # Add nodes with attributes
    for task in tasks:
        G.add_node(task.id, 
                  memory=task.memory_required,
                  compute=task.compute_time)
        for dep in task.dependencies:
            G.add_edge(dep, task.id)
    
    plt.figure(figsize=(12, 10))
    
    # Layout - try hierarchical for LLM
    if any("layer" in t.id for t in tasks):
        # Group by layer for better layout
        shells = []
        if any(t.id == "embedding" for t in tasks):
            shells.append(["embedding"])
        
        max_layer = -1
        for t in tasks:
            if "layer_" in t.id and "_output" in t.id:
                layer_num = int(t.id.split('_')[1])
                max_layer = max(max_layer, layer_num)
        
        for i in range(max_layer + 1):
            layer_nodes = [t.id for t in tasks if f"layer_{i}" in t.id]
            if layer_nodes:
                shells.append(layer_nodes)
        
        if any(t.id == "output" for t in tasks):
            shells.append(["output"])
            
        pos = nx.shell_layout(G, shells)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Node colors by memory
    node_colors = [task_map[node].memory_required for node in G.nodes()]
    
    # Node sizes by compute time
    node_sizes = [1000 + task_map[node].compute_time * 3000 for node in G.nodes()]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos,
                                  node_color=node_colors,
                                  node_size=node_sizes,
                                  cmap='YlOrRd',
                                  vmin=0,
                                  vmax=max(node_colors))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          alpha=0.6,
                          arrowstyle='->')
    
    # Labels
    labels = {}
    for node in G.nodes():
        task = task_map[node]
        labels[node] = f"{node}\n{task.memory_required:.1f}GB\n{task.compute_time:.2f}s"
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', 
                              norm=plt.Normalize(vmin=0, vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(), label='Memory Required (GB)')
    
    plt.title(f"{title}\nNode size = compute time, Color = memory requirement", 
              fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_schedule_simple(schedule: Dict[str, List[str]], 
                            tasks: List[Task],
                            nodes: List[Node]):
    """Simple schedule visualization"""
    task_map = {task.id: task for task in tasks}
    
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    node_colors = {node.id: colors[i % len(colors)] for i, node in enumerate(nodes)}
    
    y_pos = 0
    y_labels = []
    
    for node_id, task_ids in schedule.items():
        node = next(n for n in nodes if n.id == node_id)
        current_time = 0
        
        for task_id in task_ids:
            if task_id in task_map:
                task = task_map[task_id]
                duration = task.compute_time / node.compute_speed
                
                plt.barh(y_pos, duration, left=current_time,
                        height=0.8, color=node_colors[node_id],
                        edgecolor='black', linewidth=1)
                
                # Task label
                plt.text(current_time + duration/2, y_pos, task_id,
                        ha='center', va='center', fontsize=9,
                        color='white', weight='bold')
                
                current_time += duration
        
        y_labels.append(f"{node_id}\n({node.total_memory:.1f}GB)")
        y_pos += 1
    
    plt.yticks(range(len(y_labels)), y_labels)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.title('Task Schedule Gantt Chart', fontsize=14)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

def test_all_visualizations():
    """Test all visualization types"""
    print("Testing DAG Visualizations...\n")
    
    # Test 1: Simple DAG
    print("1. Testing simple 4-task DAG:")
    tasks = create_simple_dag()
    visualize_dag_simple(tasks, "Simple 4-Task DAG")
    visualize_dag_detailed(tasks, "Simple 4-Task DAG (Detailed)")
    
    # Test 2: Mini LLM DAG
    print("\n2. Testing mini LLM DAG:")
    llm_tasks = create_mini_llm_dag(3)
    visualize_dag_simple(llm_tasks, "Mini LLM DAG (3 layers)")
    visualize_dag_detailed(llm_tasks, "Mini LLM DAG (3 layers) - Detailed")
    
    # Test 3: Random DAG
    print("\n3. Testing random DAG:")
    random_tasks = create_random_dag(15)
    visualize_dag_simple(random_tasks, "Random DAG (15 tasks)")
    visualize_dag_detailed(random_tasks, "Random DAG (15 tasks) - Detailed")
    
    # Test 4: Schedule visualization
    print("\n4. Testing schedule visualization:")
    # Create a simple schedule
    nodes = [
        Node("node_0", total_memory=5.0, compute_speed=1.2),
        Node("node_1", total_memory=4.0, compute_speed=1.0),
        Node("node_2", total_memory=3.0, compute_speed=0.8)
    ]
    
    # Manual schedule for testing
    schedule = {
        "node_0": ["t1", "t4"],
        "node_1": ["t2"],
        "node_2": ["t3"]
    }
    
    visualize_schedule_simple(schedule, tasks, nodes)
    
    print("\n5. Testing larger LLM DAG:")
    large_llm = create_mini_llm_dag(6)
    visualize_dag_detailed(large_llm, "Larger LLM DAG (6 layers)")

def interactive_test():
    """Interactive testing"""
    while True:
        print("\n" + "="*50)
        print("DAG Visualization Tester")
        print("="*50)
        print("1. Simple 4-task DAG")
        print("2. Mini LLM DAG (choose layers)")
        print("3. Random DAG (choose size)")
        print("4. Test schedule visualization")
        print("5. Run all tests")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '0':
            break
        elif choice == '1':
            tasks = create_simple_dag()
            visualize_dag_simple(tasks, "Simple 4-Task DAG")
            visualize_dag_detailed(tasks, "Simple 4-Task DAG (Detailed)")
        elif choice == '2':
            n = int(input("Number of layers (1-10): "))
            tasks = create_mini_llm_dag(min(max(n, 1), 10))
            visualize_dag_simple(tasks, f"Mini LLM DAG ({n} layers)")
            visualize_dag_detailed(tasks, f"Mini LLM DAG ({n} layers) - Detailed")
        elif choice == '3':
            n = int(input("Number of tasks (5-50): "))
            tasks = create_random_dag(min(max(n, 5), 50))
            visualize_dag_simple(tasks, f"Random DAG ({n} tasks)")
            visualize_dag_detailed(tasks, f"Random DAG ({n} tasks) - Detailed")
        elif choice == '4':
            tasks = create_simple_dag()
            nodes = [
                Node("GPU_0", total_memory=5.0, compute_speed=1.5),
                Node("CPU_1", total_memory=8.0, compute_speed=1.0),
            ]
            schedule = {
                "GPU_0": ["t1", "t3"],
                "CPU_1": ["t2", "t4"]
            }
            visualize_schedule_simple(schedule, tasks, nodes)
        elif choice == '5':
            test_all_visualizations()
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    print("DAG Visualization Test Script")
    print("This will test various DAG visualization methods\n")
    
    # You can either run all tests automatically
    # test_all_visualizations()
    
    # Or use interactive mode
    interactive_test()