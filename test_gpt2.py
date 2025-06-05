
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from typing import List, Dict, Set
import numpy as np

# Import your Task class
from schedulers import Task

class LLMDAGExtractor:
  
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.tasks = []
        self.task_id_counter = 0
        
    def estimate_memory_gb(self, module: nn.Module, input_shape: tuple) -> float:
        """Estimate memory requirement for a module"""
        # Count parameters
        param_memory = sum(p.numel() * 4 for p in module.parameters()) / 1e9  # 4 bytes per float32
        
        # Estimate activation memory (rough approximation)
        if hasattr(module, 'weight'):
            weight_shape = module.weight.shape
            batch_size = input_shape[0] if input_shape else 1
            activation_memory = np.prod(weight_shape) * batch_size * 4 / 1e9
        else:
            activation_memory = 0.1  # Default for non-parameterized layers
        
        return param_memory + activation_memory
    
    def estimate_compute_time(self, module: nn.Module) -> float:
        """Estimate computation time (relative units)"""
        if isinstance(module, nn.MultiheadAttention):
            return 0.05  # Attention is relatively fast
        elif isinstance(module, nn.Linear):
            # FFN layers are typically larger
            if hasattr(module, 'out_features') and module.out_features > 2048:
                return 0.1  # Large FFN
            return 0.03
        else:
            return 0.02  # Other operations
    
    def extract_gpt2_dag(self) -> List[Task]:
        # Load model configuration
        config = GPT2Config.from_pretrained(self.model_name)
        model = GPT2Model(config)
        
        tasks = []
        
        # Input embedding
        embedding_memory = self.estimate_memory_gb(model.wte, (1, 512))  # 512 tokens
        tasks.append(Task(
            "embedding",
            memory_required=embedding_memory,
            compute_time=0.1,
            dependencies=[],
            params_needed={"embedding_weights", "position_weights"}
        ))
        
        # Extract transformer layers
        for layer_idx in range(config.n_layer):
            prev_output = "embedding" if layer_idx == 0 else f"layer_{layer_idx-1}_output"
            
            # Attention block
            attn_module = model.h[layer_idx].attn
            attn_memory = self.estimate_memory_gb(attn_module, (1, 512, config.n_embd))
            
            # Layer norm before attention
            tasks.append(Task(
                f"layer_{layer_idx}_ln1",
                memory_required=0.01,  # LayerNorm is lightweight
                compute_time=0.01,
                dependencies=[prev_output],
                params_needed={f"layer_{layer_idx}_ln1_weights"}
            ))
            
            # Multi-head attention
            tasks.append(Task(
                f"layer_{layer_idx}_attention",
                memory_required=attn_memory,
                compute_time=0.05,
                dependencies=[f"layer_{layer_idx}_ln1"],
                params_needed={
                    f"layer_{layer_idx}_attn_qkv_weights",
                    f"layer_{layer_idx}_attn_proj_weights"
                }
            ))
            
            # Residual connection
            tasks.append(Task(
                f"layer_{layer_idx}_attn_residual",
                memory_required=0.01,
                compute_time=0.01,
                dependencies=[f"layer_{layer_idx}_attention", prev_output],
                params_needed=set()
            ))
            
            # FFN block
            mlp = model.h[layer_idx].mlp
            
            # Layer norm before FFN
            tasks.append(Task(
                f"layer_{layer_idx}_ln2",
                memory_required=0.01,
                compute_time=0.01,
                dependencies=[f"layer_{layer_idx}_attn_residual"],
                params_needed={f"layer_{layer_idx}_ln2_weights"}
            ))
            
            # FFN expand (4x hidden size in GPT-2)
            ffn_memory = self.estimate_memory_gb(mlp.c_fc, (1, 512, config.n_embd))
            tasks.append(Task(
                f"layer_{layer_idx}_ffn_expand",
                memory_required=ffn_memory,
                compute_time=0.08,
                dependencies=[f"layer_{layer_idx}_ln2"],
                params_needed={f"layer_{layer_idx}_ffn_expand_weights"}
            ))
            
            # FFN activation (GELU)
            tasks.append(Task(
                f"layer_{layer_idx}_ffn_activation",
                memory_required=0.01,
                compute_time=0.01,
                dependencies=[f"layer_{layer_idx}_ffn_expand"],
                params_needed=set()
            ))
            
            # FFN contract
            tasks.append(Task(
                f"layer_{layer_idx}_ffn_contract",
                memory_required=ffn_memory,
                compute_time=0.08,
                dependencies=[f"layer_{layer_idx}_ffn_activation"],
                params_needed={f"layer_{layer_idx}_ffn_contract_weights"}
            ))
            
            # Final residual
            tasks.append(Task(
                f"layer_{layer_idx}_output",
                memory_required=0.01,
                compute_time=0.01,
                dependencies=[f"layer_{layer_idx}_ffn_contract", f"layer_{layer_idx}_attn_residual"],
                params_needed=set()
            ))
        
        # Final layer norm
        tasks.append(Task(
            "final_ln",
            memory_required=0.01,
            compute_time=0.01,
            dependencies=[f"layer_{config.n_layer-1}_output"],
            params_needed={"final_ln_weights"}
        ))
        
        # Output projection (for language modeling)
        output_memory = self.estimate_memory_gb(model.wte, (1, 512))  # Reuses embedding weights
        tasks.append(Task(
            "output_projection",
            memory_required=output_memory,
            compute_time=0.1,
            dependencies=["final_ln"],
            params_needed={"embedding_weights"}  # Weight tying
        ))
        
        return tasks
    
    def extract_from_traced_model(self, model: nn.Module, sample_input: torch.Tensor) -> List[Task]:
        tasks = []
        execution_order = []
        
        def hook_fn(module, input, output, name):
            """Hook to track execution"""
            execution_order.append({
                'name': name,
                'module': module,
                'input_shape': input[0].shape if input else None,
                'output_shape': output.shape if hasattr(output, 'shape') else None
            })
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                hooks.append(hook)
        
        # Run forward pass
        with torch.no_grad():
            model(sample_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Convert execution order to tasks
        for i, exec_info in enumerate(execution_order):
            dependencies = []
            # Simple dependency: previous operation
            if i > 0:
                dependencies.append(f"op_{i-1}")
            
            task = Task(
                f"op_{i}_{exec_info['name'].replace('.', '_')}",
                memory_required=self.estimate_memory_gb(exec_info['module'], exec_info['input_shape']),
                compute_time=self.estimate_compute_time(exec_info['module']),
                dependencies=dependencies,
                params_needed={f"{exec_info['name']}_params"} if list(exec_info['module'].parameters()) else set()
            )
            tasks.append(task)
        
        return tasks
    
    def analyze_dag(self, tasks: List[Task]):
        print(f"DAG Analysis:")
        print(f"Total tasks: {len(tasks)}")
        
        # Memory analysis
        total_memory = sum(t.memory_required for t in tasks)
        max_memory = max(t.memory_required for t in tasks)
        print(f"Total memory (if sequential): {total_memory:.2f} GB")
        print(f"Max single task memory: {max_memory:.2f} GB")
        
        # Parameter analysis
        all_params = set()
        for task in tasks:
            all_params.update(task.params_needed)
        print(f"Unique parameters: {len(all_params)}")
        print(f"Parameter memory: {len(all_params) * 0.5:.2f} GB")
        
        # Compute analysis
        total_compute = sum(t.compute_time for t in tasks)
        print(f"Total compute time (sequential): {total_compute:.2f} seconds")
        
        # Dependency analysis
        max_deps = max(len(t.dependencies) for t in tasks)
        avg_deps = np.mean([len(t.dependencies) for t in tasks])
        print(f"Max dependencies: {max_deps}")
        print(f"Avg dependencies: {avg_deps:.2f}")


def test_extraction():
    extractor = LLMDAGExtractor("gpt2") 
    
    print("Extracting DAG from GPT-2...")
    tasks = extractor.extract_gpt2_dag()
    
    print(f"\nExtracted {len(tasks)} tasks")
    
    # Show first few tasks
    print("\nFirst 5 tasks:")
    for task in tasks[:5]:
        print(f"  {task.id}: mem={task.memory_required:.3f}GB, "
              f"compute={task.compute_time:.3f}s, "
              f"deps={task.dependencies}")
    
    # Analyze the DAG
    print("\n")
    extractor.analyze_dag(tasks)
    
    # Save DAG for scheduling
    import pickle
    with open("gpt2_dag.pkl", "wb") as f:
        pickle.dump(tasks, f)
    print("\nDAG saved to gpt2_dag.pkl")
    
    return tasks


def test_with_your_scheduler(tasks: List[Task]):
    from schedulers import MRUScheduler, GreedyScheduler, Node
    
    # Create realistic nodes (e.g., 4 laptops with 8GB each)
    nodes = [
        Node("laptop_0", total_memory=8.0, compute_speed=1.0),
        Node("laptop_1", total_memory=8.0, compute_speed=1.2),
        Node("laptop_2", total_memory=6.0, compute_speed=0.8),
        Node("laptop_3", total_memory=6.0, compute_speed=0.9),
    ]
    
    print("\nTesting MRU Scheduler on real GPT-2 DAG...")
    
    # Test MRU
    mru_scheduler = MRUScheduler(nodes)
    for task in tasks:
        mru_scheduler.add_task(task)
    
    schedule = mru_scheduler.schedule()
    
    print(f"MRU Results:")
    print(f"  Completed: {len(mru_scheduler.completed_tasks)}/{len(tasks)}")
    print(f"  Failed: {len(mru_scheduler.failed_tasks)}")
    
    for node_id, task_ids in schedule.items():
        print(f"  {node_id}: {len(task_ids)} tasks")


if __name__ == "__main__":
    # Extract DAG from real model
    tasks = test_extraction()
    
    # Test with your schedulers
    test_with_your_scheduler(tasks)