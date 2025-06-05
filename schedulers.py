import heapq
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import time
import copy

class Task:

    def __init__(self, task_id: str, memory_required: float, compute_time: float, 
                 dependencies: List[str] = None, params_needed: Set[str] = None):
        self.id = task_id
        self.memory_required = memory_required  # in GB
        self.compute_time = compute_time  # in seconds
        self.dependencies = dependencies or []
        self.params_needed = params_needed or set()  # model parameters needed
        self.completed = False
        self.assigned_node = None
        
class Node:
   
    def __init__(self, node_id: str, total_memory: float, compute_speed: float = 1.0):
        self.id = node_id
        self.total_memory = total_memory  # in GB
        self.available_memory = total_memory
        self.compute_speed = compute_speed  # relative speed multiplier
        self.cached_params = set()  # currently cached model parameters
        self.running_tasks = []
        self.completed_tasks = []
        self.last_used_params = deque(maxlen=10)  # for MRU tracking
        
class BaseScheduler:
   
    def __init__(self, nodes: List[Node]):
        self.nodes = {node.id: node for node in nodes}
        self.tasks = {}
        self.ready_queue = []
        self.completed_tasks = set()
        self.failed_tasks = set()
        self.task_dependencies = defaultdict(list)
        self.param_locations = defaultdict(set)  # param_id -> set of node_ids
        self.pending_tasks = set()  # Tasks not yet scheduled
        
    def add_task(self, task: Task):
 
        self.tasks[task.id] = task
        self.pending_tasks.add(task.id)
        for dep in task.dependencies:
            self.task_dependencies[dep].append(task.id)
            
    def is_task_ready(self, task_id: str) -> bool:

        task = self.tasks[task_id]
        return all(dep in self.completed_tasks for dep in task.dependencies)
    
    def get_ready_tasks(self) -> List[Task]:

        ready = []
        for task_id in self.pending_tasks:
            if self.is_task_ready(task_id):
                ready.append(self.tasks[task_id])
        return ready
    
    def calculate_memory_requirement(self, task: Task, node: Node) -> float:
    
        # Base task memory
        memory_needed = task.memory_required
        
        # Add memory for parameters not cached on this node
        params_to_load = task.params_needed - node.cached_params
        memory_needed += len(params_to_load) * 0.5  # 0.5GB per param
        
        return memory_needed
    
    def can_fit_on_node(self, task: Task, node: Node) -> bool:
  
        return self.calculate_memory_requirement(task, node) <= node.available_memory
    
    def assign_task_to_node(self, task: Task, node: Node) -> bool:
       
        memory_needed = self.calculate_memory_requirement(task, node)
        
        if memory_needed > node.available_memory:
            return False
            
        # Load parameters if needed
        params_to_load = task.params_needed - node.cached_params
        for param in params_to_load:
            node.cached_params.add(param)
            node.available_memory -= 0.5
            self.param_locations[param].add(node.id)
            
        # Assign task
        task.assigned_node = node.id
        node.running_tasks.append(task.id)
        node.available_memory -= task.memory_required
        self.pending_tasks.discard(task.id)
        
        # Update MRU for MRU-enhanced scheduler
        node.last_used_params.extend(task.params_needed)
        
        # Mark as completed (simulating immediate execution)
        self.complete_task(task.id)
        
        return True
        
    def complete_task(self, task_id: str):
     
        if task_id not in self.tasks:
            return
            
        task = self.tasks[task_id]
        if not task.assigned_node:
            return
            
        node = self.nodes[task.assigned_node]
        
        task.completed = True
        self.completed_tasks.add(task_id)
        self.pending_tasks.discard(task_id)
        
        if task_id in node.running_tasks:
            node.running_tasks.remove(task_id)
        node.completed_tasks.append(task_id)
        
        # Free task memory (but keep params cached)
        node.available_memory += task.memory_required
        
    def fail_task(self, task_id: str):
      
        self.failed_tasks.add(task_id)
        self.pending_tasks.discard(task_id)
        
    def schedule(self) -> Dict[str, List[str]]:
        
        raise NotImplementedError


class DFSScheduler(BaseScheduler):
 
    def compute_depth(self, task_id: str, memo: Dict[str, int]) -> int:
      
        if task_id in memo:
            return memo[task_id]
            
        task = self.tasks[task_id]
        if not task.dependencies:
            depth = 0
        else:
            depth = 1 + max(self.compute_depth(dep, memo) for dep in task.dependencies)
            
        memo[task_id] = depth
        return depth
    
    def schedule(self) -> Dict[str, List[str]]:

        schedule = defaultdict(list)
        
        # Compute depths for all tasks
        depths = {}
        for task_id in self.tasks:
            self.compute_depth(task_id, depths)
        
        # Keep scheduling until no more tasks can be scheduled
        iterations = 0
        max_iterations = len(self.tasks) * 2  # Prevent infinite loops
        
        while self.pending_tasks and iterations < max_iterations:
            iterations += 1
            
            # Get ready tasks
            ready_tasks = self.get_ready_tasks()
            if not ready_tasks:
                # No more tasks can be scheduled
                break
                
            # Sort by depth (deeper first)
            ready_tasks.sort(key=lambda t: depths.get(t.id, 0), reverse=True)
            
            scheduled_this_round = False
            
            for task in ready_tasks:
                if task.id not in self.pending_tasks:
                    continue
                    
                # Try to assign to node with most available memory
                best_node = None
                max_memory = -1
                
                for node in self.nodes.values():
                    if self.can_fit_on_node(task, node) and node.available_memory > max_memory:
                        best_node = node
                        max_memory = node.available_memory
                        
                if best_node:
                    if self.assign_task_to_node(task, best_node):
                        schedule[best_node.id].append(task.id)
                        scheduled_this_round = True
                else:
                    # Can't schedule this task
                    self.fail_task(task.id)
                    
            if not scheduled_this_round:
                # No progress made, fail remaining tasks
                for task_id in list(self.pending_tasks):
                    self.fail_task(task_id)
                break
                
        return dict(schedule)


class GreedyScheduler(BaseScheduler):
    
    def identify_sequential_chains(self) -> List[List[str]]:
        """Identify sequential chains in the DAG"""
        chains = []
        visited = set()
        
        # Find chain starts (tasks with no dependencies)
        starts = [t for t in self.tasks.values() if not t.dependencies]
        
        for start in starts:
            if start.id in visited:
                continue
                
            chain = []
            current = start
            
            while current and current.id not in visited:
                chain.append(current.id)
                visited.add(current.id)
                
                # Find single dependent
                dependents = self.task_dependencies.get(current.id, [])
                if len(dependents) == 1 and dependents[0] in self.tasks:
                    current = self.tasks[dependents[0]]
                else:
                    break
                    
            if len(chain) > 1:
                chains.append(chain)
                
        return chains
    
    def schedule(self) -> Dict[str, List[str]]:
        """Greedy scheduling with chain awareness"""
        schedule = defaultdict(list)
        
        # Process all tasks iteratively
        iterations = 0
        max_iterations = len(self.tasks) * 2
        
        while self.pending_tasks and iterations < max_iterations:
            iterations += 1
            
            ready_tasks = self.get_ready_tasks()
            if not ready_tasks:
                break
                
            scheduled_this_round = False
            
            # Sort by parameter overlap with existing nodes
            for task in ready_tasks:
                if task.id not in self.pending_tasks:
                    continue
                    
                # Find best node (with most params already cached)
                best_node = None
                min_params_to_load = float('inf')
                best_available_memory = 0
                
                for node in self.nodes.values():
                    if self.can_fit_on_node(task, node):
                        params_to_load = len(task.params_needed - node.cached_params)
                        
                        # Prefer nodes with params cached, then by available memory
                        if (params_to_load < min_params_to_load or 
                            (params_to_load == min_params_to_load and 
                             node.available_memory > best_available_memory)):
                            best_node = node
                            min_params_to_load = params_to_load
                            best_available_memory = node.available_memory
                            
                if best_node:
                    if self.assign_task_to_node(task, best_node):
                        schedule[best_node.id].append(task.id)
                        scheduled_this_round = True
                else:
                    self.fail_task(task.id)
                    
            if not scheduled_this_round:
                # No progress, fail remaining
                for task_id in list(self.pending_tasks):
                    self.fail_task(task_id)
                break
                
        return dict(schedule)


class CriticalPathScheduler(BaseScheduler):
    
    def compute_critical_path(self, task_id: str, memo: Dict[str, float]) -> float:
        if task_id in memo:
            return memo[task_id]
            
        task = self.tasks[task_id]
        
        # Find dependent tasks
        dependents = self.task_dependencies.get(task_id, [])
        if not dependents:
            path_length = task.compute_time
        else:
            valid_dependents = [d for d in dependents if d in self.tasks]
            if valid_dependents:
                path_length = task.compute_time + max(
                    self.compute_critical_path(dep, memo) for dep in valid_dependents
                )
            else:
                path_length = task.compute_time
                
        memo[task_id] = path_length
        return path_length
    
    def schedule(self) -> Dict[str, List[str]]:

        schedule = defaultdict(list)
        
        # Compute critical paths
        critical_paths = {}
        for task_id in self.tasks:
            self.compute_critical_path(task_id, critical_paths)
        
        iterations = 0
        max_iterations = len(self.tasks) * 2
        
        while self.pending_tasks and iterations < max_iterations:
            iterations += 1
            
            ready_tasks = self.get_ready_tasks()
            if not ready_tasks:
                break
                
            # Sort by critical path (longest first)
            ready_tasks.sort(key=lambda t: critical_paths.get(t.id, 0), reverse=True)
            
            scheduled_this_round = False
            
            for task in ready_tasks:
                if task.id not in self.pending_tasks:
                    continue
                    
                # Assign to fastest available node
                best_node = None
                best_speed = 0
                
                for node in self.nodes.values():
                    if self.can_fit_on_node(task, node) and node.compute_speed > best_speed:
                        best_node = node
                        best_speed = node.compute_speed
                        
                if best_node:
                    if self.assign_task_to_node(task, best_node):
                        schedule[best_node.id].append(task.id)
                        scheduled_this_round = True
                else:
                    self.fail_task(task.id)
                    
            if not scheduled_this_round:
                for task_id in list(self.pending_tasks):
                    self.fail_task(task_id)
                break
                
        return dict(schedule)


class MRUScheduler(BaseScheduler):
    
    def __init__(self, nodes: List[Node]):
        super().__init__(nodes)
        self.param_usage_count = defaultdict(int)
        self.param_last_used = {}
        self.time_step = 0
        
    def calculate_eviction_score(self, param: str, node: Node) -> float:
        """Calculate score for eviction (lower = evict first)"""
        score = 0.0
        
        # Frequency component
        score += self.param_usage_count[param] * 10
        
        # Recency component  
        if param in self.param_last_used:
            recency = self.time_step - self.param_last_used[param]
            score += 100.0 / (recency + 1)
            
        # Check if needed by any pending task with satisfied dependencies
        for task_id in self.pending_tasks:
            if self.is_task_ready(task_id):
                task = self.tasks[task_id]
                if param in task.params_needed:
                    score += 1000  # Very high score - likely needed soon
                    
        return score
    
    def evict_params_for_task(self, node: Node, task: Task) -> bool:
        """Try to evict params to make room for task"""
        memory_needed = self.calculate_memory_requirement(task, node)
        memory_shortage = memory_needed - node.available_memory
        
        if memory_shortage <= 0:
            return True
            
        # Get evictable params
        evictable = []
        for param in node.cached_params:
            if param not in task.params_needed:  # Don't evict params needed by this task
                score = self.calculate_eviction_score(param, node)
                evictable.append((score, param))
                
        # Sort by score (ascending - evict lowest scores first)
        evictable.sort()
        
        freed = 0
        evicted = []
        
        for score, param in evictable:
            if freed >= memory_shortage:
                break
            node.cached_params.remove(param)
            node.available_memory += 0.5
            self.param_locations[param].discard(node.id)
            freed += 0.5
            evicted.append(param)
            
        if freed >= memory_shortage:
            return True
        else:
            # Rollback evictions
            for param in evicted:
                node.cached_params.add(param)
                node.available_memory -= 0.5
                self.param_locations[param].add(node.id)
            return False
    
    def schedule(self) -> Dict[str, List[str]]:
        """MRU-aware scheduling"""
        schedule = defaultdict(list)
        
        iterations = 0
        max_iterations = len(self.tasks) * 2
        
        while self.pending_tasks and iterations < max_iterations:
            iterations += 1
            self.time_step += 1
            
            ready_tasks = self.get_ready_tasks()
            if not ready_tasks:
                break
                
            scheduled_this_round = False
            
            # Score tasks by urgency and parameter overlap
            task_scores = []
            for i, task in enumerate(ready_tasks):
                if task.id not in self.pending_tasks:
                    continue
                    
                # Count dependents waiting
                urgency = len([d for d in self.task_dependencies.get(task.id, [])
                             if d in self.pending_tasks])
                
                # Add index as tiebreaker to avoid comparing Task objects
                task_scores.append((urgency, i, task))
                
            # Sort by urgency (most dependents first), then by index
            task_scores.sort(key=lambda x: (x[0], -x[1]), reverse=True)
            
            for _, _, task in task_scores:
                best_node = None
                best_score = -float('inf')
                
                for node in self.nodes.values():
                    # Calculate score for this node
                    score = 0.0
                    
                    # Cached params bonus
                    cached = len(task.params_needed & node.cached_params)
                    score += cached * 20
                    
                    # Available memory bonus
                    if self.can_fit_on_node(task, node):
                        score += node.available_memory
                    elif self.evict_params_for_task(node, task):
                        score += 5  # Can fit with eviction
                    else:
                        continue  # Can't fit even with eviction
                        
                    # Load balance penalty
                    score -= len(node.completed_tasks) * 0.5
                    
                    if score > best_score:
                        best_score = score
                        best_node = node
                        
                if best_node:
                    # Perform eviction if needed
                    if not self.can_fit_on_node(task, best_node):
                        self.evict_params_for_task(best_node, task)
                        
                    if self.assign_task_to_node(task, best_node):
                        schedule[best_node.id].append(task.id)
                        scheduled_this_round = True
                        
                        # Update usage stats
                        for param in task.params_needed:
                            self.param_usage_count[param] += 1
                            self.param_last_used[param] = self.time_step
                else:
                    self.fail_task(task.id)
                    
            if not scheduled_this_round:
                for task_id in list(self.pending_tasks):
                    self.fail_task(task_id)
                break
                
        return dict(schedule)


# Test function
def test_schedulers():

    print("Testing Schedulers\n")
    
    # Create simple test DAG
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
    
    nodes = [
        Node("n1", total_memory=3.0),
        Node("n2", total_memory=2.5),
    ]
    
    schedulers = {
        "DFS": DFSScheduler,
        "Greedy": GreedyScheduler,
        "Critical Path": CriticalPathScheduler,
        "MRU_spec": MRUScheduler,
    }
    
    for name, SchedulerClass in schedulers.items():
        print(f"\n{name}:")
        scheduler = SchedulerClass([Node(n.id, n.total_memory) for n in nodes])
        
        for task in tasks:
            scheduler.add_task(copy.deepcopy(task))
            
        schedule = scheduler.schedule()
        
        print(f"  Completed: {len(scheduler.completed_tasks)}/{len(tasks)}")
        print(f"  Failed: {len(scheduler.failed_tasks)}")
        print(f"  Schedule: {dict(schedule)}")


if __name__ == "__main__":
    test_schedulers()