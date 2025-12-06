import sys
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from custom_environment_RL import WarehouseEnv, manhattan


class ACOAgent:
    
    def __init__(self, agent_id: str, env: WarehouseEnv, pheromone_matrix: np.ndarray):
        self.agent_id = agent_id
        self.env = env
        self.pheromone_matrix = pheromone_matrix  # shared pheromone matrix
        self.current_task = None
        
    def get_action(self):
        ag = self.env.agent_state[self.agent_id]
        
        # if carrying, continue to drop-off (noop to follow path)
        if ag.carrying is not None:
            return 10 
            
        # if has path, continue following it (noop to follow path)
        if len(ag.path) > 0:
            return 10  
        
        # check if already assigned to a task
        has_assignment = any(self.env.tasks[tid].assigned_to == self.agent_id 
                           for tid in self.env.backlog 
                           if not self.env.tasks[tid].done)
        if has_assignment:
            return 10  # Already assigned, continue with current task
            
        # use ACO-assigned task if available
        if self.current_task is not None:
            tk = self.env.tasks.get(self.current_task)
            if tk and tk.assigned_to is None and not tk.done:
                # directly assign the task (bypass action space)
                tk.assigned_to = self.agent_id
                ag.path.clear()
                self.env._plan_append(ag, ag.pos, tk.pickup)
                self.current_task = None  # clear after assignment
            else:
                self.current_task = None
                
        return 10  # return noop since assignment is done directly


def solve_task_assignment_aco(
    env: WarehouseEnv, 
    pheromone_matrix: np.ndarray,
    alpha: float = 1.0,  # pheromone importance
    beta: float = 2.0,   # heuristic importance
    evaporation: float = 0.1
) -> Dict[str, int]:

    available_tasks = [tid for tid in env.backlog 
                      if not env.tasks[tid].done and env.tasks[tid].assigned_to is None]
    available_agents = [aid for aid in env.agents 
                       if env.agent_state[aid].carrying is None]
    
    if not available_tasks or not available_agents:
        return {}
    
    # create task index mapping
    task_to_idx = {tid: idx for idx, tid in enumerate(available_tasks)}
    agent_to_idx = {aid: idx for idx, aid in enumerate(available_agents)}
    
    assignment = {}
    
    # for each available agent, probabilistically select a task
    for agent_id in available_agents:
        ag = env.agent_state[agent_id]
        agent_idx = agent_to_idx[agent_id]
        
        # calculate probabilities for each available task
        probabilities = []
        task_ids = []
        
        for task_id in available_tasks:
            # skip if already assigned
            if task_id in assignment.values():
                continue
                
            tk = env.tasks[task_id]
            task_idx = task_to_idx[task_id]
            
            # get pheromone value (use agent_idx and task_idx, with bounds checking)
            if agent_idx < pheromone_matrix.shape[0] and task_idx < pheromone_matrix.shape[1]:
                pheromone = pheromone_matrix[agent_idx, task_idx]
            else:
                pheromone = 0.1  # Default small value
            
            # heuristic: inverse of distance (closer tasks are more attractive)
            distance = manhattan(ag.pos, tk.pickup)
            heuristic = 1.0 / (1.0 + distance)  # Add 1 to avoid division by zero
            
            # calculate probability using ACO formula
            prob = (pheromone ** alpha) * (heuristic ** beta)
            
            probabilities.append(prob)
            task_ids.append(task_id)
        
        if not probabilities:
            continue
        
        # normalize probabilities
        prob_sum = sum(probabilities)
        if prob_sum > 0:
            probabilities = [p / prob_sum for p in probabilities]
        else:
            # if all probabilities are zero, use uniform distribution
            probabilities = [1.0 / len(probabilities)] * len(probabilities)
        
        # select task probabilistically
        selected_task = np.random.choice(task_ids, p=probabilities)
        assignment[agent_id] = selected_task
    
    return assignment


def update_pheromone_matrix(
    pheromone_matrix: np.ndarray,
    assignments: List[Dict[str, int]],
    env: WarehouseEnv,
    quality_scores: List[float],
    evaporation: float = 0.1,
    Q: float = 100.0  # pheromone deposit constant
):
    # evaporate all pheromones
    pheromone_matrix *= (1.0 - evaporation)
    
    # deposit pheromones based on solution quality
    if not assignments or not quality_scores:
        return
    
    # create agent and task index mappings
    agent_to_idx = {aid: idx for idx, aid in enumerate(env.agents)}
    
    for assignment, quality in zip(assignments, quality_scores):
        if quality <= 0:
            continue
            
        # deposit pheromone proportional to solution quality
        deposit = Q * quality / max(quality_scores)  # normalize by best quality
        
        for agent_id, task_id in assignment.items():
            if agent_id in agent_to_idx:
                agent_idx = agent_to_idx[agent_id]
                task_idx = task_id % pheromone_matrix.shape[1]
                if agent_idx < pheromone_matrix.shape[0]:
                    pheromone_matrix[agent_idx, task_idx] += deposit


def run_aco_episode(
    env: WarehouseEnv, 
    max_steps: int = 600, 
    render: bool = False, 
    render_frequency: int = 10,
    pheromone_matrix: Optional[np.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.1
):
    obs, info = env.reset()
    
    if pheromone_matrix is None:
        max_tasks = 100  # estimate maximum number of tasks
        pheromone_matrix = np.ones((len(env.agents), max_tasks)) * 0.1
    
    agents = [ACOAgent(agent_id, env, pheromone_matrix) for agent_id in env.agents]
    
    episode_stats = {
        'total_reward': 0,
        'steps': 0,
        'deliveries': 0,
        'idle_time_per_agent': {agent_id: 0 for agent_id in env.agents},
        'total_idle_steps': 0,
        'episode_steps': 0,
        'total_agent_steps': 0,
        'idle_percentage': 0,
        # Late task metrics
        'late_deliveries': 0,
        'delivery_priorities': [], 
        'total_late_penalty': 0.0
    }
    
    # track assignments for pheromone update
    episode_assignments = []
    
    # track which tasks were done before this episode to detect new deliveries
    previous_done_tasks = set()
    
    # initial assignment
    assignment = solve_task_assignment_aco(env, pheromone_matrix, alpha, beta, evaporation)
    episode_assignments.append(assignment.copy())
    for agent_id, task_id in assignment.items():
        for agent in agents:
            if agent.agent_id == agent_id:
                agent.current_task = task_id
                break
    
    for step in range(max_steps):
        if render and step % render_frequency == 0:
            env.render()
            print(f"Step {step}: Deliveries={info.get('__env__', {}).get('deliveries_cum', 0)}")
        
        actions = {}
        for i, agent_id in enumerate(env.agents):
            actions[agent_id] = agents[i].get_action()
            
        obs, rewards, terms, truncs, info = env.step(actions)
        
        # track late deliveries: check which tasks became done this step
        for tid, tk in env.tasks.items():
            if tk.done and tid not in previous_done_tasks:
                # task was just delivered
                episode_stats['delivery_priorities'].append(tk.priority)
                if tk.priority >= 100.0:
                    episode_stats['late_deliveries'] += 1
                    # calculate late penalty (same formula as in environment)
                    late_penalty = (tk.priority - 100.0) * 0.1
                    episode_stats['total_late_penalty'] += late_penalty
                previous_done_tasks.add(tid)
        
        episode_stats['total_reward'] += sum(rewards.values())
        episode_stats['steps'] += 1
        
        # reassign tasks periodically using ACO
        if step % 10 == 0:
            assignment = solve_task_assignment_aco(env, pheromone_matrix, alpha, beta, evaporation)
            if assignment:
                episode_assignments.append(assignment.copy())
            for agent_id, task_id in assignment.items():
                for agent in agents:
                    if agent.agent_id == agent_id:
                        agent.current_task = task_id
                        break
        
        if all(terms.values()):
            break
    
    if render:
        env.render()
        print(f"Episode completed! Final deliveries={info.get('__env__', {}).get('deliveries_cum', 0)}")
            
    env_info = info.get('__env__', {})
    episode_stats['deliveries'] = env_info.get('deliveries_cum', 0)
    episode_stats['idle_time_per_agent'] = env_info.get('idle_time_per_agent', {})
    episode_stats['total_idle_steps'] = env_info.get('total_idle_steps', 0)
    episode_stats['episode_steps'] = env_info.get('episode_steps', 0)
    episode_stats['total_agent_steps'] = env_info.get('total_agent_steps', 0)
    episode_stats['idle_percentage'] = env_info.get('idle_percentage', 0)
    
    # return pheromone matrix for next episode
    return episode_stats, pheromone_matrix, episode_assignments


def run_aco_baseline(n_episodes: int = 10, debug: bool = False):
    print("Running ACO Algorithm Baseline...")
    print("=" * 80)
    
    env = WarehouseEnv(
        n_agents=3,
        n_initial_tasks=20,
        task_spawn_rate=0.05,
        horizon_steps=600,
        K=10,
        render_mode=None,
        seed=7,
        debug=debug,
    )
    
    # initialize pheromone matrix
    max_tasks = 100
    pheromone_matrix = np.ones((len(env.agents), max_tasks)) * 0.1
    
    # ACO parameters
    alpha = 1.0  # pheromone importance
    beta = 2.0   # huristic importance
    evaporation = 0.1
    
    all_stats = []
    recent_assignments = []  # store last N assignments for pheromone update
    recent_qualities = []    # store quality scores
    
    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}...")
        
        stats, pheromone_matrix, episode_assignments = run_aco_episode(
            env, max_steps=600, render=False,
            pheromone_matrix=pheromone_matrix,
            alpha=alpha, beta=beta, evaporation=evaporation
        )
        all_stats.append(stats)
        
        quality = stats['deliveries']
        recent_assignments.extend(episode_assignments)
        recent_qualities.extend([quality] * len(episode_assignments))
        
        # keep only last 5 episodes of assignments
        if len(recent_assignments) > 5:
            recent_assignments = recent_assignments[-5:]
            recent_qualities = recent_qualities[-5:]
        
        # update pheromones periodically
        if (episode + 1) % 5 == 0:
            update_pheromone_matrix(
                pheromone_matrix, recent_assignments, env, recent_qualities,
                evaporation=evaporation
            )
            recent_assignments = []
            recent_qualities = []
        
        deliveries = stats['deliveries']
        idle_percentage = stats['idle_percentage']
        total_idle_steps = stats['total_idle_steps']
        idle_per_agent = stats['idle_time_per_agent']
        
        idle_per_agent_str = ", ".join([f"{agent}: {time}" for agent, time in idle_per_agent.items()])
        
        print(f"  Deliveries: {deliveries} | "
              f"Deliveries/min: {deliveries/(600/60):.1f} | "
              f"Idle: {idle_percentage:.1f}% ({total_idle_steps} steps) | "
              f"Per-agent: [{idle_per_agent_str}]")
    
    env.close()
    
    print("\n" + "=" * 80)
    print("ACO ALGORITHM SUMMARY")
    print("=" * 80)
    
    deliveries = [s['deliveries'] for s in all_stats]
    idle_percentages = [s['idle_percentage'] for s in all_stats]
    total_rewards = [s['total_reward'] for s in all_stats]
    
    print(f"Episodes: {n_episodes}")
    print(f"Average Deliveries: {np.mean(deliveries):.1f} ± {np.std(deliveries):.1f}")
    print(f"Average Deliveries/min: {np.mean(deliveries)/(600/60):.1f}")
    print(f"Average Idle Time: {np.mean(idle_percentages):.1f}% ± {np.std(idle_percentages):.1f}%")
    print(f"Average Total Reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    print(f"Best Episode Deliveries: {max(deliveries)}")
    print(f"Worst Episode Deliveries: {min(deliveries)}")
    
    if np.mean(deliveries) > 58:
        print("ACO algorithm performs excellently!")
    elif np.mean(deliveries) > 55:
        print("ACO algorithm performs very well!")
    elif np.mean(deliveries) > 50:
        print("ACO algorithm performs moderately")
    else:
        print("ACO algorithm underperforms")
        
    return all_stats


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--detailed":
            run_aco_baseline(n_episodes=20, debug=False)
        else:
            print("Usage: python custom_environment_aco.py [--detailed]")
            print("  --detailed: Run detailed baseline comparison")
            print("  (no args): Run quick test")
    else:
        print("Running quick ACO test...")
        run_aco_baseline(n_episodes=5, debug=False)


