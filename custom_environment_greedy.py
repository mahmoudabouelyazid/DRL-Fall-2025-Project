import sys
import numpy as np
from collections import deque
from custom_environment_RL import WarehouseEnv, manhattan

class GreedyAgent:
    """Greedy agent that always chooses the closest available task."""
    
    def __init__(self, agent_id: str, env: WarehouseEnv):
        self.agent_id = agent_id
        self.env = env
        
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
            return 10  # already assigned, continue with current task
            
        # find the best unassigned task and assign it directly
        available_tasks = [tid for tid in self.env.backlog 
                          if not self.env.tasks[tid].done and self.env.tasks[tid].assigned_to is None]
        
        if not available_tasks:
            return 10  # no tasks available
            
        # greedy selection: choose closest task
        best_task = None
        best_distance = float('inf')
        
        for tid in available_tasks:
            tk = self.env.tasks[tid]
            distance = manhattan(ag.pos, tk.pickup)
            if distance < best_distance:
                best_distance = distance
                best_task = tid
                
        # directly assign the task (bypass action space)
        if best_task is not None:
            tk = self.env.tasks[best_task]
            if tk.assigned_to is None and not tk.done:
                tk.assigned_to = self.agent_id
                ag.path.clear()
                self.env._plan_append(ag, ag.pos, tk.pickup)
                
        return 10  # return noop since assignment is done directly

def run_greedy_episode(env: WarehouseEnv, max_steps: int = 600, render: bool = False, render_frequency: int = 10):
    """Run a single episode with greedy algorithm."""
    obs, info = env.reset()
    
    agents = [GreedyAgent(agent_id, env) for agent_id in env.agents]
    
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
        'delivery_priorities': [],  # track priority of each delivery
        'total_late_penalty': 0.0
    }
    
    # track which tasks were done before this episode to detect new deliveries
    previous_done_tasks = set()
    
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
    
    return episode_stats

def run_greedy_baseline(n_episodes: int = 10, debug: bool = False):
    """Run multiple episodes with greedy algorithm and collect statistics."""
    print("Running Greedy Algorithm Baseline...")
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
    
    all_stats = []
    
    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}...")
        
        stats = run_greedy_episode(env, max_steps=600)
        all_stats.append(stats)
        
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
    print("GREEDY ALGORITHM SUMMARY")
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
    
    
    if np.mean(deliveries) > 55:
        print(" Greedy algorithm performs well!")
    elif np.mean(deliveries) > 45:
        print(" Greedy algorithm performs moderately")
    else:
        print("Greedy algorithm underperforms")
        
    return all_stats

def run_comparison_visualization():
    """Run both RL and Greedy algorithms with visualization for comparison."""
    print("Running RL vs Greedy Algorithm Comparison with Visualization...")
    print("=" * 80)
    
    print("1. Running RL Algorithm...")
    try:
        from custom_environment_train import train
        print("RL algorithm training (first few updates)...")
        print("(Skipping RL training for now - would need trained model)")
    except ImportError:
        print("RL training module not available")
    
    print("\n2. Running Greedy Algorithm...")
    run_greedy_visualization()

def run_greedy_visualization():
    """Run greedy algorithm with full visualization."""
    print("Running Greedy Algorithm with Visualization...")
    print("=" * 80)
    
    env = WarehouseEnv(
        n_agents=3,
        n_initial_tasks=20,
        task_spawn_rate=0.05,
        horizon_steps=600,
        K=10,
        render_mode="human",
        seed=7,
        debug=False,
    )
    
    
    stats = run_greedy_episode(env, max_steps=600, render=True, render_frequency=5)
    
    print("\n" + "=" * 80)
    print("GREEDY ALGORITHM VISUALIZATION RESULTS")
    print("=" * 80)
    
    deliveries = stats['deliveries']
    idle_percentage = stats['idle_percentage']
    total_idle_steps = stats['total_idle_steps']
    idle_per_agent = stats['idle_time_per_agent']
    
    idle_per_agent_str = ", ".join([f"{agent}: {time}" for agent, time in idle_per_agent.items()])
    
    print(f"Deliveries: {deliveries} | "
          f"Deliveries/min: {deliveries/(600/60):.1f} | "
          f"Idle: {idle_percentage:.1f}% ({total_idle_steps} steps) | "
          f"Per-agent: [{idle_per_agent_str}]")
    
    print(f"\nEpisode completed successfully!")
    print(f"Greedy algorithm achieved {deliveries} deliveries in {stats['episode_steps']} steps.")
    
    env.close()

def run_greedy_with_same_logging():
    """Run greedy algorithm with the same logging format as RL training."""
    print("Starting greedy algorithm with RL-style logging...")
    
    env = WarehouseEnv(
        n_agents=3,
        n_initial_tasks=15,
        task_spawn_rate=0.05,
        horizon_steps=600,
        K=10,
        render_mode=None,
        seed=7,
        debug=False,
    )
    
    n_updates = 10
    episodes_per_update = 4
    
    for update in range(1, n_updates + 1):
        update_deliveries = []
        update_idle_percentages = []
        update_total_idle_steps = []
        update_idle_per_agent = {agent_id: [] for agent_id in env.agents}
        
        for ep in range(episodes_per_update):
            stats = run_greedy_episode(env, max_steps=600)
            update_deliveries.append(stats['deliveries'])
            update_idle_percentages.append(stats['idle_percentage'])
            update_total_idle_steps.append(stats['total_idle_steps'])
            
            for agent_id in env.agents:
                update_idle_per_agent[agent_id].append(stats['idle_time_per_agent'][agent_id])
        
        # averages for this update
        avg_deliveries = np.mean(update_deliveries)
        avg_idle_percentage = np.mean(update_idle_percentages)
        avg_total_idle_steps = int(np.mean(update_total_idle_steps))
        
        # idle time per agent (average across episodes)
        idle_per_agent_str = ", ".join([
            f"{agent_id}: {int(np.mean(update_idle_per_agent[agent_id]))}" 
            for agent_id in env.agents
        ])
        
        # curriculum learning (same as RL)
        curriculum_step = min(15, update // 5)
        n_tasks = min(40, 15 + curriculum_step * 3)
        
        print(f"[Update {update:03d}] Steps {update * episodes_per_update * 600:7d} | "
              f"EpRet mean N/A | EpLen mean 600 | "
              f"Deliveries {avg_deliveries:.0f} | "
              f"Deliveries/min {avg_deliveries/(600/60):.1f} | "
              f"Tasks: {n_tasks} | "
              f"Idle: {avg_idle_percentage:.1f}% ({avg_total_idle_steps} steps) | "
              f"Per-agent: [{idle_per_agent_str}]")
        
        if update % 5 == 0:
            print(f"Update {update}/{n_updates} done.")
    
    env.close()
    print("Greedy algorithm training simulation completed!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--detailed":
            run_greedy_baseline(n_episodes=20, debug=False)
        elif sys.argv[1] == "--visualize":
            run_greedy_visualization()
        elif sys.argv[1] == "--compare":
            run_comparison_visualization()
        else:
            print("Usage: python custom_environment_greedy.py [--detailed|--visualize|--compare]")
            print("  --detailed: Run detailed baseline comparison")
            print("  --visualize: Run greedy algorithm with visualization")
            print("  --compare: Run RL vs Greedy comparison with visualization")
            print("  (no args): Run with RL-style logging")
    else:
        run_greedy_with_same_logging()
