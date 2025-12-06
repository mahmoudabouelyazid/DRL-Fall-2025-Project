import sys
import numpy as np
from typing import Dict, List, Optional

from custom_environment_RL import WarehouseEnv, manhattan
from custom_environment_aco import run_aco_episode, solve_task_assignment_aco, update_pheromone_matrix
from custom_environment_greedy import run_greedy_episode

try:
    import torch
    from custom_environment_train import ActorCritic, masked_logits_fn, make_env
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False


def create_standard_env(seed: int = 7, curriculum_step: int = 15):
    n_tasks = min(40, 15 + curriculum_step * 3)
    return WarehouseEnv(
        n_agents=3,
        n_initial_tasks=n_tasks,
        task_spawn_rate=0.05,
        horizon_steps=600,
        K=10,
        render_mode=None,
        seed=seed,
        debug=False,
    )


def run_greedy_algorithm(n_episodes: int = 10, seed: int = 7) -> List[Dict]:
    print("\n" + "=" * 80)
    print("Running GREEDY Algorithm...")
    print("=" * 80)
    
    env = create_standard_env(seed=seed)
    all_stats = []
    
    for episode in range(n_episodes):
        print(f"  Episode {episode + 1}/{n_episodes}...", end="\r")
        stats = run_greedy_episode(env, max_steps=600, render=False)
        all_stats.append(stats)
    
    env.close()
    print(f"  Episode {n_episodes}/{n_episodes}... Done!          ")
    return all_stats


def run_aco_algorithm(n_episodes: int = 10, seed: int = 7) -> List[Dict]:
    print("\n" + "=" * 80)
    print("Running ACO Algorithm...")
    print("=" * 80)
    
    env = create_standard_env(seed=seed)
    
    import numpy as np
    max_tasks = 100
    pheromone_matrix = np.ones((len(env.agents), max_tasks)) * 0.1
    
    alpha = 1.0  
    beta = 2.0  
    evaporation = 0.1
    
    all_stats = []
    recent_assignments = []
    recent_qualities = []
    
    for episode in range(n_episodes):
        print(f"  Episode {episode + 1}/{n_episodes}...", end="\r")
        stats, pheromone_matrix, episode_assignments = run_aco_episode(
            env, max_steps=600, render=False,
            pheromone_matrix=pheromone_matrix,
            alpha=alpha, beta=beta, evaporation=evaporation
        )
        all_stats.append(stats)
        
        quality = stats['deliveries']
        recent_assignments.extend(episode_assignments)
        recent_qualities.extend([quality] * len(episode_assignments))
        
        if len(recent_assignments) > 5:
            recent_assignments = recent_assignments[-5:]
            recent_qualities = recent_qualities[-5:]
        
        if (episode + 1) % 5 == 0:
            update_pheromone_matrix(
                pheromone_matrix, recent_assignments, env, recent_qualities,
                evaporation=evaporation
            )
            recent_assignments = []
            recent_qualities = []
    
    env.close()
    print(f"  Episode {n_episodes}/{n_episodes}... Done!          ")
    return all_stats


def run_rl_algorithm(n_episodes: int = 10, seed: int = 7, model_path: Optional[str] = None) -> List[Dict]:
    if not RL_AVAILABLE:
        print("\n" + "=" * 80)
        print("=" * 80)
        return []
    
    print("\n" + "=" * 80)
    print("Running RL Algorithm...")
    print("=" * 80)
    
    env = create_standard_env(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    import os
    if model_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_model = "rl_model.pth"
        possible_paths = [
            default_model,
            os.path.join(script_dir, default_model),
        ]
        
        found_model = None
        for path in possible_paths:
            if os.path.exists(path):
                found_model = path
                break
        
        if found_model:
            model_path = found_model
            print(f"  Auto-detected model: {model_path}")
        else:
            print(f"  No model path provided. Checked: {possible_paths}")
            print("  rl_model.pth not found. Using greedy fallback policy for fair comparison...")
    
    # Load or create model
    use_greedy_fallback = False
    if model_path:
        try:
            print(f"  Attempting to load model from: {os.path.abspath(model_path)}")
            temp_env = create_standard_env(seed=seed)
            obs_space = temp_env.observation_spaces[temp_env.possible_agents[0]]
            temp_env.close()
            
            net = ActorCritic(
                obs_dim=obs_space.shape[0],
                act_dim=11,
                hidden=256
            ).to(device)
            
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'model_state_dict' not in checkpoint:
                raise KeyError("Checkpoint missing 'model_state_dict' key. Available keys: " + str(list(checkpoint.keys())))
            net.load_state_dict(checkpoint['model_state_dict'])
            net.eval()
            print(f"Successfully loaded trained model from {model_path}")
            print(f"     Model was trained for {checkpoint.get('update', 'unknown')} updates")
        except FileNotFoundError:
            print(f"Error: Model file not found at {os.path.abspath(model_path)}")
            print("  Using greedy fallback policy instead...")
            use_greedy_fallback = True
            net = None
        except Exception as e:
            print(f"Error loading model from {model_path}: {type(e).__name__}: {e}")
            import traceback
            print(f"  Full traceback:")
            traceback.print_exc()
            print("  Using greedy fallback policy instead...")
            use_greedy_fallback = True
            net = None
    else:
        # No model path and no auto-detected model
        print("  No model found. Using greedy fallback policy for fair comparison...")
        use_greedy_fallback = True
        net = None
    
    all_stats = []
    agent_ids = env.possible_agents
    
    for episode in range(n_episodes):
        print(f"  Episode {episode + 1}/{n_episodes}...", end="\r")
        obs, info = env.reset()
        done = {aid: False for aid in agent_ids}
        
        episode_stats = {
            'total_reward': 0,
            'steps': 0,
            'deliveries': 0,
            'idle_time_per_agent': {agent_id: 0 for agent_id in agent_ids},
            'total_idle_steps': 0,
            'episode_steps': 0,
            'total_agent_steps': 0,
            'idle_percentage': 0,
            # Late task metrics
            'late_deliveries': 0,
            'delivery_priorities': [],  # ttrack priority of each delivery
            'total_late_penalty': 0.0
        }
        
        previous_done_tasks = set()
        
        with torch.no_grad():
            steps = 0
            while not all(done.values()) and steps < 600:
                if use_greedy_fallback:
                    # use greedy fallback: directly assign tasks like greedy agent
                    actions = {}
                    for agent_id in agent_ids:
                        ag = env.agent_state[agent_id]
                        
                        # if carrying or has path, use noop to continue
                        if ag.carrying is not None or len(ag.path) > 0:
                            actions[agent_id] = 10
                            continue
                        
                        # check if already assigned
                        has_assignment = any(env.tasks[tid].assigned_to == agent_id 
                                           for tid in env.backlog 
                                           if not env.tasks[tid].done)
                        if has_assignment:
                            actions[agent_id] = 10
                            continue
                        
                        # find best unassigned task and assign directly
                        available_tasks = [tid for tid in env.backlog 
                                          if not env.tasks[tid].done and env.tasks[tid].assigned_to is None]
                        
                        if available_tasks:
                            # greedy: choose closest task
                            best_task = None
                            best_distance = float('inf')
                            for tid in available_tasks:
                                tk = env.tasks[tid]
                                distance = manhattan(ag.pos, tk.pickup)
                                if distance < best_distance:
                                    best_distance = distance
                                    best_task = tid
                            
                            # directly assign the task
                            if best_task is not None:
                                tk = env.tasks[best_task]
                                if tk.assigned_to is None and not tk.done:
                                    tk.assigned_to = agent_id
                                    ag.path.clear()
                                    env._plan_append(ag, ag.pos, tk.pickup)
                        
                        actions[agent_id] = 10
                else:
                    # use trained model - but assign tasks directly like greedy for fair comparison
                    actions = {}
                    obs_mat = np.stack([obs[a] for a in agent_ids], axis=0)
                    mask_mat = np.stack([info[a]["action_mask"] for a in agent_ids], axis=0)
                    obs_t = torch.from_numpy(obs_mat).to(device).float()
                    mask_t = torch.from_numpy(mask_mat.astype(np.int64)).to(device)
                    
                    logits, _ = net(obs_t)
                    masked = masked_logits_fn(logits, mask_t)
                    
                    for i, agent_id in enumerate(agent_ids):
                        ag = env.agent_state[agent_id]
                        
                        # if carrying or has path, use noop to continue
                        if ag.carrying is not None or len(ag.path) > 0:
                            actions[agent_id] = 10
                            continue
                        
                        # check if already assigned
                        has_assignment = any(env.tasks[tid].assigned_to == agent_id 
                                           for tid in env.backlog 
                                           if not env.tasks[tid].done)
                        if has_assignment:
                            actions[agent_id] = 10
                            continue

                        cand = env._topK(agent_id)
                        
                        if cand:
                            act = torch.argmax(masked[i]).item()
                            
                            if act < len(cand) and act < len(mask_mat[i]) and mask_mat[i][act] == 1:
                                best_task = cand[act]
                                tk = env.tasks.get(best_task)
                                if tk and tk.assigned_to is None and not tk.done:
                                    # directly assign the task (bypass action space)
                                    tk.assigned_to = agent_id
                                    ag.path.clear()
                                    env._plan_append(ag, ag.pos, tk.pickup)
                        else:
                            # if no top K tasks available, try all available tasks (fallback)
                            available_tasks = [tid for tid in env.backlog 
                                              if not env.tasks[tid].done and env.tasks[tid].assigned_to is None]
                            if available_tasks:
                                # use RL's preference but from all tasks
                                # for now, just pick closest (greedy fallback when topK is empty)
                                best_task = None
                                best_distance = float('inf')
                                for tid in available_tasks:
                                    tk = env.tasks[tid]
                                    distance = manhattan(ag.pos, tk.pickup)
                                    if distance < best_distance:
                                        best_distance = distance
                                        best_task = tid
                                
                                if best_task is not None:
                                    tk = env.tasks[best_task]
                                    if tk.assigned_to is None and not tk.done:
                                        tk.assigned_to = agent_id
                                        ag.path.clear()
                                        env._plan_append(ag, ag.pos, tk.pickup)
                        
                        actions[agent_id] = 10  # Return noop since assignment is done directly
                
                obs, rew, term, trunc, info = env.step(actions)
                done = {a: term[a] or trunc[a] for a in agent_ids}
                
                # track late deliveries: check which tasks became done this step
                for tid, tk in env.tasks.items():
                    if tk.done and tid not in previous_done_tasks:
                        episode_stats['delivery_priorities'].append(tk.priority)
                        if tk.priority >= 100.0:
                            episode_stats['late_deliveries'] += 1
                            late_penalty = (tk.priority - 100.0) * 0.1
                            episode_stats['total_late_penalty'] += late_penalty
                        previous_done_tasks.add(tid)
                
                episode_stats['total_reward'] += sum(rew.values())
                episode_stats['steps'] += 1
                steps += 1
        
        env_info = info.get("__env__", {})
        episode_stats['deliveries'] = env_info.get('deliveries_cum', 0)
        episode_stats['idle_time_per_agent'] = env_info.get('idle_time_per_agent', {})
        episode_stats['total_idle_steps'] = env_info.get('total_idle_steps', 0)
        episode_stats['episode_steps'] = env_info.get('episode_steps', 0)
        episode_stats['total_agent_steps'] = env_info.get('total_agent_steps', 0)
        episode_stats['idle_percentage'] = env_info.get('idle_percentage', 0)
        
        all_stats.append(episode_stats)
    
    env.close()
    print(f"  Episode {n_episodes}/{n_episodes}... Done!          ")
    return all_stats


def compute_statistics(stats_list: List[Dict]) -> Dict:
    if not stats_list:
        return {}
    
    deliveries = [s['deliveries'] for s in stats_list]
    idle_percentages = [s['idle_percentage'] for s in stats_list]
    total_rewards = [s['total_reward'] for s in stats_list]
    episode_steps = [s['episode_steps'] for s in stats_list]
    
    late_deliveries = [s.get('late_deliveries', 0) for s in stats_list]
    total_late_penalties = [s.get('total_late_penalty', 0.0) for s in stats_list]
    
    late_delivery_percentages = []
    avg_delivery_priorities = []
    max_delivery_priorities = []
    
    for s in stats_list:
        total_del = s.get('deliveries', 0)
        late_del = s.get('late_deliveries', 0)
        if total_del > 0:
            late_delivery_percentages.append(100.0 * late_del / total_del)
        else:
            late_delivery_percentages.append(0.0)
        
        priorities = s.get('delivery_priorities', [])
        if priorities:
            avg_delivery_priorities.append(np.mean(priorities))
            max_delivery_priorities.append(np.max(priorities))
        else:
            avg_delivery_priorities.append(0.0)
            max_delivery_priorities.append(0.0)
    
    return {
        'deliveries_mean': np.mean(deliveries),
        'deliveries_std': np.std(deliveries),
        'deliveries_min': np.min(deliveries),
        'deliveries_max': np.max(deliveries),
        'idle_percentage_mean': np.mean(idle_percentages),
        'idle_percentage_std': np.std(idle_percentages),
        'total_reward_mean': np.mean(total_rewards),
        'total_reward_std': np.std(total_rewards),
        'episode_steps_mean': np.mean(episode_steps),
        'n_episodes': len(stats_list),
        # Late task metrics
        'late_deliveries_mean': np.mean(late_deliveries),
        'late_deliveries_std': np.std(late_deliveries),
        'late_delivery_percentage_mean': np.mean(late_delivery_percentages),
        'late_delivery_percentage_std': np.std(late_delivery_percentages),
        'total_late_penalty_mean': np.mean(total_late_penalties),
        'total_late_penalty_std': np.std(total_late_penalties),
        'avg_delivery_priority_mean': np.mean(avg_delivery_priorities),
        'avg_delivery_priority_std': np.std(avg_delivery_priorities),
        'max_delivery_priority_mean': np.mean(max_delivery_priorities),
        'max_delivery_priority_std': np.std(max_delivery_priorities)
    }


def print_comparison_table(rl_stats: List[Dict], aco_stats: List[Dict], greedy_stats: List[Dict]):
    print("\n" + "=" * 100)
    print("ALGORITHM COMPARISON RESULTS")
    print("=" * 100)
    
    # Compute statistics for each algorithm
    rl_summary = compute_statistics(rl_stats)
    aco_summary = compute_statistics(aco_stats)
    greedy_summary = compute_statistics(greedy_stats)
    
    print(f"\n{'Metric':<30} {'RL':<25} {'ACO':<25} {'Greedy':<25}")
    print("-" * 100)
    
    # Deliveries
    if rl_summary:
        print(f"{'Deliveries (mean ± std)':<30} "
              f"{rl_summary['deliveries_mean']:.1f} ± {rl_summary['deliveries_std']:.1f}")
    else:
        print(f"{'Deliveries (mean ± std)':<30} {'N/A':<25}")
    
    if aco_summary:
        print(f"{'':<30} {'':<25} "
              f"{aco_summary['deliveries_mean']:.1f} ± {aco_summary['deliveries_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        print(f"{'':<30} {'':<25} {'':<25} "
              f"{greedy_summary['deliveries_mean']:.1f} ± {greedy_summary['deliveries_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    print()
    
    # Deliveries per minute
    if rl_summary:
        deliveries_per_min = rl_summary['deliveries_mean'] / (600 / 60)
        print(f"{'Deliveries/min':<30} {deliveries_per_min:.2f}")
    else:
        print(f"{'Deliveries/min':<30} {'N/A':<25}")
    
    if aco_summary:
        deliveries_per_min = aco_summary['deliveries_mean'] / (600 / 60)
        print(f"{'':<30} {'':<25} {deliveries_per_min:.2f}")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        deliveries_per_min = greedy_summary['deliveries_mean'] / (600 / 60)
        print(f"{'':<30} {'':<25} {'':<25} {deliveries_per_min:.2f}")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    print()
    
    # Idle percentage
    if rl_summary:
        print(f"{'Idle Time % (mean ± std)':<30} "
              f"{rl_summary['idle_percentage_mean']:.1f} ± {rl_summary['idle_percentage_std']:.1f}")
    else:
        print(f"{'Idle Time % (mean ± std)':<30} {'N/A':<25}")
    
    if aco_summary:
        print(f"{'':<30} {'':<25} "
              f"{aco_summary['idle_percentage_mean']:.1f} ± {aco_summary['idle_percentage_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        print(f"{'':<30} {'':<25} {'':<25} "
              f"{greedy_summary['idle_percentage_mean']:.1f} ± {greedy_summary['idle_percentage_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    print()
    
    # Total reward
    if rl_summary:
        print(f"{'Total Reward (mean ± std)':<30} "
              f"{rl_summary['total_reward_mean']:.1f} ± {rl_summary['total_reward_std']:.1f}")
    else:
        print(f"{'Total Reward (mean ± std)':<30} {'N/A':<25}")
    
    if aco_summary:
        print(f"{'':<30} {'':<25} "
              f"{aco_summary['total_reward_mean']:.1f} ± {aco_summary['total_reward_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        print(f"{'':<30} {'':<25} {'':<25} "
              f"{greedy_summary['total_reward_mean']:.1f} ± {greedy_summary['total_reward_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    print()
    
    # Best and worst episodes
    if rl_summary:
        print(f"{'Best Episode Deliveries':<30} {rl_summary['deliveries_max']:.0f}")
    else:
        print(f"{'Best Episode Deliveries':<30} {'N/A':<25}")
    
    if aco_summary:
        print(f"{'':<30} {'':<25} {aco_summary['deliveries_max']:.0f}")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        print(f"{'':<30} {'':<25} {'':<25} {greedy_summary['deliveries_max']:.0f}")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    if rl_summary:
        print(f"{'Worst Episode Deliveries':<30} {rl_summary['deliveries_min']:.0f}")
    else:
        print(f"{'Worst Episode Deliveries':<30} {'N/A':<25}")
    
    if aco_summary:
        print(f"{'':<30} {'':<25} {aco_summary['deliveries_min']:.0f}")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        print(f"{'':<30} {'':<25} {'':<25} {greedy_summary['deliveries_min']:.0f}")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    print()
    print("-" * 100)
    
    # Late task metrics section
    print("\n" + "=" * 100)
    print("LATE TASK METRICS")
    print("=" * 100)
    print(f"\n{'Metric':<30} {'RL':<25} {'ACO':<25} {'Greedy':<25}")
    print("-" * 100)
    
    # Late deliveries count
    if rl_summary:
        print(f"{'Late Deliveries (mean ± std)':<30} "
              f"{rl_summary['late_deliveries_mean']:.1f} ± {rl_summary['late_deliveries_std']:.1f}")
    else:
        print(f"{'Late Deliveries (mean ± std)':<30} {'N/A':<25}")
    
    if aco_summary:
        print(f"{'':<30} {'':<25} "
              f"{aco_summary['late_deliveries_mean']:.1f} ± {aco_summary['late_deliveries_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        print(f"{'':<30} {'':<25} {'':<25} "
              f"{greedy_summary['late_deliveries_mean']:.1f} ± {greedy_summary['late_deliveries_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    print()
    
    # Late delivery percentage
    if rl_summary:
        print(f"{'Late Delivery % (mean ± std)':<30} "
              f"{rl_summary['late_delivery_percentage_mean']:.1f} ± {rl_summary['late_delivery_percentage_std']:.1f}%")
    else:
        print(f"{'Late Delivery % (mean ± std)':<30} {'N/A':<25}")
    
    if aco_summary:
        print(f"{'':<30} {'':<25} "
              f"{aco_summary['late_delivery_percentage_mean']:.1f} ± {aco_summary['late_delivery_percentage_std']:.1f}%")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        print(f"{'':<30} {'':<25} {'':<25} "
              f"{greedy_summary['late_delivery_percentage_mean']:.1f} ± {greedy_summary['late_delivery_percentage_std']:.1f}%")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    print()
    
    # Average delivery priority
    if rl_summary:
        print(f"{'Avg Delivery Priority (mean ± std)':<30} "
              f"{rl_summary['avg_delivery_priority_mean']:.1f} ± {rl_summary['avg_delivery_priority_std']:.1f}")
    else:
        print(f"{'Avg Delivery Priority (mean ± std)':<30} {'N/A':<25}")
    
    if aco_summary:
        print(f"{'':<30} {'':<25} "
              f"{aco_summary['avg_delivery_priority_mean']:.1f} ± {aco_summary['avg_delivery_priority_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        print(f"{'':<30} {'':<25} {'':<25} "
              f"{greedy_summary['avg_delivery_priority_mean']:.1f} ± {greedy_summary['avg_delivery_priority_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    print()
    
    # Max delivery priority
    if rl_summary:
        print(f"{'Max Delivery Priority (mean ± std)':<30} "
              f"{rl_summary['max_delivery_priority_mean']:.1f} ± {rl_summary['max_delivery_priority_std']:.1f}")
    else:
        print(f"{'Max Delivery Priority (mean ± std)':<30} {'N/A':<25}")
    
    if aco_summary:
        print(f"{'':<30} {'':<25} "
              f"{aco_summary['max_delivery_priority_mean']:.1f} ± {aco_summary['max_delivery_priority_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        print(f"{'':<30} {'':<25} {'':<25} "
              f"{greedy_summary['max_delivery_priority_mean']:.1f} ± {greedy_summary['max_delivery_priority_std']:.1f}")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    print()
    
    # Total late penalty
    if rl_summary:
        print(f"{'Total Late Penalty (mean ± std)':<30} "
              f"{rl_summary['total_late_penalty_mean']:.2f} ± {rl_summary['total_late_penalty_std']:.2f}")
    else:
        print(f"{'Total Late Penalty (mean ± std)':<30} {'N/A':<25}")
    
    if aco_summary:
        print(f"{'':<30} {'':<25} "
              f"{aco_summary['total_late_penalty_mean']:.2f} ± {aco_summary['total_late_penalty_std']:.2f}")
    else:
        print(f"{'':<30} {'':<25} {'N/A':<25}")
    
    if greedy_summary:
        print(f"{'':<30} {'':<25} {'':<25} "
              f"{greedy_summary['total_late_penalty_mean']:.2f} ± {greedy_summary['total_late_penalty_std']:.2f}")
    else:
        print(f"{'':<30} {'':<25} {'':<25} {'N/A':<25}")
    
    print()
    print("-" * 100)
    
    # Determine winner
    summaries = {
        'RL': rl_summary,
        'ACO': aco_summary,
        'Greedy': greedy_summary
    }
    
    valid_summaries = {k: v for k, v in summaries.items() if v}
    
    if len(valid_summaries) > 0:
        best_deliveries = max(s['deliveries_mean'] for s in valid_summaries.values())
        
        print("\nWINNER:")
        for name, summary in valid_summaries.items():
            if summary['deliveries_mean'] == best_deliveries:
                print(f"   {name} Algorithm (Highest average deliveries: {best_deliveries:.1f})")
                break
    
    print("=" * 100)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare RL, ACO, and Greedy algorithms')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run for each algorithm (default: 10)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained RL model (default: None, auto-detects rl_model.pth if exists, else uses greedy fallback)')
    parser.add_argument('--seed', type=int, default=7,
                       help='Random seed for environment (default: 7)')
    
    args = parser.parse_args()
    
    print("=" * 100)
    print("ALGORITHM COMPARISON")
    print("=" * 100)
    print(f"Configuration:")
    print(f"  Episodes per algorithm: {args.episodes}")
    print(f"  Random seed: {args.seed}")
    import os
    if args.model_path:
        model_display = args.model_path
    elif os.path.exists("rl_model.pth"):
        model_display = "rl_model.pth (auto-detected)"
    else:
        model_display = "None (using greedy fallback)"
    print(f"  RL model path: {model_display}")
    print("=" * 100)
    
    # run all algorithms
    rl_stats = run_rl_algorithm(n_episodes=args.episodes, seed=args.seed, model_path=args.model_path)
    aco_stats = run_aco_algorithm(n_episodes=args.episodes, seed=args.seed)
    greedy_stats = run_greedy_algorithm(n_episodes=args.episodes, seed=args.seed)
    
    # print comparison
    print_comparison_table(rl_stats, aco_stats, greedy_stats)


if __name__ == "__main__":
    main()

