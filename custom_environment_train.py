import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from custom_environment_RL import WarehouseEnv

# ---------------------------
# Repro & Device
# ---------------------------
def set_seed(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Masked Categorical helpers
# ---------------------------
def masked_logits_fn(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    very_neg = torch.finfo(logits.dtype).min / 2
    masked = logits.clone()
    masked[mask == 0] = very_neg
    return masked

def masked_sample(logits: torch.Tensor, mask: torch.Tensor):
    masked = masked_logits_fn(logits, mask)
    dist = torch.distributions.Categorical(logits=masked)
    a = dist.sample()
    logp = dist.log_prob(a)
    return a, logp, dist

def masked_logp(logits: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor):
    masked = masked_logits_fn(logits, mask)
    dist = torch.distributions.Categorical(logits=masked)
    return dist.log_prob(actions), dist


# ---------------------------
# Actor-Critic Network
# ---------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs: torch.Tensor):
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value


# ---------------------------
# Overflow-safe Rollout Buffer
# ---------------------------
class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, act_dim: int):
        self.max = int(size)
        self.obs = np.zeros((self.max, obs_dim), dtype=np.float32)
        self.mask = np.zeros((self.max, act_dim), dtype=np.int8)
        self.actions = np.zeros((self.max,), dtype=np.int64)
        self.logp = np.zeros((self.max,), dtype=np.float32)
        self.rews = np.zeros((self.max,), dtype=np.float32)
        self.vals = np.zeros((self.max,), dtype=np.float32)
        self.dones = np.zeros((self.max,), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, mask, action, logp, rew, val, done) -> bool:
        if self.ptr >= self.max:
            return False
        self.obs[self.ptr] = obs
        self.mask[self.ptr] = mask
        self.actions[self.ptr] = action
        self.logp[self.ptr] = logp
        self.rews[self.ptr] = rew
        self.vals[self.ptr] = val
        self.dones[self.ptr] = done
        self.ptr += 1
        return True

    def ready(self) -> bool:
        return self.ptr >= self.max

    def get(self):
        assert self.ready(), "Buffer not full yet"
        data = dict(
            obs=torch.from_numpy(self.obs),
            mask=torch.from_numpy(self.mask.astype(np.int64)),
            actions=torch.from_numpy(self.actions),
            logp=torch.from_numpy(self.logp),
            rews=torch.from_numpy(self.rews),
            vals=torch.from_numpy(self.vals),
            dones=torch.from_numpy(self.dones),
        )
        self.ptr = 0  # reuse arrays
        return {k: v.to(device) for k, v in data.items()}


# ---------------------------
# Advantage (GAE-Lambda)
# ---------------------------
def compute_gae_from_flat(rews, vals, dones, gamma=0.99, lam=0.95):
    T = rews.shape[0]
    adv = torch.zeros_like(rews, device=rews.device)
    # next values: shift left, last next value = 0
    next_vals = torch.roll(vals, shifts=-1, dims=0)
    next_vals[-1] = 0.0
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rews[t] + gamma * next_vals[t] * nonterminal - vals[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + vals
    return adv, ret


# ---------------------------
# Env factory
# ---------------------------
def make_env(debug=False, curriculum_step=0):
    # curriculum learning: start with fewer tasks, gradually increase
    n_tasks = min(40, 15 + curriculum_step * 3)  # start with 15, add 3 every step, max 40
    return WarehouseEnv(
        n_agents=3,
        n_initial_tasks=n_tasks,
        task_spawn_rate=0.05,
        horizon_steps=600,
        K=10,
        render_mode=None,
        seed=7,
        debug=debug,
    )


# ---------------------------
# Training
# ---------------------------
def train(save_path: str = "rl_model.pth"):
    set_seed(7)

    # ----- Create env -----
    env = make_env(debug=False, curriculum_step=0)
    obs, info = env.reset()
    agent_ids = list(env.agents)
    obs_dim = env.observation_spaces[agent_ids[0]].shape[0]
    act_dim = env.action_spaces[agent_ids[0]].n

    # training configuration optimized for better learning
    steps_per_update = 8192      # total transitions (across ALL agents) per update
    minibatch_size   = 2048
    update_epochs    = 6
    total_updates    = 500      
    
    # PPO hyperparameters
    gamma = 0.99                 # Discount factor
    gae_lam = 0.95               # GAE lambda
    clip_ratio = 0.2             # PPO clip ratio
    ent_coef = 0.05              
    vf_coef = 0.5                # Value function loss coefficient
    max_grad_norm = 0.5          # Gradient clipping
    lr = 3e-4                    # Learning rate (slightly lower for more stable learning)                    

    net = ActorCritic(obs_dim, act_dim, hidden=256).to(device)
    optim_ = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.LinearLR(optim_, start_factor=1.0, end_factor=0.1, total_iters=total_updates)
    buf = RolloutBuffer(steps_per_update, obs_dim, act_dim)

    ep_ret = {aid: 0.0 for aid in agent_ids}
    ep_len = {aid: 0 for aid in agent_ids}
    deliveries_ep_accum = 0 
    global_steps = 0

    print("Starting training...")
    for update in range(1, total_updates + 1):
        # collect until buffer is full
        while not buf.ready():
            # stack current per-agent tensors
            obs_mat = np.stack([obs[a] for a in agent_ids], axis=0)                  
            mask_mat = np.stack([info[a]["action_mask"] for a in agent_ids], axis=0) 

            obs_t = torch.from_numpy(obs_mat).to(device).float()
            mask_t = torch.from_numpy(mask_mat.astype(np.int64)).to(device)

            with torch.no_grad():
                logits, values = net(obs_t)
                actions_t, logp_t, _ = masked_sample(logits, mask_t)

            actions = {aid: int(a.item()) for aid, a in zip(agent_ids, actions_t)}
            logp_np = logp_t.cpu().numpy()
            vals_np = values.cpu().numpy()

            next_obs, rew, term, trunc, info_next = env.step(actions)

            # per-step deliveries for robust logging (works with any env version)
            deliveries_ep_accum += info_next.get("__env__", {}).get("deliveries_step", 0)

            # add transitions PER AGENT; stop early if buffer fills mid-loop
            stop_collecting = False
            for i, aid in enumerate(agent_ids):
                ok = buf.add(
                    obs=obs_mat[i],
                    mask=mask_mat[i],
                    action=actions_t[i].item(),
                    logp=logp_np[i],
                    rew=float(rew[aid]),
                    val=float(vals_np[i]),
                    done=float(term[aid] or trunc[aid]),
                )
                ep_ret[aid] += rew[aid]
                ep_len[aid] += 1
                global_steps += 1
                if not ok: 
                    stop_collecting = True
                    break

            obs, info = next_obs, info_next

            if all(term[a] or trunc[a] for a in agent_ids):
                mean_ret = np.mean([ep_ret[a] for a in agent_ids])
                mean_len = np.mean([ep_len[a] for a in agent_ids])

                d_cum = info.get("__env__", {}).get("deliveries_cum", None)
                deliveries_to_report = d_cum if d_cum is not None else deliveries_ep_accum

                # idle time statistics
                env_info = info.get("__env__", {})
                idle_percentage = env_info.get("idle_percentage", 0)
                total_idle_steps = env_info.get("total_idle_steps", 0)
                idle_time_per_agent = env_info.get("idle_time_per_agent", {})

                curriculum_step = min(15, update // 5)
                n_tasks = min(40, 15 + curriculum_step * 3)
                
                # idle time per agent
                idle_per_agent_str = ", ".join([f"{agent}: {time}" for agent, time in idle_time_per_agent.items()])
                
                print(
                    f"[Update {update:03d}] Steps {global_steps:7d} | "
                    f"EpRet mean {mean_ret:.3f} | EpLen mean {int(mean_len)} | "
                    f"Deliveries {deliveries_to_report} | "
                    f"Deliveries/min {deliveries_to_report/(mean_len/60):.1f} | "
                    f"Tasks: {n_tasks} | "
                    f"Idle: {idle_percentage:.1f}% ({total_idle_steps} steps) | "
                    f"Per-agent: [{idle_per_agent_str}]"
                )

                # reset stats
                ep_ret = {aid: 0.0 for aid in agent_ids}
                ep_len = {aid: 0 for aid in agent_ids}
                deliveries_ep_accum = 0
                # gradually increase task difficulty (slower curriculum)
                curriculum_step = min(15, update // 10)
                env = make_env(debug=False, curriculum_step=curriculum_step)
                obs, info = env.reset()

            # if buffer filled inside the per-agent loop, break to PPO update
            if stop_collecting:
                break

        # PPO UPDATE
        batch = buf.get()
        # advantages/returns (flat)
        with torch.no_grad():
            adv, ret = compute_gae_from_flat(
                rews=batch["rews"],
                vals=batch["vals"],
                dones=batch["dones"],
                gamma=gamma,
                lam=gae_lam
            )
            # normalize advantages
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        N = batch["obs"].shape[0]
        idx = np.arange(N)

        for _ in range(update_epochs):
            np.random.shuffle(idx)
            for start in range(0, N, minibatch_size):
                end = start + minibatch_size
                mb = idx[start:end]

                obs_b = batch["obs"][mb].float()
                mask_b = batch["mask"][mb].long()
                act_b = batch["actions"][mb].long()
                old_logp_b = batch["logp"][mb].float()
                adv_b = adv[mb].float()
                ret_b = ret[mb].float()

                logits, values = net(obs_b)
                new_logp, _ = masked_logp(logits, mask_b, act_b)
                ratio = torch.exp(new_logp - old_logp_b)

                # policy loss (clipped surrogate)
                pg_unclipped = ratio * adv_b
                pg_clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_b
                pg_loss = -torch.min(pg_unclipped, pg_clipped).mean()

                # value loss
                v_loss = ((values - ret_b) ** 2).mean()

                # entropy (masked)
                ent = torch.distributions.Categorical(
                    logits=masked_logits_fn(logits, mask_b)
                ).entropy().mean()

                loss = pg_loss + vf_coef * v_loss - ent_coef * ent

                optim_.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optim_.step()

        # update learning rate
        scheduler.step()
        
        if update % 5 == 0:
            current_lr = optim_.param_groups[0]['lr']
            print(f"Update {update}/{total_updates} done. LR: {current_lr:.6f}")
        
        # save checkpoint periodically and at the end
        if update % 50 == 0 or update == total_updates:
            checkpoint_path = f"rl_model_checkpoint_update_{update}.pth"
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optim_.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'update': update,
                'obs_dim': obs_dim,
                'act_dim': act_dim,
            }, checkpoint_path)
            print(f" saved checkpoint to {checkpoint_path}")

    # Save final model
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optim_.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'update': total_updates,
        'obs_dim': obs_dim,
        'act_dim': act_dim,
    }, save_path)
    print(f"\n final model saved to {save_path}")

    print("Training finished. Running a short render episode...")
    env = make_env(debug=True, curriculum_step=15)
    env.render_mode = "human"
    obs, info = env.reset()
    done = {aid: False for aid in agent_ids}
    with torch.no_grad():
        steps = 0
        while not all(done.values()) and steps < 600:
            obs_mat = np.stack([obs[a] for a in agent_ids], axis=0)
            mask_mat = np.stack([info[a]["action_mask"] for a in agent_ids], axis=0)
            obs_t = torch.from_numpy(obs_mat).to(device).float()
            mask_t = torch.from_numpy(mask_mat.astype(np.int64)).to(device)

            logits, _ = net(obs_t)
            masked = masked_logits_fn(logits, mask_t)
            acts = torch.argmax(masked, dim=-1)
            actions = {aid: int(a.item()) for aid, a in zip(agent_ids, acts)}

            obs, rew, term, trunc, info = env.step(actions)
            env.render()
            done = {a: term[a] or trunc[a] for a in agent_ids}
            steps += 1
    env.close()


def run_rl_baseline(n_episodes: int = 10, model_path: str = None):
    import gymnasium
    
    print("running RL Algorithm Baseline...")
    print("=" * 80)
    
    # create environment
    env = make_env(debug=False, curriculum_step=15)
    
    # load or create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        # observation space from environment
        temp_env = make_env(debug=False, curriculum_step=15)
        obs_space = temp_env.observation_spaces[temp_env.possible_agents[0]]
        temp_env.close()
        
        net = ActorCritic(
            obs_dim=obs_space.shape[0],
            act_dim=11,
            hidden=256
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
    else:
        print("No trained model found. Running baseline with random policy...")
        # observation space from environment
        temp_env = make_env(debug=False, curriculum_step=15)
        obs_space = temp_env.observation_spaces[temp_env.possible_agents[0]]
        temp_env.close()
        
        # random policy instead of training
        net = ActorCritic(
            obs_dim=obs_space.shape[0],
            act_dim=11,
            hidden=256
        ).to(device)
        net.eval()
    
    all_stats = []
    agent_ids = env.possible_agents
    
    for episode in range(n_episodes):
        print(f"Episode {episode + 1}/{n_episodes}...")
        
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
            'idle_percentage': 0
        }
        
        with torch.no_grad():
            steps = 0
            while not all(done.values()) and steps < 600:
                obs_mat = np.stack([obs[a] for a in agent_ids], axis=0)
                mask_mat = np.stack([info[a]["action_mask"] for a in agent_ids], axis=0)
                obs_t = torch.from_numpy(obs_mat).to(device).float()
                mask_t = torch.from_numpy(mask_mat.astype(np.int64)).to(device)

                logits, _ = net(obs_t)
                masked = masked_logits_fn(logits, mask_t)
                acts = torch.argmax(masked, dim=-1)  # greedy eval
                actions = {aid: int(a.item()) for aid, a in zip(agent_ids, acts)}

                obs, rew, term, trunc, info = env.step(actions)
                done = {a: term[a] or trunc[a] for a in agent_ids}
                
                # accumulate statistics
                episode_stats['total_reward'] += sum(rew.values())
                episode_stats['steps'] += 1
                steps += 1
        
        # final episode statistics from environment info
        env_info = info.get("__env__", {})
        episode_stats['deliveries'] = env_info.get('deliveries_cum', 0)
        episode_stats['idle_time_per_agent'] = env_info.get('idle_time_per_agent', {})
        episode_stats['total_idle_steps'] = env_info.get('total_idle_steps', 0)
        episode_stats['episode_steps'] = env_info.get('episode_steps', 0)
        episode_stats['total_agent_steps'] = env_info.get('total_agent_steps', 0)
        episode_stats['idle_percentage'] = env_info.get('idle_percentage', 0)
        
        all_stats.append(episode_stats)
        
        # episode results
        deliveries = episode_stats['deliveries']
        idle_percentage = episode_stats['idle_percentage']
        total_idle_steps = episode_stats['total_idle_steps']
        idle_per_agent = episode_stats['idle_time_per_agent']
        
        # idle time per agent
        idle_per_agent_str = ", ".join([f"{agent}: {time}" for agent, time in idle_per_agent.items()])
        
        print(f"  Deliveries: {deliveries} | "
              f"Deliveries/min: {deliveries/(600/60):.1f} | "
              f"Idle: {idle_percentage:.1f}% ({total_idle_steps} steps) | "
              f"Per-agent: [{idle_per_agent_str}]")
    
    env.close()
    
    # calculate and print summary statistics
    print("\n" + "=" * 80)
    print("RL ALGORITHM SUMMARY")
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
        print(" RL algorithm performs excellently!")
    elif np.mean(deliveries) > 55:
        print(" RL algorithm performs very well!")
    elif np.mean(deliveries) > 50:
        print(" RL algorithm performs moderately")
    else:
        print(" RL algorithm underperforms")
        
    return all_stats


def run_rl_visualization(model_path: str = None):
    """Run RL algorithm with full visualization."""
    import gymnasium
    
    print("Running RL Algorithm with Visualization...")
    print("=" * 80)
    
    env = make_env(debug=True, curriculum_step=15)
    env.render_mode = "human"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        temp_env = make_env(debug=False, curriculum_step=15)
        obs_space = temp_env.observation_spaces[temp_env.possible_agents[0]]
        temp_env.close()
        
        net = ActorCritic(
            obs_dim=obs_space.shape[0],
            act_dim=11,
            hidden=256
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
    else:
        print("No trained model found. Training a new model first...")
        train()
        return
    
    
    obs, info = env.reset()
    done = {aid: False for aid in env.possible_agents}
    
    episode_stats = {
        'total_reward': 0,
        'steps': 0,
        'deliveries': 0,
        'idle_time_per_agent': {agent_id: 0 for agent_id in env.possible_agents},
        'total_idle_steps': 0,
        'episode_steps': 0,
        'total_agent_steps': 0,
        'idle_percentage': 0
    }
    
    with torch.no_grad():
        steps = 0
        while not all(done.values()) and steps < 600:
            obs_mat = np.stack([obs[a] for a in env.possible_agents], axis=0)
            mask_mat = np.stack([info[a]["action_mask"] for a in env.possible_agents], axis=0)
            obs_t = torch.from_numpy(obs_mat).to(device).float()
            mask_t = torch.from_numpy(mask_mat.astype(np.int64)).to(device)

            logits, _ = net(obs_t)
            masked = masked_logits_fn(logits, mask_t)
            acts = torch.argmax(masked, dim=-1)  # greedy eval
            actions = {aid: int(a.item()) for aid, a in zip(env.possible_agents, acts)}

            obs, rew, term, trunc, info = env.step(actions)
            env.render()
            done = {a: term[a] or trunc[a] for a in env.possible_agents}
            
            # accumulate statistics
            episode_stats['total_reward'] += sum(rew.values())
            episode_stats['steps'] += 1
            steps += 1
            
            # print progress every 50 steps
            if steps % 50 == 0:
                env_info = info.get("__env__", {})
                current_deliveries = env_info.get('deliveries_cum', 0)
                print(f"Step {steps}: Deliveries={current_deliveries}")
    
    # get final episode statistics
    env_info = info.get("__env__", {})
    episode_stats['deliveries'] = env_info.get('deliveries_cum', 0)
    episode_stats['idle_time_per_agent'] = env_info.get('idle_time_per_agent', {})
    episode_stats['total_idle_steps'] = env_info.get('total_idle_steps', 0)
    episode_stats['episode_steps'] = env_info.get('episode_steps', 0)
    episode_stats['total_agent_steps'] = env_info.get('total_agent_steps', 0)
    episode_stats['idle_percentage'] = env_info.get('idle_percentage', 0)
    
    # print final results
    print("\n" + "=" * 80)
    print("RL ALGORITHM VISUALIZATION RESULTS")
    print("=" * 80)
    
    deliveries = episode_stats['deliveries']
    idle_percentage = episode_stats['idle_percentage']
    total_idle_steps = episode_stats['total_idle_steps']
    idle_per_agent = episode_stats['idle_time_per_agent']
    
    # idle time per agent
    idle_per_agent_str = ", ".join([f"{agent}: {time}" for agent, time in idle_per_agent.items()])
    
    print(f"Deliveries: {deliveries} | "
          f"Deliveries/min: {deliveries/(600/60):.1f} | "
          f"Idle: {idle_percentage:.1f}% ({total_idle_steps} steps) | "
          f"Per-agent: [{idle_per_agent_str}]")
    
    print(f"\nEpisode completed successfully!")
    print(f"RL algorithm achieved {deliveries} deliveries in {episode_stats['episode_steps']} steps.")
    
    env.close()


def run_rl_save_png(model_path: str = None, save_final: bool = True, save_animation: bool = False):
    """Run RL algorithm and save simulation as PNG/GIF."""
    import gymnasium
    from PIL import Image
    import io
    
    print("Running RL Algorithm and Saving PNG...")
    print("=" * 80)
    
    # environment with RGB render mode for saving
    env = make_env(debug=True, curriculum_step=15)
    env.render_mode = "rgb_array"
    
    # load or create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_path and os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        temp_env = make_env(debug=False, curriculum_step=15)
        obs_space = temp_env.observation_spaces[temp_env.possible_agents[0]]
        temp_env.close()
        
        net = ActorCritic(
            obs_dim=obs_space.shape[0],
            act_dim=11,
            hidden=256
        ).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
    else:
        print("No trained model found. Using random policy...")
        temp_env = make_env(debug=False, curriculum_step=15)
        obs_space = temp_env.observation_spaces[temp_env.possible_agents[0]]
        temp_env.close()
        
        net = ActorCritic(
            obs_dim=obs_space.shape[0],
            act_dim=11,
            hidden=256
        ).to(device)
        net.eval()
    
    
    obs, info = env.reset()
    done = {aid: False for aid in env.possible_agents}
    
    episode_stats = {
        'total_reward': 0,
        'steps': 0,
        'deliveries': 0,
        'idle_time_per_agent': {agent_id: 0 for agent_id in env.possible_agents},
        'total_idle_steps': 0,
        'episode_steps': 0,
        'total_agent_steps': 0,
        'idle_percentage': 0
    }
    
    frames = []
    final_frame = None
    
    with torch.no_grad():
        steps = 0
        while not all(done.values()) and steps < 600:
            obs_mat = np.stack([obs[a] for a in env.possible_agents], axis=0)
            mask_mat = np.stack([info[a]["action_mask"] for a in env.possible_agents], axis=0)
            obs_t = torch.from_numpy(obs_mat).to(device).float()
            mask_t = torch.from_numpy(mask_mat.astype(np.int64)).to(device)

            logits, _ = net(obs_t)
            masked = masked_logits_fn(logits, mask_t)
            acts = torch.argmax(masked, dim=-1)  # greedy eval
            actions = {aid: int(a.item()) for aid, a in zip(env.possible_agents, acts)}

            obs, rew, term, trunc, info = env.step(actions)
            
            # capture frame
            frame = env.render()
            if frame is not None:
                # convert numpy array to PIL Image
                pil_frame = Image.fromarray(frame)
                frames.append(pil_frame.copy())
                final_frame = pil_frame.copy()
            
            done = {a: term[a] or trunc[a] for a in env.possible_agents}
            
            # accumulate statistics
            episode_stats['total_reward'] += sum(rew.values())
            episode_stats['steps'] += 1
            steps += 1
            
            # print progress every 50 steps
            if steps % 50 == 0:
                env_info = info.get("__env__", {})
                current_deliveries = env_info.get('deliveries_cum', 0)
                print(f"Step {steps}: Deliveries={current_deliveries}, Frames captured={len(frames)}")
    
    # final episode statistics
    env_info = info.get("__env__", {})
    episode_stats['deliveries'] = env_info.get('deliveries_cum', 0)
    episode_stats['idle_time_per_agent'] = env_info.get('idle_time_per_agent', {})
    episode_stats['total_idle_steps'] = env_info.get('total_idle_steps', 0)
    episode_stats['black_time_per_agent'] = env_info.get('black_time_per_agent', {})
    episode_stats['total_black_steps'] = env_info.get('total_black_steps', 0)
    episode_stats['episode_steps'] = env_info.get('episode_steps', 0)
    episode_stats['total_agent_steps'] = env_info.get('total_agent_steps', 0)
    episode_stats['idle_percentage'] = env_info.get('idle_percentage', 0)
    episode_stats['black_percentage'] = env_info.get('black_percentage', 0)
    
    deliveries = episode_stats['deliveries']
    idle_percentage = episode_stats['idle_percentage']
    black_percentage = episode_stats['black_percentage']
    
    # final frame as PNG
    if save_final and final_frame is not None:
        png_filename = f"rl_simulation_final_deliveries_{deliveries}_idle_{idle_percentage:.1f}pct.png"
        final_frame.save(png_filename)
        print(f"\n💾 Final frame saved as: {png_filename}")
    
    # animation as GIF
    if save_animation and len(frames) > 10:
        # frames for GIF (every nth frame to keep file size reasonable)
        sample_rate = max(1, len(frames) // 30)
        sample_frames = frames[::sample_rate]
        
        gif_filename = f"rl_simulation_animation_deliveries_{deliveries}_steps_{len(frames)}.gif"
        sample_frames[0].save(
            gif_filename,
            save_all=True,
            append_images=sample_frames[1:],
            duration=300,  # 300ms per frame
            loop=0  # Infinite loop
        )
        
    
    # print final results
    print("\n" + "=" * 80)
    print("RL ALGORITHM PNG SAVE RESULTS")
    print("=" * 80)
    
    idle_per_agent = episode_stats['idle_time_per_agent']
    black_per_agent = episode_stats['black_time_per_agent']
    idle_per_agent_str = ", ".join([f"{agent}: {time}" for agent, time in idle_per_agent.items()])
    black_per_agent_str = ", ".join([f"{agent}: {time}" for agent, time in black_per_agent.items()])
    
    print(f"Deliveries: {deliveries} | "
          f"Deliveries/min: {deliveries/(600/60):.1f} | "
          f"Idle: {idle_percentage:.1f}% | "
          f"Black: {black_percentage:.1f}% | "
          f"Idle Time: [{idle_per_agent_str}] | "
          f"Black Time: [{black_per_agent_str}]")
    
    print(f"\nEpisode completed successfully!")
    print(f"RL algorithm achieved {deliveries} deliveries in {episode_stats['episode_steps']} steps.")
    print(f"Captured {len(frames)} frames during simulation.")
    
    env.close()


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--baseline":
            model_path = "rl_model.pth" if len(sys.argv) < 3 else sys.argv[2]
            episodes = 20 if len(sys.argv) < 4 else int(sys.argv[3])
            run_rl_baseline(n_episodes=episodes, model_path=model_path)
        elif sys.argv[1] == "--visualize":
            model_path = "rl_model.pth" if len(sys.argv) < 3 else sys.argv[2]
            run_rl_visualization(model_path=model_path)
        elif sys.argv[1] == "--save-png":
            model_path = "rl_model.pth" if len(sys.argv) < 3 else sys.argv[2]
            save_animation = len(sys.argv) > 3 and sys.argv[3] == "--animation"
            run_rl_save_png(model_path=model_path, save_final=True, save_animation=save_animation)
        else:
            print("Usage: python custom_environment_train.py [--train [model_path]|--baseline [model_path] [episodes]|--visualize [model_path]|--save-png [model_path] [--animation]]")
            print("  --train [model_path]: Train new model and save to model_path (default: rl_model.pth)")
            print("  --baseline: Run detailed baseline comparison")
            print("  --visualize: Run with visualization")
            print("  --save-png: Save final frame as PNG")
            print("  --save-png [model] --animation: Also save animated GIF")
            print("  (no args): Train new model and save to rl_model.pth")
    else:
        # train new model (default behavior)
        train(save_path="rl_model.pth")
