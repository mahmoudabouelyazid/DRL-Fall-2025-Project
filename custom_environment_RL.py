import random
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces

Coord = Tuple[int, int]

# ======================
# Grid & Pathfinder
# ======================
class Maze:
    """Grid maze with walls. 0 = free, 1 = wall."""
    def __init__(self, layout: List[str]):
        maxw = max(len(r) for r in layout)
        self.h = len(layout)
        self.w = maxw
        self.grid = np.ones((self.h, self.w), dtype=np.uint8)
        for y, row in enumerate(layout):
            for x, ch in enumerate(row.ljust(maxw)):
                self.grid[y, x] = 1 if ch == "#" else 0
        self.free_cells = [(y, x) for y in range(self.h) for x in range(self.w) if self.grid[y, x] == 0]

    def is_free(self, c: Coord) -> bool:
        y, x = c
        return 0 <= y < self.h and 0 <= x < self.w and self.grid[y, x] == 0

    def sample_free(self, rng: random.Random) -> Coord:
        return rng.choice(self.free_cells)


class GridAStar:
    """A* on 4-connected grid with walls."""
    def __init__(self, maze: Maze):
        self.m = maze

    def neighbors(self, c: Coord):
        y, x = c
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if self.m.is_free((ny, nx)):
                yield (ny, nx)

    def plan(self, start: Coord, goal: Coord) -> List[Coord]:
        if not self.m.is_free(start) or not self.m.is_free(goal):
            return []
        import heapq
        h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
        openq = []
        gscore = {start: 0}
        parent = {start: None}
        heapq.heappush(openq, (h(start), 0, start))
        closed = set()
        while openq:
            f, gs, cur = heapq.heappop(openq)
            if cur in closed:
                continue
            if cur == goal:
                path = []
                c = cur
                while c is not None:
                    path.append(c)
                    c = parent[c]
                path.reverse()
                return path[1:]
            closed.add(cur)
            for nb in self.neighbors(cur):
                ng = gscore[cur] + 1
                if ng < gscore.get(nb, 1e9):
                    gscore[nb] = ng
                    parent[nb] = cur
                    heapq.heappush(openq, (ng + h(nb), ng, nb))
        return []

class Task:
    __slots__ = ("id", "pickup", "drop", "assigned_to", "picked", "done", "spawn_t", "deadline", "category", "priority")
    def __init__(self, tid: int, pickup: Coord, drop: Coord, spawn_t: int, category: str, deadline: Optional[int] = None):
        self.id = tid
        self.pickup = pickup
        self.drop = drop
        self.assigned_to: Optional[str] = None
        self.picked = False
        self.done = False
        self.spawn_t = spawn_t
        self.deadline = deadline
        self.category = category
        self.priority = 1.0  # priority starts at 1, increases over time, reaches 100 when late

class Agent:
    __slots__ = ("pos", "path", "carrying", "color", "idle_time", "total_idle_time", "black_time", "total_black_time")
    def __init__(self, pos: Coord, color: str):
        self.pos = pos
        self.path: deque = deque()
        self.carrying: Optional[int] = None
        self.color = color
        self.idle_time = 0  # current consecutive idle steps
        self.total_idle_time = 0  # total idle steps in episode
        self.black_time = 0  # current consecutive steps not carrying anything
        self.total_black_time = 0  # total steps not carrying anything in episode


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class WarehouseEnv(ParallelEnv):
    """
    Warehouse:
      - Pickup points with blocks
      - Agents auto-follow shortest paths using A* algo and Manhattan distance to pickup and drop once assigned
      - Actions are assignment choices over top-K candidate tasks (+ noop)
      - Tasks have priority that increases over time (starts at 1.0, increases by 0.2 per step, caps at 100.0)
      - Tasks with priority >= 100.0 are considered "late" and incur penalties on delivery

    Rewards:
      +1.0 on delivery (base reward)
      +0.5 efficiency bonus for fast completions (decreases with task duration)
      -0.1 * (priority - 100) penalty for late deliveries (when priority >= 100)
      -0.001 for each step taken
      +0.4 for successful pickup
      +0.5 for task assignment
      -0.05 for idle agents

    """
    metadata = {"render_modes": ["human"]}
    
    # Task categories and their dedicated drop-off zones
    TASK_CATEGORIES = {
        "st1": {
            "dropoff_zones": [(0, 0), (0, 1), (0, 2)], 
            "color": "red"
        },
        "st2": {
            "dropoff_zones": [(0, 16), (0, 17), (0, 18)],
            "color": "orange"
        },
        "st3": {
            "dropoff_zones": [(11, 0), (11, 1), (11, 2)],
            "color": "blue"
        },
        "st4": {
            "dropoff_zones": [(11, 16), (11, 17), (11, 18)], 
            "color": "green"
        }
    }

    def __init__(
        self,
        maze_layout: Optional[List[str]] = None,
        n_agents: int = 3,
        n_initial_tasks: int = 20,
        task_spawn_rate: float = 0.05,
        horizon_steps: int = 600,
        K: int = 10,                     # more task options per agent
        render_mode: Optional[str] = None,
        seed: int = 0,
        debug: bool = False,         
    ):
        if maze_layout is None:
            maze_layout = [
                ".....#.....#...#...",  
                "# .....#.....#... #",
                "# ### # ### # # # #",
                "# #   #     # # # #",
                "# # ### ### # # # #",
                "# # #   #   #   # #",
                "#   # # # # ### # #",
                "### # # # #   # # #",
                "#   #   # ### #   #",
                "# ### ###   # ### #",
                "#     .     .     #",
                ".....#.....#...#...",
            ]

        self.maze = Maze(maze_layout)
        self.astar = GridAStar(self.maze)

        self.agents = [f"a_{i}" for i in range(n_agents)]
        self.possible_agents = list(self.agents)
        palette = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                   "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
        self._palette = [palette[i % len(palette)] for i in range(n_agents)]

        self._rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.horizon_steps = int(horizon_steps)
        self.task_spawn_rate = float(task_spawn_rate)
        self.K = int(K)

        self.agent_state: Dict[str, Agent] = {}
        self.tasks: Dict[int, Task] = {}
        self.backlog: List[int] = []
        self._next_tid = 0

        self.render_mode = render_mode
        self._fig = None
        self._ax = None

        # observation/action spaces
        self.self_dim = 3
        self.task_dim = 7  # (py, px, dy, dx, d_agent_pick, d_pick_drop, priority) --this is normalized--
        obs_dim = self.self_dim + self.K * self.task_dim
        self.observation_spaces = {
            a: spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for a in self.agents
        }
        self.n_actions = self.K + 1  # K choices + noop
        self.action_spaces = {a: spaces.Discrete(self.n_actions) for a in self.agents}

        self.t = 0
        self._n_initial_tasks = int(n_initial_tasks)

        # delivery counters
        self._deliveries_step = 0
        self._deliveries_episode = 0

        # debug
        self.debug = debug
        self._no_delivery_steps = 0  # watchdog

    # ------------- PettingZoo API -------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.t = 0
        self.tasks.clear()
        self.backlog.clear()
        self._next_tid = 0
        self.agent_state.clear()
        self._deliveries_step = 0
        self._deliveries_episode = 0
        self._no_delivery_steps = 0
        
        # reset idle time tracking
        self.idle_time_per_agent = {agent: 0 for agent in self.agents}  # track idle time per agent
        self.total_idle_steps = 0
        
        # reset black time tracking (when agents are not carrying packages)
        self.black_time_per_agent = {agent: 0 for agent in self.agents}  # track black time per agent
        self.total_black_steps = 0

        for i, a in enumerate(self.agents):
            pos = self.maze.sample_free(self._rng)
            self.agent_state[a] = Agent(pos=pos, color=self._palette[i])
            if self.debug:
                print(f"[RESET] Agent {a} @ {pos}")

        for _ in range(self._n_initial_tasks):
            self._spawn_task()

        if self.debug:
            print(f"[RESET] Spawned {len(self.backlog)} tasks.")

        if self.render_mode == "human" and self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(7, 5))
            try:
                self._fig.canvas.manager.set_window_title("WarehouseEnv")
            except Exception:
                pass
            plt.show(block=False)

        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {"action_mask": self._mask(a)} for a in self.agents}
        infos["__env__"] = {"deliveries_step": self._deliveries_step, "deliveries_cum": self._deliveries_episode}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        rewards = {a: 0.0 for a in self.agents}
        terms = {a: False for a in self.agents}
        truncs = {a: False for a in self.agents}

        # assignment actions
        for a, act in actions.items():
            mask = self._mask(a)
            if act >= len(mask) or mask[act] == 0:
                if self.debug:
                    print(f"[t={self.t}] Agent {a} chose INVALID act={act} (masked); ignoring.")
                rewards[a] -= 0.1  # penalty for invalid action
                continue
            if act == self.K:
                if self.debug:
                    print(f"[t={self.t}] Agent {a} NOOP")
                rewards[a] -= 0.2  # stronger penalty for NOOP to encourage task assignment
                continue  # noop
            cand = self._topK(a)
            if act < len(cand):
                tid = cand[act]
                tk = self.tasks.get(tid)
                ag = self.agent_state[a]
                # only allow assignment if task is unassigned and agent is idle (no existing assignment)
                can_assign = (tk.assigned_to is None) and (not tk.done) and ag.carrying is None
                is_idle = not any(self.tasks[tid].assigned_to == a for tid in self.backlog if not self.tasks[tid].done)
                
                # IMPORTANT --> once an agent is assigned to a task, they cannot be reassigned until they complete it (avoids going back and forth and "too dynamic" assignments)
                if tk and can_assign and is_idle:
                    tk.assigned_to = a
                    ag.path.clear()
                    self._plan_append(ag, ag.pos, tk.pickup)
                    rewards[a] += 0.5  # high reward for successful task assignment
                    if self.debug:
                        print(f"[t={self.t}] ASSIGN a={a} -> task={tid} pickup={tk.pickup} drop={tk.drop} "
                              f"pos={ag.pos} path_len={len(ag.path)}")
                elif self.debug:
                    print(f"[t={self.t}] Agent {a} failed to assign task {tid}: "
                          f"assigned_to={tk.assigned_to if tk else None}, done={tk.done if tk else None}, carrying={ag.carrying}")

        self._progress_agents_and_tasks(rewards)
        self._track_idle_time()
        
        # apply idle penalties (mixing penalty for idle time plus no multiple assignments increases the throughput)
        self._apply_idle_penalties(rewards)
        

        # random spawns with increased rate towards end of episode
        remaining_steps = self.horizon_steps - self.t
        spawn_rate = self.task_spawn_rate
        if remaining_steps < 100: 
            spawn_rate *= 2.0  
        elif remaining_steps < 200:  
            spawn_rate *= 1.5 
            
        if self.np_rng.random() < spawn_rate:
            self._spawn_task()
            
        # emergency task spawning if too few tasks available
        available_tasks = len([tid for tid in self.backlog if not self.tasks[tid].done])

        if available_tasks < len(self.agents):
            # in the case that we have no enough tasks for all agents - spawn more
            if self.debug:
                print(f"[t={self.t}] EMERGENCY SPAWN: Only {available_tasks} tasks for {len(self.agents)} agents")
            for _ in range(len(self.agents) - available_tasks + 1):
                self._spawn_task()
            
        # periodic auto-reassignment and dynamic rebalancing DISABLED -- this needs to be disabled because agents CANNOT change direction
        # assignments are now fixed once made to prevent dynamic changes
        if self.t % 5 == 0:
            # auto-reassignment and dynamic rebalancing disabled to maintain stable assignments
            pass

        self.t += 1
        done = self.t >= self.horizon_steps
        
        # throughput bonus ------> reward agents for high delivery rates
        if done:
            total_deliveries = self._deliveries_episode
            throughput_bonus = total_deliveries * 0.15  # higher bonus per delivery
            for a in self.agents:
                rewards[a] += throughput_bonus
                
            # team coordination bonus ---> reward if all agents are actively working
            active_agents = sum(1 for ag in self.agent_state.values() 
                              if ag.carrying is not None or any(self.tasks[tid].assigned_to == ag for tid in self.backlog))
            if active_agents == len(self.agents):
                for a in self.agents:
                    rewards[a] += 1.0  # team coordination bonus
        
        for a in self.agents:
            terms[a] = done

        # watchdog: warn if too long without delivery
        if self._deliveries_step == 0:
            self._no_delivery_steps += 1
            if self.debug and (self._no_delivery_steps % 100 == 0):
                print(f"[t={self.t}] WARNING: {self._no_delivery_steps} consecutive steps with NO deliveries.")
        else:
            self._no_delivery_steps = 0

        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {"action_mask": self._mask(a)} for a in self.agents}
        # calculate idle time and black time statistics
        total_agent_steps = self.t * len(self.agents)
        idle_percentage = (self.total_idle_steps / total_agent_steps * 100) if total_agent_steps > 0 else 0
        black_percentage = (self.total_black_steps / total_agent_steps * 100) if total_agent_steps > 0 else 0
        
        infos["__env__"] = {
            "deliveries_step": self._deliveries_step,
            "deliveries_cum": self._deliveries_episode,
            "idle_time_per_agent": self.idle_time_per_agent.copy(),
            "total_idle_steps": self.total_idle_steps,
            "black_time_per_agent": self.black_time_per_agent.copy(),
            "total_black_steps": self.total_black_steps,
            "total_agent_steps": total_agent_steps,
            "idle_percentage": idle_percentage,
            "black_percentage": black_percentage,
            "episode_steps": self.t,
        }
        # reset per-step counter
        self._deliveries_step = 0
        return obs, rewards, terms, truncs, infos

    def render(self):
        if self.render_mode == "human":
            if self._fig is None or self._ax is None:
                plt.ion()
                self._fig, self._ax = plt.subplots(figsize=(10, 5))
                plt.tight_layout()
                plt.subplots_adjust(right=0.75)
                plt.show(block=False)

            self._ax.clear()
            H, W = self.maze.h, self.maze.w
            self._ax.imshow(self.maze.grid, cmap="gray_r", vmin=0, vmax=1, origin="upper")
            
            self._ax.set_title(f"t={self.t} | tasks={len(self.backlog)} | delivered(ep)={self._deliveries_episode}")
            self._ax.set_xticks([]); self._ax.set_yticks([])

            category_data = {}
            for tid in self.backlog:
                tk = self.tasks[tid]
                if tk.category not in category_data:
                    category_data[tk.category] = {"pickup": [], "drop": []}
                
                if not tk.picked and not tk.done:
                    category_data[tk.category]["pickup"].append((tk.pickup[0], tk.pickup[1]))
                if not tk.done:
                    category_data[tk.category]["drop"].append((tk.drop[0], tk.drop[1]))
            
            for category, data in category_data.items():
                category_info = self.TASK_CATEGORIES.get(category, {"color": "red"})
                color = category_info["color"]
                
                if data["pickup"]:
                    py, px = zip(*data["pickup"]) if data["pickup"] else ([], [])
                    self._ax.scatter(px, py, marker="o", s=30, c=color, edgecolors="black", alpha=0.7, label=f"{category} pickup")
                
                if data["drop"]:
                    dy, dx = zip(*data["drop"]) if data["drop"] else ([], [])
                    self._ax.scatter(dx, dy, marker="x", s=34, c=color, label=f"{category} drop")

            for name, ag in self.agent_state.items():
                y, x = ag.pos
                
                # determine agent color based on what they're carrying
                if ag.carrying is not None:
                    # use category color if the agent is carrying inventory
                    carried_task = self.tasks.get(ag.carrying)
                    if carried_task:
                        category_info = self.TASK_CATEGORIES.get(carried_task.category, {"color": "gray"})
                        agent_color = category_info["color"]
                        agent_label = f"{name} (carrying {carried_task.category})"
                    else:
                        agent_color = ag.color
                        agent_label = name
                else:
                    # use black color is the agent is not carrying any inventory
                    agent_color = "black"
                    agent_label = name
                
                self._ax.scatter([x], [y], marker="s", s=60, c=agent_color, edgecolors="black", label=agent_label)
                if len(ag.path) > 0:
                    ps = [(y, x)] + list(ag.path)
                    xs = [p[1] for p in ps]; ys = [p[0] for p in ps]
                    self._ax.plot(xs, ys, linestyle="--", linewidth=1.0, alpha=0.7, c=agent_color)

            # legend (dedup and move outside map)
            handles, labels = self._ax.get_legend_handles_labels()
            seen = set(); H2=[]; L2=[]
            for h, l in zip(handles, labels):
                if l in seen: continue
                seen.add(l); H2.append(h); L2.append(l)
            if H2:
                self._ax.legend(H2, L2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, framealpha=0.8)

            self._fig.canvas.draw_idle()
            plt.pause(0.15)
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        else:
            return None
    
    def _render_rgb_array(self):
        H, W = self.maze.h, self.maze.w
        
        # create RGB array
        rgb_array = np.zeros((H, W, 3), dtype=np.uint8)
        
        # set maze walls to white
        rgb_array[self.maze.grid == 0] = [255, 255, 255] 
        rgb_array[self.maze.grid == 1] = [240, 240, 240]  
        
        # Add agent colors
        for name, ag in self.agent_state.items():
            y, x = ag.pos
            
            # agent color based on what they're carrying
            if ag.carrying is not None:
                carried_task = self.tasks.get(ag.carrying)
                if carried_task:
                    category_info = self.TASK_CATEGORIES.get(carried_task.category, {"color": "yellow"})
                    if category_info["color"] == "yellow":
                        color = [255, 255, 0]
                    elif category_info["color"] == "orange":
                        color = [255, 165, 0]
                    elif category_info["color"] == "blue":
                        color = [0, 0, 255]
                    else:
                        color = [255, 255, 0]
                else:
                    color = [0, 0, 0]  # mark black if no task is found
            else:
                color = [0, 0, 0]  # mark black if no task is found
            rgb_array[y, x] = color
        
        # pickup/drop locations with category-specific colors
        category_colors = {
            "cm1": [255, 255, 0],
            "cm2": [255, 165, 0],    
            "cm3": [0, 0, 255],        
            "cm4": [0, 0, 255]           
        }
        
        for tid, tk in self.tasks.items():
            if not tk.done:
                # pickup location
                py, px = tk.pickup
                if rgb_array[py, px].sum() < 600:  # if not occupied by agent
                    pickup_color = category_colors.get(tk.category, [255, 0, 0])
                    rgb_array[py, px] = pickup_color
                
                # drop location 
                dy, dx = tk.drop
                if rgb_array[dy, dx].sum() < 600:  # only if not occupied by agent
                    drop_color = category_colors.get(tk.category, [255, 0, 0])
                    rgb_array[dy, dx] = drop_color
        
        return rgb_array

    def close(self):
        if self._fig is not None:
            try:
                plt.close(self._fig)
            except Exception:
                pass
        self._fig, self._ax = None, None

############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
    # ------------- Internals -------------
    def _sample_dropoff_location_for_category(self, category: str) -> Coord:
        """sample a drop-off location for a specific task category."""
        category_info = self.TASK_CATEGORIES.get(category)
        if not category_info:
            # general drop-off sampling if category not found
            return self._sample_dropoff_location()
        
        # available drop-off zones for this category
        available_zones = []
        for zone in category_info["dropoff_zones"]:
            if self.maze.is_free(zone):
                available_zones.append(zone)
        
        # if no zones available for this category, fall back to general sampling
        if not available_zones:
            return self._sample_dropoff_location()
    
        return self._rng.choice(available_zones)
    
    def _sample_dropoff_location(self) -> Coord:
        top_row_cells = [(0, x) for x in range(self.maze.w) if self.maze.is_free((0, x))]
        bottom_row_cells = [(self.maze.h - 1, x) for x in range(self.maze.w) if self.maze.is_free((self.maze.h - 1, x))]
        
        dropoff_cells = top_row_cells + bottom_row_cells
        
        # if no dropoff cells available in top/bottom rows, fall back to any free cell
        if not dropoff_cells:
            return self.maze.sample_free(self._rng)
        
        # select from available dropoff cells
        return self._rng.choice(dropoff_cells)

    def _spawn_task(self):
        pickup = self.maze.sample_free(self._rng)
        
        # assign a task category
        category = self._rng.choice(list(self.TASK_CATEGORIES.keys()))
        
        # generate drop-off point for the specific category
        drop = self._sample_dropoff_location_for_category(category)
        
        while drop == pickup:
            drop = self._sample_dropoff_location_for_category(category)
            
        tid = self._next_tid
        self._next_tid += 1
        tk = Task(tid, pickup, drop, spawn_t=self.t, category=category, deadline=None)
        self.tasks[tid] = tk
        self.backlog.append(tid)
        if self.debug:
            print(f"[t={self.t}] SPAWN task={tid} category={category} pickup={pickup} drop={drop}")

    def _plan_append(self, ag: Agent, start: Coord, goal: Coord):
        path = self.astar.plan(start, goal)
        if self.debug:
            print(f"[t={self.t}] PLAN from {start} -> {goal} | path_len={len(path)} | path_tail={path[-3:] if len(path)>=3 else path}")
        for p in path:
            ag.path.append(p)
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################

    def _progress_agents_and_tasks(self, rewards: Dict[str, float]):
        # update priority for all incomplete tasks
        for tid in self.backlog:
            tk = self.tasks.get(tid)
            if tk and not tk.done:
                tk.priority = tk.priority + 0.2
        
        for name, ag in self.agent_state.items():
            rewards[name] = rewards.get(name, 0.0) - 1e-3  # step penalty to penalize travel distance
            
            # reward for being assigned to a task (encourage task assignment)
            if ag.carrying is None:
                assigned_tasks = [tid for tid in self.backlog 
                                if self.tasks[tid].assigned_to == name and not self.tasks[tid].picked]
                if assigned_tasks:
                    rewards[name] += 0.1  # small reward for being assigned
                else:
                    # penalty for idle agents (not assigned to any task)
                    rewards[name] -= 0.05  # small penalty for being idle

            if len(ag.path) > 0:
                nxt = ag.path.popleft()
                if self.debug and (nxt == ag.pos):
                    print(f"[t={self.t}] NOTE: Agent {name} path step equals current pos {nxt}")
                ag.pos = nxt
                # small reward for making progress toward goal
                if ag.carrying is not None:
                    rewards[name] += 0.01  # progress reward when carrying
                else:
                    # find assigned task
                    assigned_task = next((tid for tid in self.backlog 
                                        if self.tasks[tid].assigned_to == name and not self.tasks[tid].picked), None)
                    if assigned_task:
                        rewards[name] += 0.005  # progress reward when going to pickup

            if ag.carrying is not None:
                tk = self.tasks.get(ag.carrying)
                if tk and (ag.pos == tk.drop):
                    tk.done = True
                    ag.carrying = None
                    # efficiency bonus: reward faster completions
                    task_duration = self.t - tk.spawn_t
                    efficiency_bonus = max(0, 0.5 - task_duration * 0.001)  # quick completion bonus
                    
                    # late delivery penalty: if priority >= 100, task is late
                    is_late = tk.priority >= 100.0
                    late_penalty = 0.0
                    if is_late:
                        # penalty increases with how late the task is (priority - 100)
                        late_penalty = (tk.priority - 100.0) * 0.1  # 0.1 penalty per priority point over 100
                        if self.debug:
                            print(f"[t={self.t}] LATE DELIVERY: task={tk.id} priority={tk.priority:.1f} penalty=-{late_penalty:.2f}")
                    
                    total_reward = 1.0 + efficiency_bonus - late_penalty
                    rewards[name] += total_reward
                    if tk.id in self.backlog:
                        self.backlog.remove(tk.id)
                    self._deliveries_step += 1
                    self._deliveries_episode += 1
                    if self.debug:
                        print(f"[t={self.t}] DROP  a={name} task={tk.id} at {tk.drop} | priority={tk.priority:.1f} | +{total_reward:.2f} reward | delivered(ep)={self._deliveries_episode}")
                    
                    # agent is now free to be assigned to a new task (no auto-reassignment)
                else:
                    if self.debug:
                        print(f"[t={self.t}] MOVE a={name} carrying={ag.carrying} pos={ag.pos} -> target_drop={tk.drop if tk else None} "
                              f"rem_path_len={len(ag.path)}")
                continue  # next agent

            for tid in list(self.backlog):
                tk = self.tasks[tid]
                if tk.assigned_to == name and (not tk.picked) and (ag.pos == tk.pickup):
                    tk.picked = True
                    ag.carrying = tk.id
                    ag.path.clear()
                    self._plan_append(ag, ag.pos, tk.drop)
                    rewards[name] += 0.4  # higher reward for successful pickup
                    if self.debug:
                        print(f"[t={self.t}] PICK  a={name} task={tid} at {tk.pickup} -> plan to drop {tk.drop} (path_len={len(ag.path)})")
                    break

            # if assigned but idle (replan if path empty)
            if ag.carrying is None:
                tid = next((tid for tid in self.backlog if self.tasks[tid].assigned_to == name and not self.tasks[tid].picked), None)
                if tid is not None and len(ag.path) == 0:
                    self._plan_append(ag, ag.pos, self.tasks[tid].pickup)
                    if self.debug:
                        tk = self.tasks[tid]
                        print(f"[t={self.t}] REPLAN to PICKUP a={name} task={tid} from pos={ag.pos} to {tk.pickup} (path_len={len(ag.path)})")

    def _auto_reassign_agent(self, agent_id: str):
        """automatically reassign an agent to the best available task to prevent idle time."""
        ag = self.agent_state[agent_id]
        if ag.carrying is not None:
            return
            
        # find best available task (including reassigning from other agents if necessary)
        available_tasks = []
        for tid in self.backlog:
            tk = self.tasks[tid]
            if not tk.done:
                # include unassigned tasks or tasks assigned to other agents
                if tk.assigned_to is None or tk.assigned_to != agent_id:
                    available_tasks.append(tid)
        
        if available_tasks:
            # use same scoring as _topK with workload consideration
            def score(tid):
                tk = self.tasks[tid]
                d1 = manhattan(ag.pos, tk.pickup)
                d2 = manhattan(tk.pickup, tk.drop)
                total_dist = d1 + d2
                age_bonus = (self.t - tk.spawn_t) * 0.01
                
                # add workload penalty for auto-reassignment
                current_workload = sum(1 for t_id in self.backlog 
                                     if self.tasks[t_id].assigned_to == agent_id and not self.tasks[t_id].done)
                workload_penalty = current_workload * 2
                
                return total_dist - age_bonus + workload_penalty
            
            available_tasks.sort(key=score)
            best_task_id = available_tasks[0]
            best_task = self.tasks[best_task_id]
            
            # assign the task (even if it was assigned to someone else)
            best_task.assigned_to = agent_id
            ag.path.clear()
            self._plan_append(ag, ag.pos, best_task.pickup)
            
            if self.debug:
                print(f"[t={self.t}] AUTO-ASSIGN a={agent_id} -> task={best_task_id} pickup={best_task.pickup} drop={best_task.drop}")
        else:
            # no tasks available - spawn a new one if possible
            if self.debug:
                print(f"[t={self.t}] No tasks available for agent {agent_id}, spawning new task")
            self._spawn_task()
            # try again with the new task
            if self.backlog:
                latest_task = max(self.backlog, key=lambda tid: self.tasks[tid].spawn_t)
                self.tasks[latest_task].assigned_to = agent_id
                ag.path.clear()
                self._plan_append(ag, ag.pos, self.tasks[latest_task].pickup)
                if self.debug:
                    print(f"[t={self.t}] AUTO-ASSIGN a={agent_id} -> NEW task={latest_task} pickup={self.tasks[latest_task].pickup}")

    def _dynamic_rebalance_tasks(self):
        """Dynamically rebalance tasks between agents for better efficiency."""
        # find all active tasks that could potentially be reassigned
        reassignable_tasks = []
        for tid in self.backlog:
            tk = self.tasks[tid]
            if not tk.done and not tk.picked and tk.assigned_to is not None:
                # task is assigned but not yet picked up - can be reassigned
                reassignable_tasks.append(tid)
        
        if not reassignable_tasks:
            return
        
        # for each reassignable task, find the best agent
        for tid in reassignable_tasks:
            tk = self.tasks[tid]
            current_agent = tk.assigned_to
            
            # calculate current agent's distance to task
            current_ag = self.agent_state[current_agent]
            current_distance = manhattan(current_ag.pos, tk.pickup)
            
            # find the closest available agent
            best_agent = None
            best_distance = current_distance
            best_score = float('inf')
            
            for agent_id in self.agents:
                ag = self.agent_state[agent_id]
                if ag.carrying is None:
                    distance = manhattan(ag.pos, tk.pickup)
                    
                    workload = sum(1 for t_id in self.backlog 
                                 if self.tasks[t_id].assigned_to == agent_id and not self.tasks[t_id].done)
                    
                    # combined score: distance + workload penalty
                    score = distance + workload * 5
                    
                    if score < best_score:
                        best_score = score
                        best_agent = agent_id
                        best_distance = distance
            
            # reassign if we found a significantly better agent (this is determined by a 20% improvement threshold)
            if (best_agent and best_agent != current_agent and 
                best_distance < current_distance * 0.8):
                
                if self.debug:
                    print(f"[t={self.t}] REBALANCE: Task {tid} reassigned from {current_agent} to {best_agent} "
                          f"(distance: {current_distance} -> {best_distance})")
                
                tk.assigned_to = best_agent
                new_ag = self.agent_state[best_agent]
                new_ag.path.clear()
                self._plan_append(new_ag, new_ag.pos, tk.pickup)

    def _topK(self, agent_id: str) -> List[int]:
        ag = self.agent_state[agent_id]
        if ag.carrying is not None:
            return []
        
        # check if agent already has an assignment. if so, return empty list
        has_existing_assignment = any(self.tasks[tid].assigned_to == agent_id for tid in self.backlog if not self.tasks[tid].done)
        if has_existing_assignment:
            return []
        
        # only show unassigned tasks to idle agents
        cand = [tid for tid in self.backlog if (not self.tasks[tid].done and self.tasks[tid].assigned_to is None)]
        def score(tid):
            tk = self.tasks[tid]
            d1 = manhattan(ag.pos, tk.pickup)
            d2 = manhattan(tk.pickup, tk.drop)
            total_dist = d1 + d2
            
            # prioritize tasks that are closer and easier (smaller total distance)
            # also consider task age ----> older tasks get slight priority
            age_bonus = (self.t - tk.spawn_t) * 0.01
            
            # priority bonus: higher priority tasks should be prioritized
            # priority ranges from 1 to 100+, so we subtract priority to make higher priority = lower score (better)
            # scale priority to have similar impact as distance (normalize to 0-50 range)
            priority_urgency = tk.priority * 0.5  # convert priority (1-100) to urgency score (0.5-50)
            
            return total_dist - age_bonus - priority_urgency
        cand.sort(key=score)
        return cand[:self.K]

    def _mask(self, agent_id: str):
        cand = self._topK(agent_id)
        mask = np.zeros(self.n_actions, dtype=np.int8)
        for i in range(len(cand)):
            mask[i] = 1
        mask[self.K] = 1  # noop
        return mask

    def _obs(self, agent_id: str):
        ag = self.agent_state[agent_id]
        H, W = self.maze.h, self.maze.w
        y, x = ag.pos
        carrying = 1.0 if ag.carrying is not None else 0.0
        self_feat = np.array([y/max(1, H-1), x/max(1, W-1), carrying], dtype=np.float32)

        cand = self._topK(agent_id)
        tf = []
        for tid in cand:
            tk = self.tasks[tid]
            py, px = tk.pickup; dy, dx = tk.drop
            d1 = manhattan(ag.pos, tk.pickup)
            d2 = manhattan(tk.pickup, tk.drop)
            # normalize priority: 1.0 -> 0.01, 100.0 -> 1.0
            priority_norm = tk.priority / 100.0
            tf.extend([
                py/max(1, H-1), px/max(1, W-1),
                dy/max(1, H-1), dx/max(1, W-1),
                d1/max(1, H+W), d2/max(1, H+W),
                priority_norm
            ])
        while len(tf) < self.K * self.task_dim:
            tf.extend([0.0] * self.task_dim)
        return np.concatenate([self_feat, np.array(tf, dtype=np.float32)], axis=0).astype(np.float32)
    
    def _track_idle_time(self):
        for agent_id, agent in self.agent_state.items():
            # check if agent is idle (not carrying, no path, no assignment)
            is_carrying = agent.carrying is not None
            has_path = len(agent.path) > 0
            has_assignment = any(self.tasks[tid].assigned_to == agent_id for tid in self.backlog if not self.tasks[tid].done)
            
            # track black time (when agent is not carrying anything)
            if not is_carrying:
                agent.black_time += 1
                agent.total_black_time += 1
                self.black_time_per_agent[agent_id] += 1
                self.total_black_steps += 1
            else:
                # agent is carrying something, then reset consecutive black time
                if agent.black_time > 0 and self.debug:
                    print(f"[t={self.t}] Agent {agent_id} stopped being black after {agent.black_time} steps")
                agent.black_time = 0
            
            # track idle time (when agent has no task assignment)
            if not is_carrying and not has_path and not has_assignment:
                # agent is idle - increment idle time
                agent.idle_time += 1
                agent.total_idle_time += 1
                self.idle_time_per_agent[agent_id] += 1
                self.total_idle_steps += 1
                
                if self.debug and agent.idle_time == 1:
                    print(f"[t={self.t}] Agent {agent_id} became idle")
            else:
                if agent.idle_time > 0 and self.debug:
                    print(f"[t={self.t}] Agent {agent_id} stopped being idle after {agent.idle_time} steps")
                agent.idle_time = 0
    
    def _apply_idle_penalties(self, rewards: Dict[str, float]):
        """Apply penalties for idle agents."""
        for agent_id, agent in self.agent_state.items():
            if agent.idle_time > 0:
                idle_penalty = min(0.5, 0.05 * agent.idle_time)  # cap at 0.5 penalty
                rewards[agent_id] -= idle_penalty
                
                if self.debug and agent.idle_time % 10 == 0:  # log every 10 idle steps
                    print(f"[t={self.t}] Agent {agent_id} idle penalty: -{idle_penalty:.3f} (idle for {agent.idle_time} steps)")
    
    def _fix_idle_agents(self):
        """Fix idle agents by assigning them to available tasks, but respect path commitment."""
        for agent_id, agent in self.agent_state.items():
            is_carrying = agent.carrying is not None
            has_path = len(agent.path) > 0
            has_assignment = any(self.tasks[tid].assigned_to == agent_id for tid in self.backlog if not self.tasks[tid].done)
            
            if not is_carrying and not has_path and not has_assignment:
                available_tasks = [tid for tid in self.backlog 
                                 if not self.tasks[tid].done and self.tasks[tid].assigned_to is None]
                
                if available_tasks:
                    # find the closest task
                    best_task = None
                    best_distance = float('inf')
                    
                    for tid in available_tasks:
                        task = self.tasks[tid]
                        distance = manhattan(agent.pos, task.pickup)
                        if distance < best_distance:
                            best_distance = distance
                            best_task = tid
                    
                    if best_task is not None:
                        task = self.tasks[best_task]
                        task.assigned_to = agent_id
                        agent.path.clear()
                        self._plan_append(agent, agent.pos, task.pickup)
                        
                        if self.debug:
                            print(f"[t={self.t}] IMMEDIATE FIX: Agent {agent_id} detected as idle")
                            print(f"[t={self.t}] AUTO-ASSIGN a={agent_id} -> task={best_task} pickup={task.pickup} drop={task.drop}")
