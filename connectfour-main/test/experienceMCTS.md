## Expérimentations MCTS

### Exploration constant  
- `Fixed time_limit = 0.95`, rollout = random.  
- Tested **c** ∈ {0.5, 1.0, 1.41, 2.0}.  
- Observed best trade-off around **c = 2.0**.

### Time budget  
- Fixed **c = 2.0**, rollout = random.  
- Tested `time_limit` ∈ {0.2, 0.5, 0.95, 1.5}.  
- Win rate increases with time, with diminishing returns.

### Intelligent rollouts  
- Compared random vs heuristic policy.  
- Heuristic prioritizes immediate wins, blocks opponent threats, prefers center.

### Early termination  
- Introduced `rollout_depth` and optional heuristic evaluation at cutoff for speed.
