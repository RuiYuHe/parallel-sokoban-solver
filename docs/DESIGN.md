# Design Notes

## Background

[Sokoban](https://en.wikipedia.org/wiki/Sokoban) is a box-pushing puzzle: a
player pushes boxes (never pulls) around a grid until every box sits on a
target square. It's PSPACE-complete, and full boards blow up into a huge
state space, so the assignment is really a search-and-parallelism problem
wearing a puzzle-game costume.

The map format (`#` wall, `x`/`X` box off/on target, `o`/`O`/`!` player
off/on-target/on-`@`, `.` target, `@` a special floor tile) and the output
contract (a `WASD` move string) come from the course's reference tooling and
grading harness, which aren't part of this repo — only `src/solver.cpp` is
this project's own work. A minimal example map:

```
#########
#  xox..#
#   #####
#########
```

## State representation

A search state is a single `std::string`: byte 0 is the player's linear
position (`row * width + col`), the remaining bytes are box positions,
kept sorted so that two states with the same boxes-and-player-position hash
and compare equal regardless of box discovery order. Packing each square
index into a single byte keeps states small and hashing cheap, at the cost
of capping supported maps to 256 cells (16×16) — none of the graded test
boards come close to that, but it's a real ceiling, not a hypothetical one.

## Precompute phase

Before any search starts, two tables are built once from the static board:

- **True distances**: a reverse BFS from every target square gives the exact
  walking distance (accounting for walls) from that target to every other
  square. Nine test boards deep, this is already a meaningfully better
  signal than Manhattan distance, which ignores walls entirely.
- **Dead squares**: a square is "dead" if no sequence of pushes can ever get
  a box off it and onto a target — computed by flood-filling backward from
  each target along the *pull* direction (the reverse of a push) so any
  square not reachable that way can never have had a box pushed out of it.
  A box pushed onto a dead square is a wasted branch and gets pruned in
  `generate_successors` before it ever reaches the frontier.

## Heuristic: greedy box→target assignment

![Heuristic: greedy box-to-target assignment using precomputed true distances](heuristic.svg)

Manhattan-distance heuristics ignore which box goes to which target, so they
routinely undercount by summing each box's distance to its *nearest* target
even when two boxes want the same one. `calculate_heuristic` instead does a
one-shot greedy matching over the precomputed true-distance table: repeatedly
pick the cheapest remaining (box, target) pair, assign it, remove both from
the pool, and repeat until every box is assigned. The sum of the chosen
distances is the heuristic value.

This is a real improvement over plain Manhattan distance, but it is *not*
the true minimum-cost assignment — that requires solving a bipartite
matching (e.g. the Hungarian algorithm) exactly, which greedy nearest-pair
selection only approximates. See [Known rough edges](#known-rough-edges-documented-not-fixed)
for what that costs in practice.

## Parallel search: batched master/worker A*

![Batched master/worker A* loop: master pulls a batch under lock, OpenMP workers expand it lock-free into private buffers, master merges](architecture.svg)

The obvious way to parallelize A* — every worker thread pops one node off a
shared priority queue under a lock — turns the queue into a serialization
point: threads spend more time waiting on the lock than searching. This repo
instead batches the locking:

1. **Pull** — the master thread locks `frontier_mutex`, pops the best
   `BATCH_SIZE` (256) nodes off the global priority queue into
   `current_batch`, and unlocks.
2. **Expand** — `#pragma omp parallel for schedule(dynamic)` fans the batch
   out across worker threads. Each thread expands its assigned nodes
   (goal check, staleness check against `cost_map`, successor generation)
   and appends results to its own `thread_results[thread_id]` buffer —
   no locks, no shared writes, during this phase.
3. **Merge** — the master re-locks `frontier_mutex` and sequentially folds
   every thread's buffer back into `cost_map`, `parent_map`, and the
   frontier, deduplicating by a reachability-aware cost key (below).

The lock is only ever held for a queue-sized `pop`/`push` burst, never for
the actual search work, so contention stays low regardless of thread count.

### Why states need a *reachability* key, not just box positions

Two states with identical box layouts but the player standing in different
unreachable-from-each-other spots are genuinely different game states — but
two states where the player merely stands on *different squares of the same
reachable region* are the same state for search purposes, since the player
can walk anywhere in that region for free. `get_cost_key` reflects this: it
floods-fill from the player's position under the current box layout and uses
the *lowest-numbered square in that reachable region* as a canonical stand-in
for "where the player is," rather than the player's exact square. Skipping
this collapses the branching factor dramatically, since otherwise every
distinct walking position before a push looks like a new state.

### Getting the concurrency model right

The final design in `src/solver.cpp` is the fourth iteration, not the first:

1. **Per-node locking.** The first parallel attempt had every OpenMP thread
   independently lock `frontier_mutex`, pop one node, unlock, expand it, and
   lock again to push results — the design described in the report as
   "even slower than \[serial\] BFS and often suffer\[ing\] from deadlock,"
   since every thread contends on the same lock for every single node.
2. **Batching, with a broken dedup key.** Switching to batches fixed the
   contention, but the first batched version deduplicated states by a
   coarse connected-component ("zone") ID for the player's region instead
   of an exact reachability BFS — cheaper to compute, but too coarse: it
   could conflate genuinely different reachable regions that happened to
   share a zone ID under the *previous* box layout, corrupting search
   correctness.
3. **Exact reachability key.** Replacing the zone ID with the
   BFS-computed canonical square described above fixed correctness while
   keeping the batched, lock-light structure.
4. **Shrinking the critical section further.** The reachability key itself
   requires a BFS to compute. The final version moved that computation from
   the *serial, locked* merge step into the *parallel* expand step (each
   worker computes its own successors' keys before handing them back), so
   the only work left inside the lock is bookkeeping — no expensive
   recomputation.

## Known rough edges (documented, not fixed)

- **Batching trades away strict best-first optimality.** Textbook A* is only
  guaranteed to return the shortest solution because it expands the single
  best node on the frontier and stops the instant a popped node is a goal —
  at that point nothing left in the queue can possibly be cheaper. This
  solver instead pulls up to 256 nodes at once and accepts *any* goal state
  found anywhere in that batch, even one deeper in priority order than
  another, not-yet-expanded node in the same batch that might have led to a
  cheaper solution. In exchange for throughput, the solver gives up the
  guarantee that the first solution found is the shortest one — and because
  which node "wins" the race depends on OpenMP thread scheduling, different
  runs on the same input can return different-length solutions. This isn't
  hypothetical: a development log kept while tuning batch behavior recorded
  the same test case returning solution lengths from 109 to 181 moves across
  repeated runs.
- **`parent_map` is a `tbb::concurrent_unordered_map` it no longer needs to
  be.** It made sense in the per-node-locking design, where each thread
  could in principle touch it independently. In the final batched design,
  every write to `parent_map` happens inside the same `frontier_mutex`
  critical section that guards the frontier itself — so a plain
  `std::unordered_map` would already be safe, and the TBB dependency is a
  holdover rather than a requirement.
- **256-cell map ceiling** from the single-byte position encoding (see
  [State representation](#state-representation)).

## OpenMP vs. pthreads

OpenMP's `#pragma omp parallel for` made the batched expand step nearly free
to write — the code stays standard C++ that degrades gracefully to serial
execution if compiled without OpenMP support, and scheduling/load-balancing
across the batch is handled by `schedule(dynamic)` rather than by hand.
That convenience is also its ceiling: OpenMP's fork-join model fits this
solver's data-parallel "expand a batch of independent nodes" step well, but
would be awkward for the irregular, stateful coordination the master/merge
step needs — which is exactly why that part is written with a raw
`std::mutex` instead of an OpenMP construct. pthreads (or, here, a manually
managed mutex over an OpenMP region) gives that low-level control back, at
the cost of the programmer manually owning every lock, wakeup, and thread
lifecycle decision — real flexibility, but also the most common source of
deadlocks and races, including the one this project hit in its first
parallel attempt.
