# parallel-sokoban-solver

A parallel A* solver for Sokoban. `src/solver.cpp` is the original work here
— the map format and grading harness come from the course and aren't
included (see [Background](docs/DESIGN.md#background)). The maintained
version searches for a minimum-push solution with an admissible wall-aware
assignment heuristic and batched OpenMP expansion.

![Batched master/worker A* search loop](docs/architecture.svg)

## Highlights

- **Batched master/worker parallelism** — the main thread pulls a batch from
  the shared frontier, OpenMP workers expand it into private buffers, and the
  main thread merges the results. [How it works](docs/DESIGN.md#parallel-search-batched-masterworker-a)
- **Exact wall-aware assignment heuristic** — Hungarian matching over
  precomputed BFS distances gives the minimum box-to-target assignment cost
  without assigning two boxes to the same target. Because those distances
  ignore other boxes and player positioning, the result remains a lower bound
  on the number of pushes. [Heuristic walkthrough](docs/DESIGN.md#heuristic-exact-boxtarget-assignment)
- **Batching without giving up minimum-push optimality** — finding a goal in
  a batch records an incumbent rather than stopping immediately; search ends
  only when the smallest remaining `f = pushes + h` cannot beat that goal.
  [Termination rule](docs/DESIGN.md#preserving-minimum-push-optimality)
- **The development history is preserved** — the submitted version went
  through per-node locking, a broken deduplication key, greedy assignment,
  and first-goal batch termination before the maintained version corrected
  those issues. [The evolution](docs/DESIGN.md#getting-the-concurrency-model-right)

## Installation

Builds as ordinary C++17 with OpenMP:

```sh
g++ -std=c++17 -O3 -fopenmp src/solver.cpp -o solver
./solver <input_file>
```

The solution move string (`WASD`) prints to stdout; solve time to stderr.
The A* cost is the **number of box pushes**, not the total number of player
walking moves in the returned `WASD` string.

## Design

Precomputation, exact assignment heuristic, reachability-based state
canonicalization, the batched parallel search loop, and the post-submission
correctness fixes are documented in [`docs/DESIGN.md`](docs/DESIGN.md).

## License

MIT, scoped to the files in `src/` and `docs/` — see [LICENSE](LICENSE).
