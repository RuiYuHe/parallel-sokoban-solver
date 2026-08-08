# parallel-sokoban-solver

A parallel A* solver for Sokoban. `src/solver.cpp` is the original work here
— the map format and grading harness come from the course and aren't
included (see [Background](docs/DESIGN.md#background)).

![Batched master/worker A* search loop](docs/architecture.svg)

## Highlights

- **Batched master/worker parallelism** — a master thread pulls a batch off
  the shared frontier under lock, OpenMP workers expand it lock-free into
  private buffers, the master merges. [How it works](docs/DESIGN.md#parallel-search-batched-masterworker-a)
- **Wall-aware assignment heuristic** — a one-shot greedy matching over
  precomputed true (BFS) distances, not Manhattan distance, so two boxes
  never get double-counted onto the same target. [Heuristic walkthrough](docs/DESIGN.md#heuristic-greedy-boxtarget-assignment)
- **Four iterations to get the concurrency right**, including one that
  deadlocked and one with a real correctness bug —
  [the full evolution](docs/DESIGN.md#getting-the-concurrency-model-right)
- **A documented, not hidden, correctness tradeoff** — batching can return
  a solution that isn't the shortest one.
  [Known rough edges](docs/DESIGN.md#known-rough-edges-documented-not-fixed)

## Installation

Builds as ordinary C++17 against OpenMP and Intel TBB (`brew install tbb` /
`apt install libtbb-dev`):

```
g++ -std=c++17 -O3 -pthread -fopenmp -ltbb src/solver.cpp -o solver
./solver <input_file>
```

The solution move string (`WASD`) prints to stdout; solve time to stderr.

## Design

Precompute phase, the heuristic, the parallel search loop, the
correctness/throughput tradeoff of batching, and an OpenMP-vs-pthreads
comparison: [`docs/DESIGN.md`](docs/DESIGN.md).

## License

MIT, scoped to the files in `src/` and `docs/` — see [LICENSE](LICENSE).
