# Working on Mercury

- Inspect the implementation and relevant [docs](docs/) before proposing changes.
  Treat requests as authorization for routine, reversible work; ask only when
  missing information changes the result or an action exceeds that scope.
- Prefer the shortest clear implementation: ordinary functions, explicit inputs
  and results, and composition. Avoid speculative abstractions and unrelated
  rewrites; preserve existing staged and unstaged work.
- Follow the [architecture](docs/architecture.md): the host owns simulation state
  and time, plans own immutable wiring, and workspaces own mutable scratch.
  Keep numerical kernels deterministic and boundary validation explicit. Never
  expose partial outputs or stale caches as success after a numerical failure.
- Use `scripts/` and the pinned Nix/Enzyme environment. Builds require release
  mode and fat LTO. Do not update dependencies or toolchain pins as incidental
  setup. Use faer for general linear algebra; respect existing numeric lint rules.
- Validate changed behavior at its integration boundary. For numerical changes,
  use independent analytic answers, directional finite differences, adjoint
  identities, or perturb-and-resolve checks as appropriate. Exercise relevant
  failure recovery and workspace reuse. Avoid tests that mirror implementation.
- Run focused checks, then `./scripts/ci.sh` for code changes. For documentation
  changes, check links and affected examples; avoid unrelated test runs. Report
  what actually ran and any limitations. Performance and allocation claims need
  measurements of the affected path.
- Keep [README.md](README.md) a short splash page: purpose, one quick example,
  badges, source/doc links, and essential developer commands. Aim for roughly
  70 lines or fewer. Put API contracts in `docs/api.md`, execution details in
  `docs/advanced.md`, design in `docs/architecture.md`, and evidence and compiler
  limitations in `docs/validation.md`. Update the relevant section instead of
  appending feature inventories, roadmaps, or session narratives to the README.
- Communicate outcomes concisely: what changed, why, what was tested, and material
  limitations. Keep logs quiet and secrets out of files and output. Do not stage
  broadly, reset others' work, publish, or message others without authorization.
