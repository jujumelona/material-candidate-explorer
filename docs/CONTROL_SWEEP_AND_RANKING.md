# Adaptive Control Sweep and Multi-Candidate Ranking

## Execution contract

For every retained parent candidate, the search runner may execute several alpha/temperature control variants. The variants are offsets from the base run configuration and move with the branch-local adaptive scheduler in later rounds. Every variant receives a distinct deterministic generator seed.

All generated children enter the same candidate pool. They are evaluated by the same expert panel and selected through independent stability, target-property, novelty, expert-disagreement, and Pareto branches. No incompatible scientific units are averaged.

## Failure handling

Every control attempt is recorded in `control_attempts`. A failed variant does not erase a parent when a sibling variant succeeds. A branch failure is recorded only when every control variant for that parent fails.

## Final output

`ranked_candidates` is the de-duplicated union of the final branches. It contains the candidate payload, source controls, branch ranks and scores, original per-expert properties and uncertainties, Pareto membership, and a diagnostic priority score. The priority score is weighted reciprocal-rank fusion and is not a physical property prediction.

## Generator capability boundary

The orchestration changes both alpha and temperature, but each generator must state which controls it actually applies. MatterGen v1 applies alpha through diffusion guidance and records temperature as unsupported. REINVENT sampling applies temperature directly. Unsupported controls must never be presented as effective model inputs.
