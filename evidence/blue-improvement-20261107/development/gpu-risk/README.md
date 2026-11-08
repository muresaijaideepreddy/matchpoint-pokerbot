# Variance-aware GPU experiment

Four models received the same 20,971,520 freshly sampled deals and 251,658,240 betting trajectories per model. The risk penalties optimize expected terminal chips minus a coefficient times their second moment. This per-hand objective is only an approximation to duplicate-match variance. An independently trained neutral model separates the objective change from a different training budget.

All five CPU bots faced the actual Blueprint program on the same twelve new 2,000-hand match seeds. Their model asset was the only runtime change relative to the preserved Weights 3 package.

| Model | Wins / draws / losses | Points / 12 | BB per duplicate deal | Average duplicate SD | Average match cutoff |
|---|---:|---:|---:|---:|---:|
| Saved Weights 3 | 3 / 9 / 0 | 7.5 | +0.601 | 16.401 | 1.017 |
| New neutral | 3 / 9 / 0 | 7.5 | +0.638 | 15.175 | 0.941 |
| Penalty .0005 | 3 / 9 / 0 | 7.5 | +0.570 | 13.843 | 0.858 |
| Penalty .001 | 1 / 11 / 0 | 6.5 | +0.370 | 12.527 | 0.776 |
| Penalty .003 | 0 / 11 / 1 | 5.5 | -0.201 | 7.260 | 0.450 |

The lightest penalty reduced the average observed payout standard deviation by 15.6% relative to saved Weights 3, but tied its official score. Larger penalties lost more chip edge than they recovered through lower variance. **Keep Weights 3.** No tested risk objective demonstrated a match-point improvement. All sixty development matches completed without referee faults; the strongest penalty's loss is a real poker result and is retained.

[Raw returns](dev-returns.jsonl.gz) · [Model hashes, training objective, and deal counts](training-manifest.json) · [Pinned opponents and seeds](dev-plan.json) · [Verified scores](verified.json)
