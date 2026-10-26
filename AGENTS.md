# Working on this stack

Read README.md and docs/SOURCES.md. `stack` links to scripts/stack.py; lib/ implements the behavior.
Keep all shell utilities in scripts/ and document everyday commands through `./stack`.
Run `./scripts/test.sh` after orchestration changes. Tests must not discover real hosts, download weights or launch inference.
Run the combined runtime host tests for runtime patches; GPU/model qualification is a separate explicit operation.

Keep exact upstream/model revisions in sources.lock.json and recipes/. Preserve model and KV precision.
Keep site identities, addresses and paths in ignored cluster.json; generated files belong in .stack/.
Downloads use Hugging Face's normal cache, not a second model folder. Never copy Hugging Face credentials to peers.
Preserve unrelated processes and edits. A failed or unreachable worker is unknown, not stopped.
Never remove a container without validating this stack's ownership label. Check both ranks before mutations.
Do not describe static checks or host tests as live model or maximum-context qualification.
