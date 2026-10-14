# Collaboration

The repository currently contains planning and setup materials only. Begin analytical implementation after the team agrees on the decisions in [the experiment plan](docs/EXPERIMENT_PLAN.md).

## Changes

1. Create a focused branch from `main`, such as `docs/data-plan` or `feature/shared-split`.
2. Link the work to a task or experiment issue.
3. Describe the change and its verification in a pull request.
4. Request a teammate's review before merging when practical.
5. Keep raw data, secrets, model weights, and large run artifacts out of commits.

Review requests are a team convention. No enforced branch protection or required checks are configured at this stage.

## Research records

Record the dataset version and checksum, cleaning decisions, saved split identifier, random seeds, model checkpoint, preprocessing, tuning settings, compute resources, software versions, and code commit for each experiment.

Fit learned preprocessing on training data only; keep test data out of tuning and model selection. Report failed or inconclusive experiments honestly. Do not mark targets as achieved until measured.

## Ownership

The initial responsibilities are listed in the README. Record actual contributions as work is completed, and update ownership when responsibilities change.

Repository access is managed by the owner. Collaborator usernames must be confirmed before invitations are sent.

## Licenses and assistance

Respect the source dataset's license and attribution. A repository software license has not yet been selected; the dataset license does not license the team's code.

Codex assisted with initial repository organization and planning-document formatting based on the team's existing proposal. This setup contains no analytical implementation or generated results. Maintain an accurate record of any later assistance.
