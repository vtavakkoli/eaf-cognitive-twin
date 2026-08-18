## Summary

Describe the change and why it is needed.

## Type of change

- [ ] Bug fix
- [ ] New feature or controller
- [ ] Digital-twin / physics change
- [ ] Safety or constraint change
- [ ] Benchmark / metric change
- [ ] Documentation or metadata
- [ ] Refactoring / maintenance

## Scientific impact

Does this change affect simulator behaviour, observations/actions, constraints, rewards/objectives, scenario generation, seeds, metrics, or benchmark comparability?

If yes, explain the scientific rationale and expected effect on previous results.

## Validation

- [ ] `python -m pip check`
- [ ] `python -m compileall -q src agents`
- [ ] `python -m unittest discover -s tests -v`
- [ ] Relevant simulation / benchmark smoke test
- [ ] Documentation updated where needed

### Experiment details, if applicable

- Model fidelity: A / B / C / N/A
- Configuration:
- Training seed(s):
- Evaluation seed(s):
- Scenario count:
- Commit or baseline compared against:

## Reproducibility

- [ ] Same evaluation scenarios/seeds were used for compared policies.
- [ ] Raw outputs were preserved when results are reported.
- [ ] Simulation-only evidence is not described as plant validation.
- [ ] No credentials, confidential plant data, or unauthorized data are included.

## Notes for reviewers

Call out any design trade-offs, limitations, expected numerical changes, or follow-up work.
