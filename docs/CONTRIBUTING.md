# Contributing to TractOracle-IRT
Contributions are welcome, just make sure you follow the guidelines below before submitting a pull request.

## Issues
We use GitHub issues to keep track of public bugs. Make sure the issue is as descriptive as possible and with steps to reproduce the said issue.

## Pull requests
If it's your first time contributing to an open source project, check [this guide](https://github.com/firstcontributions/first-contributions).

1. Fork this repository.
2. If you added code that requires testing, add them under the `tests` directory as described [below](#adding-tests).
3. If the "user-facing" scripts were modified (e.g. adding/remove an argument), update the related documentation. If you change anything related to the parameters of a function, make sure the docstring is modified accordingly.
4. Every PR should be reviewed and be able to pass tests.

### Adding tests
- The tests are written according to the pytest testing library.
- The `tests` directory should ideally follow the same structure as the `tractoracle_irt` directory. If new test files need to be created, they should be placed in an equivalent place in the directory structure.
- Each file that requires tests should have at least one corresponding file within the `tests` tree structure. So, creating a new file should also mean to create a corresponding test file as well.
- Tests should aim to cover most (ideally all) of the corner cases identified.
- Avoid global variables when possible and prefer the use of pytest's [fixtures](https://docs.pytest.org/en/6.2.x/fixture.html).
- Separate distinct tests in different functions. Avoid putting everything in one single function.
