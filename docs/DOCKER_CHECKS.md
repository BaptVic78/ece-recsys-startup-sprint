# Docker checks

Local checks on Windows with Docker Desktop, 26 November 2026.

| Check | Result |
| --- | --- |
| Python 3.12 and CPU PyTorch 2.15.0 | Installed; pip check passed |
| Application tests | 73 passed |
| Pipeline tests | 7 passed |
| GitHub workflow syntax | actionlint passed |
| Compose configuration | Valid |
| Start with the image already built | 12.3 seconds |
| Missing registry image, local build fallback | Passed in 13.8 seconds with build cache |
| Custom host ports | API 18010 and frontend 18501 worked |
| Real model recommendation | 86 ms over HTTP, strategy=model |
| Done, Skip and Missed | Saved through the API and checked in the browser |
| SQLite persistence | Passed after restart and container recreation |
| Next weekly plan | Returned 4 running days |
| Deliberately failing test | Test command returned exit code 1 in a disposable container |

The failing test was never added to the working tree. The unchanged app suite was
run again and passed. Local logs and the API check report are in `artifacts/`
(ignored by Git).

The image uses CPU wheels and skips pip bytecode compilation. Docker reports
1.77 GB of local image storage, above the suggested 1.5 GB target.
There is no dataset download at startup.

## Checks still needed on GitHub

- Push the branch and open the PR after the local review.
- Record a red then green Actions run on the PR; check the green run on main.
- After merge, check the GHCR publication, make the package public and test a pull
  without login.
- Ask TaDLpZ to time a clean-machine launch and add a screenshot to the PR.
  The local timings above use cached images and do not replace this Q6 check.

The branch starts from `feat/marathon-frontend`, as requested in issue #11.
