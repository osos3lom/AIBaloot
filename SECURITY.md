# Security Policy

## Supported versions

This project is pre-1.0. Only the `master` branch receives security fixes.

| Version | Supported |
|---------|-----------|
| `master` | Yes |
| anything else | No |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Email: `osama.aalam@gmail.com` with the subject line `SECURITY: hakim-vision`.

Include:

- A clear description of the issue.
- Steps to reproduce, or a proof-of-concept.
- The affected commit hash (or `master` if unsure).
- Your assessment of impact (e.g., RCE, credential leak, DoS).

You should expect:

- Acknowledgement within **72 hours**.
- A triage decision within **7 days**.
- A fix or mitigation timeline within **14 days** for confirmed issues.

We will credit reporters in the release notes unless they ask otherwise.

## Known sensitive areas

- **Tar shard extraction.** `hakim_vision.synthetic.assets` reads `cards-*.tar` and `backgrounds-*.tar` with the stdlib `tarfile`. Open these only on shards you produced or trust. Do not feed shards from untrusted sources without first inspecting member paths for traversal (`../`).
- **OpenCV image decoding.** `cv2.imdecode` is used to decode card and background images from shard payloads. Malformed images may trigger CVEs in the bundled OpenCV; keep the `opencv-python` pin current.
- **Asset extras (development).** The `notebooks` optional-deps group pulls `jupyter`/`matplotlib` for the tester smoke-test notebook. Do not install in CI containers that don't need them.

## Out of scope

- Vulnerabilities in upstream dependencies — please report those upstream (we will pick up fixes via Dependabot).
- Findings from tools without a working PoC against this codebase.
- Self-XSS or social-engineering issues.
