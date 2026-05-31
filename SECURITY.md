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

- **Pickle loading.** The legacy notebook reads `data/backgrounds.pck` and `data/cards.pck` via `pickle.load`. Never load `.pck` files from untrusted sources. This is being migrated to `webdataset` shards; until that lands, treat pickle artifacts as code execution.
- **Runtime `wget`.** The legacy notebook downloads `dtd-r1.0.1.tar.gz` over plain HTTP. There is no checksum verification yet. Do not run the notebook on a shared/untrusted network until this is fixed.
- **External XML parsing.** `convert_voc_yolo.py` parses arbitrary VOC XML with the stdlib `ElementTree`. It is not hardened against XXE; only run it against XML you produced.

## Out of scope

- Vulnerabilities in upstream dependencies — please report those upstream (we will pick up fixes via Dependabot).
- Findings from tools without a working PoC against this codebase.
- Self-XSS or social-engineering issues.
