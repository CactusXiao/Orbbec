# Personnel statistics validation

Validated locally on 2026-09-22 against the `demo` working tree.

## Local checks

- `python -m unittest discover -s tests -p test_personnel.py`: 14 passed.
  Covers media-duration accounting, overlapping camera ranges, repeated QC jobs,
  no-error replacement, lease/ownership/assignment validation, operator handoff,
  missing timing, sparse frame IDs, registered instance history, pagination,
  HTTP endpoints, read-only episode details, and omission of account credentials.
- `python -m unittest discover -s tests -p test_label_frame_decisions.py`: 5 passed.
  Exercises the actual desktop confirmation method without opening a window,
  including original-result preservation, corrected-result replacement, backend
  failure, missing original results, and persisted local progress. On Windows
  this test isolates the import of the existing Linux-only process-lock module;
  it does not test desktop startup or single-instance locking.
- `python -m unittest discover -s tests -p test_browser_batch.py`: 23 passed.
  Includes authenticated frame attribution, revision stability after progress,
  idempotent final submission, invalid-frame rejection, and lease/session checks.
- `python -m unittest discover -s tests -p test_account_auth_smoke.py`: 1 passed.
- Python compilation and JavaScript syntax checks passed.
- Browser checks used an isolated temporary database with synthetic `demo.*`
  accounts. Verified account selection, QC and label tabs, task expansion,
  colored timelines, exact frame/duration details, and 390-pixel responsive layout
  with no horizontal overflow. No browser console errors were observed.

## Existing platform limitations

`test_workflow_smoke.py` passed 18 of 19 tests. The test
`test_manual_3d_failure_marks_whole_episode_failed_and_cleans_attempt_outputs`
also fails on the unchanged Git HEAD in this Windows environment (attempt output
directory cleanup assertion). It is not introduced by this change.

The full desktop UI test module cannot import the original
`frontend_runtime.py` on Windows because it unconditionally imports `fcntl`.
Full desktop startup still needs verification on its deployed Linux host.

## Real-environment verification

Requested target: the SSH configuration in `ssh/ssh1.txt`.
The initial connection to its configured SSH port timed out before authentication.
After the user connected the VPN, verification was retried at approximately
17:15 +08:00 on 2026-09-22. Windows routing selected `aTrustVNIC`, and TCP port 22
accepted connections. Both Windows OpenSSH 9.5 and Git OpenSSH 9.9 then failed
before receiving a remote SSH version banner, reporting
`kex_exchange_identification: Connection closed by remote host` (an earlier retry
reported a banner-exchange timeout). Authentication was never reached, so key
authorization and the deployment remain unverified. No target files, running
services, or production data have been changed. Local checks above do **not**
constitute real-environment verification.

After password credentials were supplied, SSH was retried with public-key
authentication disabled and password/keyboard-interactive authentication enabled.
The connection again closed before an authentication prompt, so no password was
transmitted. An HTTP request to the documented default backend port 8765 also
received no response before its timeout. These observations do not establish
whether the host's SSH service or VPN resource policy is responsible.

Once an accessible route or jump host is available:

1. Inspect the deployment checkout, Python environment, backend data roots,
   service launch configuration, and current processes.
2. Stage the changed code in an isolated checkout and run the targeted tests
   with the deployed Python environment. Use an independent test database for
   mutations and preserve the existing services.
3. Validate statistics against a read-only snapshot of representative real
   records, including historical operators, frame counts, capture durations,
   QC intervals, and completed manual-label jobs.
4. In the isolated instance, confirm a frame as no-error from each client, verify
   red QC / green annotation ranges and operator attribution, then reconfirm it
   as corrected and verify the false-positive total decreases.
5. Run the actual desktop client under the host's graphical environment and
   verify the browser client using real media available to that environment.

For rollout, restart the updated backend first so its additive frame-decision
table exists before starting the updated browser and desktop label clients.
