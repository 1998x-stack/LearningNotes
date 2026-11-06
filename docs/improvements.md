# Portfolio engineering improvement history

### 2026-11-06 19:14 UTC — ChatGPT Portfolio Iteration
- Run ID: portfolio-20261106T191409Z-dbf451e1
- Repo / Base SHA: 1998x-stack/LearningNotes / b073175bd6ce84653b5c5cb805db56295d739ed1
- Historical context: docs/improvements.md absent on main (GitHub 404); first initialization. No PRs returned by repository PR search. Latest baseline commit 2026-10-01.
- Product problem & expected user value: Learners running the LCS example received incorrect results and misleading 'substring' semantics; preserve the historical public function name while returning correct subsequences.
- Engineering diagnosis & evidence: algotithms/2025/01-06/longest_common_substring.py used [[0] * (n+1)] * (m+1), aliasing every DP row, and the mismatch recurrence referenced left and diagonal instead of left and above. Backtracking used the corrupted table.
- Options & decision: (1) LCS correctness P1 score 5+5+5+5-2-2=16; (2) insertion-sort stub P2 score 3+3+5+5-2-1=13; (3) empty fast_sort example P3 score 2+2+5+3-2-1=9. Fix LCS first, without renaming the legacy API.
- Changes: Correct independent DP rows, standard subsequence recurrence, backtracking and documented return semantics; add tests/test_lcs.py. No dependency or public API name changes.
- Validation: A: local Python 3 unittest with 3 groups passed on equivalent implementation before GitHub upload; C: GitHub file fetch confirms source changes. D: committed tests/test_lcs.py not yet executed from clean GitHub checkout; CI/browser validation unknown. Do not treat equivalent local test as proof of exact committed artifact.
- Security & regression review: No secrets, permissions, workflows, deployment or external dependencies modified. Existing API names preserved; tie-breaking retains previous upward-if-strictly-greater policy.
- Risk & rollback: Small educational Python example; rollback by reverting the feature commit(s). Exact committed test and branch protection checks required before merge.
- Follow-up: Run python -m unittest discover -s tests -v on clean checkout, inspect required checks and merge eligibility.
- Commit: pending
- PR: pending
