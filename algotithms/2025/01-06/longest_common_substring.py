"""Longest common subsequence examples (legacy filename uses 'substring')."""


def _table(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def lcs(s1, s2):
    """Length of the longest common subsequence."""
    return _table(s1, s2)[len(s1)][len(s2)]


def lcs_return_substring(s1, s2):
    """Return a longest common subsequence; preserve historical API name."""
    dp = _table(s1, s2)
    i, j = len(s1), len(s2)
    result = []
    while i and j:
        if s1[i - 1] == s2[j - 1]:
            result.append(s1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(result))


def test_lcs():
    for a, b, expected in (("abcde", "ace", 3), ("abc", "abc", 3), ("abcdf", "zabc", 3)):
        assert lcs(a, b) == expected
        assert len(lcs_return_substring(a, b)) == expected


if __name__ == "__main__":
    test_lcs()
