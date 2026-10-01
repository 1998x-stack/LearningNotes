def lcs(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1)] * (m+1)
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i][j-1], dp[i-1][j])
                
    return dp[m][n]

def lcs_return_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1)] * (m+1)
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i][j-1], dp[i-1][j])
    
    i, j = m, n
    common_substring = ""
    while i>0 and j>0:
        if s1[i-1] == s2[j-1]:
            common_substring = s1[i-1] + common_substring
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return common_substring


def test_lcs():
    s1 = "abcde"
    s2 = "ace"
    assert lcs(s1, s2) == 3
    print(lcs_return_substring(s1, s2))
    s1 = "abc"
    s2 = "abc"
    assert lcs(s1, s2) == 3
    print(lcs_return_substring(s1, s2))
    s1 = "abcdf"
    s2 = "zabc"
    assert lcs(s1, s2) == 3
    print(lcs_return_substring(s1, s2))
    print("Passed all tests!")


if __name__ == "__main__":
    test_lcs()