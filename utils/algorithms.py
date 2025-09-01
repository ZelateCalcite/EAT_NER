def find_max_continuous_common_subarray(arr1, arr2):
    len1, len2 = len(arr1), len(arr2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    max_len = 0
    end_idx = -1

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if arr1[i - 1] == arr2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_idx = i - 1

    if max_len > 0:
        return arr1[end_idx - max_len + 1:end_idx + 1]
    else:
        return []


def find_sublist_indices(word_lst, seq):
    word_len = len(word_lst)
    indices = []

    for i in range(len(seq) - word_len + 1):
        if seq[i:i + word_len] == word_lst:
            indices.append(i)

    return indices
