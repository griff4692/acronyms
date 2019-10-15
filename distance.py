from nltk.corpus import stopwords
from nltk.metrics.distance import edit_distance
import numpy as np

STOPWORDS = set(stopwords.words('english'))

WORD_TRANSPOSE_COST = 0


def is_covered(a, b):
    remaining = set(a) - set(b)
    other_remaining = set(b) - set(a)
    return len(remaining) == 0 or len(other_remaining) == 0


def aligned_edit_distance(a, b):
    # First check if one string is fully covered by another --> 'physical therapy', 'physical therapy (a definition)'
    covered = is_covered(a, b)

    a = list(filter(lambda x: x not in STOPWORDS, a))
    b = list(filter(lambda x: x not in STOPWORDS, b))

    a_smaller = len(a) < len(b)
    small_tokens = a if a_smaller else b
    big_tokens = b if a_smaller else a

    slimmed_big = list(filter(lambda x: x in small_tokens, big_tokens))
    small_tokens_aligned = [''] * len(big_tokens)

    dist = np.zeros([len(small_tokens), len(big_tokens)])
    for ridx, rtok in enumerate(small_tokens):
        for cidx, ctok in enumerate(big_tokens):
            dist[ridx, cidx] = edit_distance(rtok, ctok, substitution_cost=1, transpositions=True)

    dist_copy = dist.copy()
    word_transpose_cost = 0
    for _ in range(len(small_tokens)):
        smallest_distances = np.min(dist_copy, 1)
        smallest_idx = np.argmin(smallest_distances)
        smallest_distance = smallest_distances[smallest_idx]
        target_idx = np.where(dist[smallest_idx] == smallest_distance)[0][0]
        small_tokens_aligned[target_idx] = small_tokens[smallest_idx]
        dist_copy[:, target_idx] = 99999999.9
        dist_copy[smallest_idx, :] = 99999999.9

        if big_tokens[target_idx] in slimmed_big:
            adjusted_target_idx = slimmed_big.index(big_tokens[target_idx])
            if not adjusted_target_idx == smallest_idx:
                word_transpose_cost += WORD_TRANSPOSE_COST
        else:
            if not target_idx == smallest_idx:
                word_transpose_cost += WORD_TRANSPOSE_COST

    total_norm_edit_distance = []
    for x, y in zip(big_tokens, small_tokens_aligned):
        norm_factor = float(max(len(x), len(y)))
        total_norm_edit_distance.append(edit_distance(x, y, substitution_cost=1, transpositions=True) / norm_factor)

    return np.array(total_norm_edit_distance).mean() + word_transpose_cost, covered


if __name__ == '__main__':
    # a = ['the', 'physical', 'therapy']
    # b = ['physical', 'therapy', 'location']
    # print(aligned_edit_distance(a, b))

    a = ['cancerous']
    b = ['cancerous', 'lesion']
    print(aligned_edit_distance(a, b))