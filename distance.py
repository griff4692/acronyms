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
        target_idx = np.where(dist_copy[smallest_idx] == smallest_distance)[0][0]
        small_tokens_aligned[target_idx] = small_tokens[smallest_idx]
        dist_copy[:, target_idx] = float('inf')
        dist_copy[smallest_idx, :] = float('inf')

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


def jaccard_overlap(a, b):
    a = set(a)
    b = set(b)
    return (len(a.intersection(b)) / len(a.union(b)))


if __name__ == '__main__':
    a = ['Physical', 'therapy', '(field)']
    b = ['Physical', 'therapy']
    print(a, b)
    print(aligned_edit_distance(a, b))
    print(jaccard_overlap(a, b))

    a = ['cancerous']
    b = ['cancerous', 'lesion']
    print(a, b)
    print(aligned_edit_distance(a, b))
    print(jaccard_overlap(a, b))

    a = ['Electrocardiogram']
    b = ['Electrocardiography']
    print(a, b)
    print(aligned_edit_distance(a, b))
    print(jaccard_overlap(a, b))

    a = ['X-Ray', 'Computed', 'Tomography']
    b = ['Computed', 'Tomography', 'Service', '(procedure)']
    print(a, b)
    print(aligned_edit_distance(a, b))
    print(jaccard_overlap(a, b))

    a = ['Chronic', 'Obstructive', 'Airway', 'Disease']
    b = ['Chronic', 'Obstructive', 'Pulmonary', 'Disease', 'Of', 'Horses']
    print(a, b)
    print(aligned_edit_distance(a, b))
    print(jaccard_overlap(a, b))

    a = ['Atrial', 'Fibrillation']
    b = ['Atrial', 'Fibrillation', 'by', 'ECG', 'Finding']
    print(a, b)
    print(aligned_edit_distance(a, b))
    print(jaccard_overlap(a, b))

    a = ['Body', 'Weight', 'Domain']
    b = ['Body', 'Weight']
    print(a, b)
    print(aligned_edit_distance(a, b))
    print(jaccard_overlap(a, b))
