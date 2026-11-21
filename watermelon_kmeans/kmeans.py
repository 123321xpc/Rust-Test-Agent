"""K-means experiments on the 30 samples in watermelon dataset 4.0.

Run: python kmeans.py
     python kmeans.py --plot  (requires matplotlib)
The clustering and the silhouette score use only the Python standard library.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


# Row order is the sample number shown in the question (1 through 30).
DATA = [
    (0.697, 0.460), (0.774, 0.376), (0.634, 0.264),
    (0.608, 0.318), (0.556, 0.215), (0.403, 0.237),
    (0.481, 0.149), (0.437, 0.211), (0.666, 0.091),
    (0.243, 0.267), (0.245, 0.057), (0.343, 0.099),
    (0.639, 0.161), (0.657, 0.198), (0.360, 0.370),
    (0.593, 0.042), (0.719, 0.103), (0.359, 0.188),
    (0.339, 0.241), (0.282, 0.257), (0.748, 0.232),
    (0.714, 0.346), (0.483, 0.312), (0.478, 0.437),
    (0.525, 0.369), (0.751, 0.489), (0.532, 0.472),
    (0.473, 0.376), (0.725, 0.445), (0.446, 0.459),
]

# For each k, the first k sample numbers form the initial centers.
INITIAL_SAMPLES = {
    "A (spread)": (1, 11, 17, 30),
    "B (right)": (1, 2, 3, 4),
    "C (left)": (10, 19, 20, 15),
}


def squared_distance(p, q):
    return sum((x - y) ** 2 for x, y in zip(p, q))


def kmeans(data, initial_sample_ids, max_iter=300):
    """Return labels, final centers, SSE and iterations for given 1-based IDs."""
    k = len(initial_sample_ids)
    if not 1 <= k <= len(data) or len(set(initial_sample_ids)) != k:
        raise ValueError("Initial sample IDs must be distinct and 1 <= k <= n")
    if any(i < 1 or i > len(data) for i in initial_sample_ids):
        raise ValueError("An initial sample ID is outside the data set")

    centers = [data[i - 1] for i in initial_sample_ids]
    previous_labels = None

    for iteration in range(1, max_iter + 1):
        # Assign every sample to its nearest center. Ties go to the lower index.
        labels = [
            min(range(k), key=lambda j: squared_distance(point, centers[j]))
            for point in data
        ]

        # If a cluster is empty, move the most poorly represented point from a
        # cluster of size > 1 into it before recomputing the means.
        counts = [labels.count(j) for j in range(k)]
        for empty in (j for j in range(k) if counts[j] == 0):
            point_id = max(
                (i for i in range(len(data)) if counts[labels[i]] > 1),
                key=lambda i: squared_distance(data[i], centers[labels[i]]),
            )
            counts[labels[point_id]] -= 1
            labels[point_id] = empty
            counts[empty] += 1

        new_centers = []
        for j in range(k):
            members = [point for point, label in zip(data, labels) if label == j]
            new_centers.append(tuple(
                sum(point[dimension] for point in members) / len(members)
                for dimension in range(len(data[0]))
            ))

        if labels == previous_labels:
            sse = sum(squared_distance(point, new_centers[label])
                      for point, label in zip(data, labels))
            return labels, new_centers, sse, iteration

        centers = new_centers
        previous_labels = labels

    raise RuntimeError("K-means did not converge within max_iter iterations")


def silhouette_score(data, labels):
    """Mean silhouette using Euclidean distance; singleton clusters score 0."""
    clusters = sorted(set(labels))
    scores = []
    for i, point in enumerate(data):
        same = [j for j, label in enumerate(labels) if label == labels[i] and j != i]
        if not same:
            scores.append(0.0)
            continue
        a = sum(math.sqrt(squared_distance(point, data[j])) for j in same) / len(same)
        b = min(
            sum(math.sqrt(squared_distance(point, data[j]))
                for j, label in enumerate(labels) if label == other)
            / labels.count(other)
            for other in clusters if other != labels[i]
        )
        scores.append((b - a) / max(a, b) if max(a, b) else 0.0)
    return sum(scores) / len(scores)


def plot_results(results, output):
    # Plotting is optional; the algorithm and printed results need no packages.
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True,
                             constrained_layout=True)
    colors = ("#2878b5", "#e07a26", "#2a9d75", "#bd4962")
    for row, k in enumerate((2, 3, 4)):
        for col, (name, sample_ids) in enumerate(INITIAL_SAMPLES.items()):
            labels, centers, sse, _, silhouette = results[(k, name)]
            ax = axes[row, col]
            for cluster in range(k):
                points = [p for p, label in zip(DATA, labels) if label == cluster]
                ax.scatter([p[0] for p in points], [p[1] for p in points],
                           s=35, color=colors[cluster], alpha=0.85)
            ax.scatter([p[0] for p in centers], [p[1] for p in centers],
                       marker="*", color="black", s=130, zorder=5)
            initial = [DATA[i - 1] for i in sample_ids[:k]]
            ax.scatter([p[0] for p in initial], [p[1] for p in initial],
                       marker="x", color="black", s=65, linewidths=2, zorder=6)
            ax.set_title(f"k={k}, {name}\nSSE={sse:.4f}, silhouette={silhouette:.3f}")
            ax.set_xlim(0.20, 0.82)
            ax.set_ylim(0.00, 0.54)
            ax.grid(alpha=0.2)
            if row == 2:
                ax.set_xlabel("Density")
            if col == 0:
                ax.set_ylabel("Sugar content")
    fig.legend(handles=[
        Line2D([], [], marker="x", color="black", linestyle="None", label="Initial center"),
        Line2D([], [], marker="*", color="black", linestyle="None", markersize=10,
               label="Final center"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.045), ncol=2)
    fig.suptitle("K-means on watermelon dataset 4.0")
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="save the 3 x 3 plot (needs matplotlib)")
    args = parser.parse_args()

    results = {}
    for k in (2, 3, 4):
        for name, sample_ids in INITIAL_SAMPLES.items():
            selected = sample_ids[:k]
            labels, centers, sse, iterations = kmeans(DATA, selected)
            silhouette = silhouette_score(DATA, labels)
            results[(k, name)] = (labels, centers, sse, iterations, silhouette)
            print(f"\nk={k}, {name}, initial sample IDs={list(selected)}")
            print(f"  SSE={sse:.6f}; silhouette={silhouette:.6f}; iterations={iterations}")
            for cluster, center in enumerate(centers):
                members = [i + 1 for i, label in enumerate(labels) if label == cluster]
                print(f"  C{cluster + 1}: center=({center[0]:.6f}, {center[1]:.6f}), "
                      f"sample IDs={members}")

    if args.plot:
        output = Path(__file__).with_name("watermelon_kmeans.png")
        plot_results(results, output)
        print(f"\nPlot saved to: {output}")


if __name__ == "__main__":
    main()
