"""Spectral and Agglomerative Clustering for Speaker Diarization."""

from __future__ import annotations
import math
from typing import List, Optional, Tuple
import numpy as np
import torch


def cosine_affinity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Computes symmetric cosine affinity matrix scaled to [0, 1].
    embeddings: [N, D] L2-normalized
    """
    # Dot product of L2-normalized vectors is cosine similarity in [-1, 1]
    cos_sim = np.dot(embeddings, embeddings.T)
    # Rescale to positive affinity [0, 1]
    affinity = np.clip(0.5 * (1.0 + cos_sim), 0.0, 1.0)
    np.fill_diagonal(affinity, 1.0)
    return affinity


def simple_kmeans(
    data: np.ndarray,
    k: int,
    max_iters: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """K-Means clustering with k-means++ initialization on normalized vectors."""
    n_samples, d = data.shape
    if n_samples <= k:
        labels = np.arange(n_samples)
        centroids = data.copy()
        return labels, centroids

    rng = np.random.RandomState(seed)
    # K-means++ initialization
    centroids = np.zeros((k, d), dtype=np.float32)
    centroids[0] = data[rng.choice(n_samples)]

    for c_idx in range(1, k):
        # Min squared distance to existing centroids
        dists = np.min([np.sum((data - c) ** 2, axis=1) for c in centroids[:c_idx]], axis=0)
        probs = dists / (np.sum(dists) + 1e-9)
        centroids[c_idx] = data[rng.choice(n_samples, p=probs)]

    labels = np.zeros(n_samples, dtype=np.int32)
    for _ in range(max_iters):
        # Assign closest centroid
        distances = np.linalg.norm(data[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # Recompute centroids
        for c_idx in range(k):
            members = data[labels == c_idx]
            if len(members) > 0:
                mean = members.mean(axis=0)
                norm = np.linalg.norm(mean) + 1e-9
                centroids[c_idx] = mean / norm

    return labels, centroids


class SpectralSpeakerClusterer:
    """Normalized Spectral Clustering for grouping speaker embeddings."""

    def __init__(
        self,
        min_speakers: int = 1,
        max_speakers: int = 6,
        seed: int = 42,
    ) -> None:
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.seed = seed

    def cluster(self, embeddings: np.ndarray, num_speakers: Optional[int] = None) -> np.ndarray:
        """Clusters speaker embeddings into discrete speaker IDs.

        Args:
            embeddings: [N, D] array of speaker embeddings
            num_speakers: Optional exact number of speakers. If None, uses eigengap heuristic.
        """
        n_samples = len(embeddings)
        if n_samples == 0:
            return np.array([], dtype=np.int32)
        if n_samples == 1:
            return np.array([0], dtype=np.int32)

        # Scalable landmark acceleration for long meetings (large window counts)
        if n_samples > 600:
            stride = int(math.ceil(n_samples / 500))
            sub_indices = np.arange(0, n_samples, stride)
            sub_embs = embeddings[sub_indices]
            sub_labels = self.cluster(sub_embs, num_speakers=num_speakers)

            k = int(np.max(sub_labels)) + 1
            centroids = np.zeros((k, embeddings.shape[1]), dtype=np.float32)
            for cid in range(k):
                mask = sub_labels == cid
                if np.any(mask):
                    centroids[cid] = sub_embs[mask].mean(axis=0)
                    centroids[cid] /= (np.linalg.norm(centroids[cid]) + 1e-9)

            sims = np.dot(embeddings, centroids.T)
            return np.argmax(sims, axis=1).astype(np.int32)

        # 1. Cosine Affinity Matrix
        affinity = cosine_affinity_matrix(embeddings)

        # 2. Normalized Laplacian L = D^(-1/2) * A * D^(-1/2)
        degree = np.sum(affinity, axis=1)
        deg_inv_sqrt = np.power(np.maximum(degree, 1e-9), -0.5)
        laplacian_norm = deg_inv_sqrt[:, None] * affinity * deg_inv_sqrt[None, :]

        # 3. Eigen-decomposition (symmetric matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_norm)
        # Sort in descending order of eigenvalues
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 4. Determine number of speakers (Eigengap heuristic)
        if num_speakers is None:
            max_k = min(self.max_speakers, n_samples - 1)
            # Evaluate single speaker monologue vs multi-speaker dialogue
            is_single_speaker = (
                self.min_speakers == 1
                and (eigenvalues[1] < 0.035 or np.min(affinity) > 0.88)
            )

            if is_single_speaker or max_k < 2:
                k = 1
            else:
                # Multi-speaker meeting: search k >= 2 maximizing eigengap (lambda_k - lambda_{k+1})
                start_k = max(2, self.min_speakers)
                best_k = start_k
                best_gap = -1.0
                for candidate_k in range(start_k, max_k + 1):
                    gap = float(eigenvalues[candidate_k - 1] - eigenvalues[candidate_k])
                    if gap > best_gap:
                        best_gap = gap
                        best_k = candidate_k
                k = best_k
        else:
            k = max(1, min(num_speakers, n_samples))

        if k == 1:
            return np.zeros(n_samples, dtype=np.int32)

        # 5. Extract top k eigenvectors and normalize rows
        u = eigenvectors[:, :k]
        row_norms = np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
        u_norm = u / row_norms

        # 6. Cluster in spectral subspace
        labels, _ = simple_kmeans(u_norm, k=k, seed=self.seed)
        return labels


class AgglomerativeSpeakerClusterer:
    """Agglomerative Hierarchical Clustering with average linkage and cosine threshold."""

    def __init__(self, threshold: float = 0.65) -> None:
        self.threshold = threshold

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        n_samples = len(embeddings)
        if n_samples <= 1:
            return np.zeros(n_samples, dtype=np.int32)

        clusters = [[i] for i in range(n_samples)]

        while len(clusters) > 1:
            best_sim = -1.0
            best_pair = (-1, -1)

            # Pairwise cluster distance
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    embs_i = embeddings[clusters[i]]
                    embs_j = embeddings[clusters[j]]
                    sim = float(np.mean(np.dot(embs_i, embs_j.T)))

                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (i, j)

            if best_sim < self.threshold or best_pair == (-1, -1):
                break

            i, j = best_pair
            clusters[i].extend(clusters[j])
            clusters.pop(j)

        labels = np.zeros(n_samples, dtype=np.int32)
        for cluster_id, indices in enumerate(clusters):
            for idx in indices:
                labels[idx] = cluster_id

        return labels
