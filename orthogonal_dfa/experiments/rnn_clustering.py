from matplotlib import pyplot as plt
import numpy as np
import torch
import tqdm.auto as tqdm
import sklearn.cluster


def get_data(count, length, seed=0):
    rng = np.random.default_rng(seed)
    return torch.eye(4)[torch.tensor(rng.integers(4, size=(count, length)))].cuda()


def compute_dfa_states(dfa, x):
    with torch.no_grad():
        intermediates = dfa.intermediate_states(x)
        assert ((intermediates > 0.99) | (intermediates < 0.01)).all()
        states_each = intermediates.argmax(-1).cpu()
    removed = x.shape[1] - states_each.shape[-1]
    return states_each, removed


def compute_rnn_intermediates(rnn, x, removed):
    with torch.no_grad():
        h, _ = rnn(x)
        h = h.cpu()
    h = h[:, removed:]
    return h


def compute_correlations_flat(h):
    h_white = h - h.mean(dim=0)
    h_white = h_white / h_white.norm(2, dim=0)
    print(h_white.shape)
    cosine = h_white.T @ h_white
    cosine[np.arange(cosine.shape[0]), np.arange(cosine.shape[0])] = np.nan
    return cosine


def mean_cosine_by_state(h, states, subsample=10_000):
    h = h.reshape(-1, h.shape[-1]).T
    states_flat = states.flatten()
    if states_flat.shape[0] > subsample:
        indices = np.random.default_rng(0).choice(
            states_flat.shape[0], size=subsample, replace=False
        )
        h = h[:, indices]
        states_flat = states_flat[indices]
    cosine = compute_correlations_flat(h)
    num_states = 1 + states.max()
    matrix = np.zeros((num_states, num_states))
    for i, j in tqdm.tqdm(
        [(i, j) for i in range(num_states) for j in range(1 + i)],
    ):
        matrix[i, j] = matrix[j, i] = np.nanmean(
            cosine[states_flat == i][:, states_flat == j]
        )
    return matrix


def compute_clusters(h, states, num_clusters, subsample=10_000):
    h_flat = h.numpy().reshape(-1, h.shape[-1])
    states = states.flatten()
    num_states = states.max().item() + 1
    if h_flat.shape[0] > subsample:
        indices = np.random.default_rng(0).choice(
            h_flat.shape[0], size=subsample, replace=False
        )
        h_flat = h_flat[indices]
        states = states[indices]
    clusters = sklearn.cluster.KMeans(num_clusters).fit(h_flat).predict(h_flat)
    confusion = np.zeros((num_states, num_clusters), dtype=int)
    np.add.at(confusion, (states, clusters), 1)
    return confusion


def plot_mean_cosine_by_pair(mean_cosine_by_state_pair, kind_of_state):
    plt.xlabel("State 1")
    plt.ylabel("State 2")
    plt.title(
        f"Mean of the cosine similarity of RNN states, grouped by {kind_of_state} states"
    )
    _plot_matrix(mean_cosine_by_state_pair)
    plt.colorbar(label="Mean cosine similarity")
    plt.show()


def plot_clustering_results(cluster_confusion, kind_of_state):
    _plot_matrix(cluster_confusion / cluster_confusion.sum() * 100)
    plt.xlabel("Cluster")
    plt.ylabel("State")
    plt.title(f"Clustering of RNN states grouped by {kind_of_state} states")
    plt.colorbar(label="Proportion of overall frequency (%)")


def _plot_matrix(mean_cosine_by_state_pair):
    plt.imshow(mean_cosine_by_state_pair)
    maximal_by_either = np.zeros_like(mean_cosine_by_state_pair, dtype=bool)
    maximal_by_either[
        mean_cosine_by_state_pair.argmax(axis=0),
        np.arange(mean_cosine_by_state_pair.shape[0]),
    ] = True
    maximal_by_either[
        np.arange(mean_cosine_by_state_pair.shape[0]),
        mean_cosine_by_state_pair.argmax(axis=1),
    ] = True
    for i, j in zip(*np.where(maximal_by_either)):
        plt.text(
            j,
            i,
            f"{mean_cosine_by_state_pair[i, j]:.2f}",
            ha="center",
            va="center",
            color="white" if mean_cosine_by_state_pair[i, j] < 0.5 else "black",
        )
    plt.xticks(np.arange(len(mean_cosine_by_state_pair)))
    plt.yticks(np.arange(len(mean_cosine_by_state_pair)))
