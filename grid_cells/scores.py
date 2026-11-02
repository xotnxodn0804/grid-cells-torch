import math

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.stats

PAPER_OUTER_RADII_BINS = tuple(range(8, 21, 2))
DEFAULT_INNER_RADIUS_BINS = 4.0


def paper_mask_parameters(inner_radius_bins=DEFAULT_INNER_RADIUS_BINS,
                          outer_radii_bins=PAPER_OUTER_RADII_BINS):
    return [(float(inner_radius_bins), float(r)) for r in outer_radii_bins]


def deepmind_mask_parameters(nbins: int):
    # Google DeepMind released code uses 0.2*nbins inside and 10 outer
    # radii from 0.4*nbins to 1.0*nbins.
    return [(0.2 * nbins, float(x * nbins))
            for x in np.linspace(0.4, 1.0, num=10)]


def circle_mask(size, radius):
    sz = [math.floor(size[0] / 2), math.floor(size[1] / 2)]
    x = np.expand_dims(np.linspace(-sz[0], sz[1], size[1]), 0).repeat(size[0], 0)
    y = np.expand_dims(np.linspace(-sz[0], sz[1], size[1]), 1).repeat(size[1], 1)
    return np.sqrt(x ** 2 + y ** 2) <= radius


class GridScorer(object):
    def __init__(self, nbins, coords_range, mask_parameters):
        self.nbins = nbins
        self._coords_range = coords_range
        self._corr_angles = [30, 60, 90, 120, 150]
        self._masks = [(self._get_ring_mask(lo, hi), (lo, hi)) for lo, hi in mask_parameters]

    def calculate_ratemap(self, xs, ys, activations, statistic="mean"):
        return scipy.stats.binned_statistic_2d(
            xs, ys, activations, bins=self.nbins, statistic=statistic,
            range=self._coords_range)[0]

    def _get_ring_mask(self, mask_min, mask_max):
        n_points = [self.nbins * 2 - 1] * 2
        return (circle_mask(n_points, mask_max) & ~circle_mask(n_points, mask_min)).astype(float)

    def grid_score_60(self, corr):
        return np.minimum(corr[60], corr[120]) - np.maximum(
            corr[30], np.maximum(corr[90], corr[150]))

    def calculate_sac(self, rate_map):
        def filter2(b, x):
            return scipy.signal.convolve2d(x, np.rot90(b, 2), mode="full")

        seq = np.nan_to_num(rate_map)
        ones = np.ones(seq.shape)
        seq_sq = np.square(seq)

        seq1_x_seq2 = filter2(seq, seq)
        sum_seq1 = filter2(seq, ones)
        sum_seq2 = filter2(ones, seq)
        sum_seq1_sq = filter2(seq_sq, ones)
        sum_seq2_sq = filter2(ones, seq_sq)
        n_bins = filter2(ones, ones)
        n_bins_sq = np.square(n_bins)

        with np.errstate(invalid="ignore", divide="ignore"):
            std_seq1 = np.power(
                np.subtract(np.divide(sum_seq1_sq, n_bins),
                            np.divide(np.square(sum_seq1), n_bins_sq)), 0.5)
            std_seq2 = np.power(
                np.subtract(np.divide(sum_seq2_sq, n_bins),
                            np.divide(np.square(sum_seq2), n_bins_sq)), 0.5)
            covar = np.subtract(
                np.divide(seq1_x_seq2, n_bins),
                np.divide(np.multiply(sum_seq1, sum_seq2), n_bins_sq))
            x_coef = np.divide(covar, np.multiply(std_seq1, std_seq2))
        return np.nan_to_num(np.real(x_coef), posinf=0.0, neginf=0.0)

    def rotated_sacs(self, sac, angles):
        return [scipy.ndimage.rotate(sac, angle, reshape=False) for angle in angles]

    def get_grid_scores_for_mask(self, sac, rotated_sacs, mask):
        masked_sac = sac * mask
        ring_area = np.sum(mask)
        masked_sac_mean = np.sum(masked_sac) / ring_area
        masked_sac_centered = (masked_sac - masked_sac_mean) * mask
        variance = np.sum(masked_sac_centered ** 2) / ring_area + 1e-5
        corrs = {}
        for angle, rotated_sac in zip(self._corr_angles, rotated_sacs):
            masked_rotated_sac = (rotated_sac - masked_sac_mean) * mask
            corrs[angle] = np.sum(masked_sac_centered * masked_rotated_sac) / ring_area / variance
        return self.grid_score_60(corrs)

    def get_scores(self, rate_map):
        sac = self.calculate_sac(rate_map)
        rotated = self.rotated_sacs(sac, self._corr_angles)
        scores = np.asarray([self.get_grid_scores_for_mask(sac, rotated, mask)
                             for mask, _ in self._masks])
        best = int(np.argmax(scores))
        return scores[best], self._masks[best][1], sac


HD_BINS = 20
HD_THRESHOLD = 0.47


def directional_ratemap(headings: np.ndarray, activations: np.ndarray,
                        n_bins: int = HD_BINS) -> np.ndarray:
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    idx = np.clip(np.digitize(headings, edges) - 1, 0, n_bins - 1)
    onehot = np.zeros((len(idx), n_bins), dtype=np.float32)
    onehot[np.arange(len(idx)), idx] = 1.0
    counts = onehot.sum(axis=0)
    sums = onehot.T @ activations
    return (sums / np.maximum(counts, 1)[:, None]).T


def resultant_vector_length(tuning: np.ndarray, shift_to_nonnegative: bool = True
                            ) -> np.ndarray:
    tuning = np.asarray(tuning, dtype=float)
    if shift_to_nonnegative:
        tuning = tuning - tuning.min(axis=-1, keepdims=True)
    n = tuning.shape[-1]
    angles = -np.pi + (np.arange(n) + 0.5) * (2 * np.pi / n)
    total = tuning.sum(axis=-1)
    x = (tuning * np.cos(angles)).sum(axis=-1)
    y = (tuning * np.sin(angles)).sum(axis=-1)
    return np.where(total == 0, 0.0,
                    np.sqrt(x ** 2 + y ** 2) / np.where(total == 0, 1.0, total))


def sac_local_maxima(sac: np.ndarray, within_radius: float | None = None) -> np.ndarray:
    footprint = np.ones((3, 3), dtype=bool)
    footprint[1, 1] = False
    neighbour_max = scipy.ndimage.maximum_filter(sac, footprint=footprint,
                                                 mode="constant", cval=-np.inf)
    peaks = np.argwhere(sac > neighbour_max)
    if within_radius is not None and len(peaks):
        centre = (np.asarray(sac.shape) - 1) / 2.0
        keep = np.linalg.norm(peaks - centre, axis=1) <= within_radius
        peaks = peaks[keep]
    return peaks


def grid_scale(sac: np.ndarray, bin_size_m: float, n_peaks: int = 6,
               search_radius_bins: float = PAPER_OUTER_RADII_BINS[-1]) -> float:
    centre = (np.asarray(sac.shape) - 1) / 2.0
    peaks = sac_local_maxima(sac, within_radius=search_radius_bins)
    if len(peaks) == 0:
        return float("nan")

    distances = np.linalg.norm(peaks - centre, axis=1)
    distances = distances[distances > 0]
    if len(distances) < n_peaks:
        return float("nan")
    return float(np.median(np.sort(distances)[:n_peaks]) * bin_size_m)


BORDER_THRESHOLD = 0.50
BORDER_BINS = 20
BORDER_WALL_BINS = 3


def border_score(ratemap: np.ndarray, d_b: int = BORDER_WALL_BINS,
                 shift_to_nonnegative: bool = True) -> float:
    rm = np.asarray(ratemap, dtype=float)
    if shift_to_nonnegative:
        rm = rm - np.nanmin(rm)

    n = rm.shape[0]
    assert d_b * 2 < n, f"d_b={d_b} leaves no interior in a {n}x{n} ratemap"
    walls = (rm[:d_b, :], rm[-d_b:, :], rm[:, :d_b], rm[:, -d_b:])
    c = np.nanmean(rm[d_b:n - d_b, d_b:n - d_b])

    scores = []
    for wall in walls:
        b = np.nanmean(wall)
        denom = b + c
        scores.append(0.0 if denom == 0 else (b - c) / denom)
    return float(np.max(scores))


DISCRETENESS_BINS = 13


def discreteness(scales: np.ndarray, n_bins: int = DISCRETENESS_BINS,
                 value_range: tuple[float, float] | None = None) -> float:
    scales = np.asarray(scales, dtype=float)
    scales = scales[np.isfinite(scales)]
    if len(scales) == 0:
        return float("nan")
    counts, _ = np.histogram(scales, bins=n_bins, range=value_range)
    return float(np.std(counts))


def discreteness_significance(scales: np.ndarray, n_shuffles: int = 500,
                              n_bins: int = DISCRETENESS_BINS, seed: int = 0
                              ) -> tuple[float, np.ndarray, float]:
    scales = np.asarray(scales, dtype=float)
    scales = scales[np.isfinite(scales)]
    if len(scales) == 0:
        return float("nan"), np.array([]), float("nan")

    half = scales.min() / 2.0
    observed = discreteness(scales, n_bins=n_bins)

    rng = np.random.default_rng(seed)
    null = np.array([
        discreteness(scales + rng.uniform(-half, half, size=len(scales)),
                     n_bins=n_bins)
        for _ in range(n_shuffles)])
    p = float((null >= observed).sum() + 1) / (n_shuffles + 1)
    return observed, null, p


def cluster_scales(scales: np.ndarray, max_components: int = 8, seed: int = 0):
    from sklearn.mixture import GaussianMixture

    scales = np.asarray(scales, dtype=float)
    scales = scales[np.isfinite(scales)].reshape(-1, 1)
    if len(scales) < max_components:
        return 0, np.array([]), np.array([]), np.array([])

    bics, models = [], []
    for k in range(1, max_components + 1):
        gm = GaussianMixture(n_components=k, random_state=seed, n_init=5).fit(scales)
        bics.append(gm.bic(scales))
        models.append(gm)

    best = int(np.argmin(bics))
    means = np.sort(models[best].means_.ravel())
    ratios = means[1:] / means[:-1] if len(means) > 1 else np.array([])
    return best + 1, means, np.array(bics), ratios


def ratemap_stability(ratemap_a: np.ndarray, ratemap_b: np.ndarray) -> float:
    a = np.asarray(ratemap_a, dtype=float).ravel()
    b = np.asarray(ratemap_b, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return float("nan")
    a, b = a[valid], b[valid]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def score_units(scorer: GridScorer, xy: np.ndarray, activations: np.ndarray):
    ratemaps = [scorer.calculate_ratemap(xy[:, 0], xy[:, 1], activations[:, i])
                for i in range(activations.shape[1])]
    results = [scorer.get_scores(rm) for rm in ratemaps]
    return (np.array([r[0] for r in results]), ratemaps,
            [r[2] for r in results], [r[1] for r in results])
