"""Tests for Dataset location sampling.

The contract every branch of Dataset._random_location must satisfy: locations
are drawn uniformly from `nonzero(mask) & valid`. These tests pin that down for
the bounding-box rejection path, the index-array path, and the fallback between
them.

Run: python test/test_locs.py
"""
import unittest

import numpy as np

from dataprovider3 import Dataset
from dataprovider3.geometry import Vec3d


SEED = 20260827


def make_dataset(mask, fov, offset=(0, 0, 0)):
    img = np.zeros(mask.shape, dtype="float32")
    ds = Dataset(tag="t")
    ds.add_data("input", img, offset=offset)
    ds.add_mask("m", mask, offset=offset, loc=True)
    spec = {"input": (1,) + (fov,) * 3, "m": (1,) + (fov,) * 3}
    ds.set_spec(spec)
    return ds, spec


def expected_support(ds, mask, spec, offset=(0, 0, 0)):
    """Every location the sampler is allowed to return: nonzero(mask) & valid."""
    valid = ds.valid_range(spec)
    out = set()
    for p in zip(*np.nonzero(mask)):
        g = tuple(Vec3d(p) + Vec3d(offset))
        if valid.contains(Vec3d(g)):
            out.add(g)
    return out


def draw(ds, spec, n, seed=SEED):
    np.random.seed(seed)
    return [tuple(ds._random_location(spec)) for _ in range(n)]


def box_mask(shape, lo, hi):
    m = np.zeros(shape, dtype="uint8")
    m[lo:hi, lo:hi, lo:hi] = 1
    return m


class TestRepresentation(unittest.TestCase):
    def test_solid_box_uses_rejection(self):
        mask = box_mask((32,) * 3, 6, 26)
        ds, _ = make_dataset(mask, 8)
        self.assertIsNone(ds.locs["data"], "solid box should not build an index array")
        self.assertIsNotNone(ds.locs["box"])
        self.assertEqual(ds.locs["count"], 20 ** 3)
        self.assertEqual(tuple(ds.locs["box"].min()), (6, 6, 6))
        self.assertEqual(tuple(ds.locs["box"].max()), (26, 26, 26))

    def test_num_samples_is_nonzero_count(self):
        mask = box_mask((32,) * 3, 6, 26)
        ds, _ = make_dataset(mask, 8)
        self.assertEqual(ds.num_samples(), int(np.count_nonzero(mask)))

    def test_sparse_mask_falls_back_to_index_array(self):
        rng = np.random.RandomState(0)
        mask = np.zeros((32,) * 3, dtype="uint8")
        # ~1% fill spread over the whole volume -> below REJECT_MIN_FILL
        flat = rng.choice(mask.size, size=mask.size // 100, replace=False)
        mask.ravel()[flat] = 1
        ds, _ = make_dataset(mask, 8)
        self.assertIsNotNone(ds.locs["data"], "sparse mask should use the index array")
        self.assertIsNone(ds.locs["box"])
        self.assertEqual(ds.locs["count"], int(np.count_nonzero(mask)))

    def test_empty_mask_has_no_locations(self):
        mask = np.zeros((32,) * 3, dtype="uint8")
        ds, spec = make_dataset(mask, 8)
        self.assertEqual(ds.locs["count"], 0)
        with self.assertRaises(Dataset.OutOfRangeError):
            ds._random_location(spec)


class TestDistribution(unittest.TestCase):
    """The load-bearing property: same support, uniform, both representations."""

    def _check_uniform(self, samples, support, sigma=5.0):
        counts = {}
        for s in samples:
            self.assertIn(s, support, f"drew {s} outside the allowed support")
            counts[s] = counts.get(s, 0) + 1
        n, k = len(samples), len(support)
        exp = n / k
        sd = (n * (1 / k) * (1 - 1 / k)) ** 0.5
        worst = max(abs(counts.get(p, 0) - exp) for p in support)
        self.assertLess(worst / sd, sigma,
                        f"non-uniform: worst cell off by {worst/sd:.1f} sigma")
        return counts

    def _compare_paths(self, mask, fov, n=120000, offset=(0, 0, 0)):
        ds_a, spec = make_dataset(mask, fov, offset)
        self.assertIsNone(ds_a.locs["data"], "path A must be the rejection path")
        support = expected_support(ds_a, mask, spec, offset)
        a = draw(ds_a, spec, n)

        # Same mask, forced onto the historical index-array path.
        ds_b, spec_b = make_dataset(mask, fov, offset)
        ds_b._materialize_locs()
        self.assertIsNotNone(ds_b.locs["data"])
        b = draw(ds_b, spec_b, n, seed=SEED + 1)

        ca = self._check_uniform(a, support)
        cb = self._check_uniform(b, support)
        self.assertEqual(set(ca), set(cb), "the two paths cover different supports")
        return support

    def test_solid_box_matches_index_array(self):
        mask = box_mask((28,) * 3, 6, 22)
        support = self._compare_paths(mask, 8)
        self.assertEqual(len(support), 16 ** 3)

    def test_box_with_holes_matches_index_array(self):
        """Dense but not a box: rejection must still never return a hole."""
        mask = box_mask((28,) * 3, 6, 22)
        mask[10:13, 10:13, 10:13] = 0          # carve a hole, fill stays ~0.96
        ds, _ = make_dataset(mask, 8)
        self.assertIsNone(ds.locs["data"], "should still use rejection")
        support = self._compare_paths(mask, 8)
        for p in [(11, 11, 11), (12, 12, 12)]:
            self.assertNotIn(p, support)

    def test_offset_is_honored(self):
        mask = box_mask((28,) * 3, 6, 22)
        self._compare_paths(mask, 8, n=60000, offset=(100, 200, 300))

    def test_mask_clipped_by_valid_range(self):
        """Nonzero region sticking out past valid must be clipped, not sampled."""
        mask = np.zeros((28,) * 3, dtype="uint8")
        mask[0:22, 0:22, 0:22] = 1             # reaches the array border
        ds, spec = make_dataset(mask, 8)
        self.assertIsNone(ds.locs["data"])
        support = expected_support(ds, mask, spec)
        valid = ds.valid_range(spec)
        self.assertEqual(tuple(valid.min()), (4, 4, 4))
        for s in draw(ds, spec, 20000):
            self.assertIn(s, support)


class TestUnion(unittest.TestCase):
    def test_two_masks_union_exactly(self):
        a = box_mask((28,) * 3, 4, 16)
        b = box_mask((28,) * 3, 12, 24)
        img = np.zeros(a.shape, dtype="float32")
        ds = Dataset(tag="t")
        ds.add_data("input", img)
        ds.add_mask("a", a, loc=True)
        ds.add_mask("b", b, loc=True)
        self.assertIsNotNone(ds.locs["data"], "a second mask must materialize")
        expected = np.union1d(np.flatnonzero(a), np.flatnonzero(b))
        np.testing.assert_array_equal(ds.locs["data"], expected)
        self.assertEqual(ds.locs["count"], expected.size)
        self.assertEqual(ds.locs["keys"], ["a", "b"])

    def test_mismatched_shape_is_rejected(self):
        img = np.zeros((28,) * 3, dtype="float32")
        ds = Dataset(tag="t")
        ds.add_data("input", img)
        ds.add_mask("a", box_mask((28,) * 3, 4, 16), loc=True)
        with self.assertRaises(AssertionError):
            ds.add_mask("b", box_mask((20,) * 3, 4, 16), loc=True)


class TestTermination(unittest.TestCase):
    def test_disjoint_mask_raises_instead_of_hanging(self):
        """Previously an unbounded `while True`."""
        mask = np.zeros((40,) * 3, dtype="uint8")
        mask[0:3, 0:3, 0:3] = 1                # entirely inside the fov margin
        ds, spec = make_dataset(mask, 16)
        valid = ds.valid_range(spec)
        self.assertFalse(valid.contains(Vec3d(2, 2, 2)))
        with self.assertRaises(Dataset.OutOfRangeError):
            ds._random_location(spec)

    def test_index_path_is_bounded(self):
        mask = np.zeros((40,) * 3, dtype="uint8")
        mask[0:3, 0:3, 0:3] = 1
        ds, spec = make_dataset(mask, 16)
        ds._materialize_locs()
        with self.assertRaises(Dataset.OutOfRangeError):
            ds._random_location(spec)


class TestNoLocMask(unittest.TestCase):
    def test_uniform_over_valid_when_no_loc_mask(self):
        img = np.zeros((24,) * 3, dtype="float32")
        ds = Dataset(tag="t")
        ds.add_data("input", img)
        spec = {"input": (1, 8, 8, 8)}
        ds.set_spec(spec)
        self.assertIsNone(ds.locs)
        valid = ds.valid_range(spec)
        for s in draw(ds, spec, 5000):
            self.assertTrue(valid.contains(Vec3d(s)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
