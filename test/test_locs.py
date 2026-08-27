"""Tests for Dataset location sampling.

The contract every branch of Dataset._random_location must satisfy: locations
are drawn uniformly from `nonzero(mask) & valid`. These tests pin that down for
the bounding-box rejection path, the index-array path, and the fallback between
them.

Run: python test/test_locs.py
"""
import contextlib
import unittest

import numpy as np

from dataprovider3 import Dataset
from dataprovider3.geometry import Vec3d


SEED = 20260827


@contextlib.contextmanager
def small_scan_chunk(n=4096):
    """Shrink the scan chunk so multi-chunk behaviour is reachable in a test.

    The shipped chunk is 262,144 indices; a mask big enough to span two of them
    is not worth allocating just to check that the loop advances.
    """
    saved = Dataset.LOC_SCAN_CHUNK
    Dataset.LOC_SCAN_CHUNK = n
    try:
        yield
    finally:
        Dataset.LOC_SCAN_CHUNK = saved


@contextlib.contextmanager
def force_box():
    """Make Dataset prefer the bounding box regardless of mask size.

    Real masks reach the box path by being too big for an index array; test
    masks are small on purpose, so lower the bar instead of allocating 64 MiB
    of them.
    """
    saved = Dataset.LOC_INDEX_MAX_BYTES
    Dataset.LOC_INDEX_MAX_BYTES = 0
    try:
        yield
    finally:
        Dataset.LOC_INDEX_MAX_BYTES = saved


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
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        return [tuple(ds._random_location(spec)) for _ in range(n)]
    finally:
        np.random.set_state(state)


def box_mask(shape, lo, hi):
    m = np.zeros(shape, dtype="uint8")
    m[lo:hi, lo:hi, lo:hi] = 1
    return m


class TestRepresentation(unittest.TestCase):
    def test_solid_box_uses_rejection(self):
        mask = box_mask((32,) * 3, 6, 26)
        with force_box():
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

    def test_small_mask_prefers_the_index_array(self):
        """Below LOC_INDEX_MAX_BYTES the array costs nothing; prefer exactness."""
        mask = box_mask((32,) * 3, 6, 26)
        ds, _ = make_dataset(mask, 8)   # no force_box
        self.assertIsNotNone(ds.locs["data"])
        self.assertIsNone(ds.locs["box"])
        self.assertEqual(ds.locs["count"], 20 ** 3)

    def test_real_policy_picks_the_box_for_a_large_mask(self):
        """The shipped threshold, not force_box(), must reach the box path."""
        n = 210                       # 210^3 = 9.26M nonzeros > 64 MiB / 8
        mask = np.ones((n,) * 3, dtype="uint8")
        img = np.zeros(mask.shape, dtype="float32")
        ds = Dataset(tag="t")
        ds.add_data("input", img)
        ds.add_mask("m", mask, loc=True)
        spec = {"input": (1,) + (8,) * 3, "m": (1,) + (8,) * 3}
        ds.set_spec(spec)

        self.assertGreater(ds.locs["count"] * 8, Dataset.LOC_INDEX_MAX_BYTES)
        self.assertIsNone(ds.locs["data"], "large mask must not build an index array")
        self.assertIsNotNone(ds.locs["box"])
        valid = ds.valid_range(spec)
        for loc in draw(ds, spec, 2000):
            self.assertTrue(valid.contains(Vec3d(loc)))

    def test_rejection_converges_on_a_sparse_mask(self):
        """Low fill costs draws, not correctness."""
        rng = np.random.RandomState(0)
        mask = np.zeros((32,) * 3, dtype="uint8")
        flat = rng.choice(mask.size, size=mask.size // 100, replace=False)
        mask.ravel()[flat] = 1
        with force_box():
            ds, spec = make_dataset(mask, 8)
        self.assertIsNone(ds.locs["data"])
        support = expected_support(ds, mask, spec)
        for loc in draw(ds, spec, 3000):
            self.assertIn(loc, support)

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
        with force_box():
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
        mask[10:13, 10:13, 10:13] = 0          # carve a hole; fill stays 0.993
        with force_box():
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
        with force_box():
            ds, spec = make_dataset(mask, 8)
        self.assertIsNone(ds.locs["data"])
        support = expected_support(ds, mask, spec)
        valid = ds.valid_range(spec)
        self.assertEqual(tuple(valid.min()), (4, 4, 4))
        for s in draw(ds, spec, 20000):
            self.assertIn(s, support)


class TestRareSupport(unittest.TestCase):
    """Locations that are real but hard to hit must still be served.

    The loop this code replaced was unbounded: it always eventually found a
    location when one existed. A fixed draw budget alone would turn that into a
    spurious OutOfRangeError, so the index path resolves the tail exactly.
    """

    def _rare_mask(self, n=64, margin_slabs=2):
        # Almost all the mass sits in the fov margin; a handful of voxels are
        # reachable. Ratio here is ~8 / 16k.
        mask = np.zeros((n,) * 3, dtype="uint8")
        mask[0:margin_slabs, :, :] = 1
        c = n // 2
        mask[c:c + 2, c:c + 2, c:c + 2] = 1
        return mask

    def test_rare_support_is_found_not_raised(self):
        mask = self._rare_mask()
        ds, spec = make_dataset(mask, 9)
        self.assertIsNotNone(ds.locs["data"])
        support = expected_support(ds, mask, spec)
        self.assertTrue(0 < len(support) <= 64, f"support {len(support)}")

        # far more calls than LOC_RETRY_LIMIT makes likely to succeed by luck
        for _ in range(300):
            loc = tuple(ds._random_location(spec))
            self.assertIn(loc, support)

    def test_scan_fallback_is_uniform(self):
        mask = self._rare_mask()
        ds, spec = make_dataset(mask, 9)
        valid = ds.valid_range(spec)
        support = expected_support(ds, mask, spec)
        counts = {}
        for _ in range(4000):
            loc = tuple(ds._scan_location(valid))
            self.assertIn(loc, support)
            counts[loc] = counts.get(loc, 0) + 1
        self.assertEqual(set(counts), support, "scan must reach every location")
        exp = 4000 / len(support)
        sd = (4000 * (1 / len(support)) * (1 - 1 / len(support))) ** 0.5
        worst = max(abs(c - exp) for c in counts.values())
        self.assertLess(worst / sd, 5.0)


class TestBoxPathRareSupport(unittest.TestCase):
    """The box path must not raise where the index path would have served.

    A mask can clear the byte budget on nonzero voxels that all sit in the fov
    margin, so `count` says "dense" while acceptance over `box & valid` is
    near zero. Rejection then exhausts its budget on a mask that does have
    reachable locations.
    """

    def _margin_heavy(self, core=8):
        # z-margin slabs carry the mass; a small core is the only reachable part
        mask = np.zeros((128, 512, 512), dtype="uint8")
        mask[0:20] = 1
        mask[108:] = 1
        c = 64, 256, 256
        h = core // 2
        mask[c[0]-h:c[0]+h, c[1]-h:c[1]+h, c[2]-h:c[2]+h] = 1
        return mask

    def test_margin_heavy_mask_is_served_not_raised(self):
        mask = self._margin_heavy()
        img = np.zeros(mask.shape, dtype="float32")
        ds = Dataset(tag="t")
        ds.add_data("input", img)
        ds.add_mask("m", mask, loc=True)
        fov = (41, 9, 9)
        spec = {"input": (1,) + fov, "m": (1,) + fov}
        ds.set_spec(spec)

        # precondition: over the budget, so the box path is chosen
        self.assertGreater(ds.locs["count"] * 8, Dataset.LOC_INDEX_MAX_BYTES)
        self.assertIsNotNone(ds.locs["box"])

        valid = ds.valid_range(spec)
        raised, seen = 0, set()
        for _ in range(40):
            try:
                loc = tuple(ds._random_location(spec))
            except Dataset.OutOfRangeError:
                raised += 1
                continue
            self.assertTrue(valid.contains(Vec3d(loc)))
            self.assertTrue(mask[loc], f"{loc} is not a nonzero mask voxel")
            seen.add(loc)
        self.assertEqual(raised, 0, f"raised {raised}/40 on a reachable mask")
        self.assertGreater(len(seen), 1)

    def test_scan_region_is_uniform(self):
        mask = self._margin_heavy(core=4)
        img = np.zeros(mask.shape, dtype="float32")
        ds = Dataset(tag="t")
        ds.add_data("input", img)
        ds.add_mask("m", mask, loc=True)
        fov = (41, 9, 9)
        spec = {"input": (1,) + fov, "m": (1,) + fov}
        ds.set_spec(spec)
        region = ds.locs["box"].intersect(ds.valid_range(spec))

        counts = {}
        for _ in range(2000):
            loc = tuple(ds._scan_region(region))
            self.assertTrue(mask[loc])
            counts[loc] = counts.get(loc, 0) + 1
        self.assertEqual(len(counts), 4 ** 3, "scan must reach every core voxel")
        exp = 2000 / len(counts)
        sd = (2000 * (1 / len(counts)) * (1 - 1 / len(counts))) ** 0.5
        self.assertLess(max(abs(c - exp) for c in counts.values()) / sd, 5.0)


class TestReservoirWeighting(unittest.TestCase):
    """Both scans keep a size-1 reservoir across batches of unequal size.

    A reservoir that weighted batches equally instead of by hit count would
    still pass a test whose batches happen to be the same size, so these make
    the sizes deliberately lopsided.
    """

    def test_scan_region_weights_slabs_by_hit_count(self):
        # z-slabs carrying 1, 60, 1, 200, 1 hits
        shape = (24, 40, 40)
        mask = np.zeros(shape, dtype="uint8")
        plan = {8: 1, 10: 60, 12: 1, 14: 200, 16: 1}
        for z, k in plan.items():
            flat = np.arange(k) * 7 % (30 * 30)
            ys, xs = np.unravel_index(flat, (30, 30))
            mask[z, ys + 5, xs + 5] = 1
        total = int(np.count_nonzero(mask))

        img = np.zeros(shape, dtype="float32")
        with force_box():
            ds = Dataset(tag="t")
            ds.add_data("input", img)
            ds.add_mask("m", mask, loc=True)
            spec = {"input": (1, 5, 9, 9), "m": (1, 5, 9, 9)}
            ds.set_spec(spec)
        region = ds.locs["box"].intersect(ds.valid_range(spec))

        n = 20000
        per_z = {}
        for _ in range(n):
            loc = tuple(ds._scan_region(region))
            self.assertTrue(mask[loc])
            per_z[loc[0]] = per_z.get(loc[0], 0) + 1

        # each slab's share must track its hit count, not 1/len(slabs)
        for z, k in plan.items():
            exp = n * k / total
            sd = (n * (k / total) * (1 - k / total)) ** 0.5
            self.assertLess(abs(per_z.get(z, 0) - exp) / sd, 5.0,
                            f"slab z={z} got {per_z.get(z, 0)}, expected ~{exp:.0f}")

    def test_scan_location_spans_multiple_chunks(self):
        """The index array must be walked in chunks, not just the first one."""
        n = 96
        mask = np.zeros((n, n, n), dtype="uint8")
        mask[0:2] = 1                       # bulk, all inside the fov margin
        mask[48:50, 48:50, 48:50] = 1       # the reachable part
        ds, spec = make_dataset(mask, 17)
        self.assertIsNotNone(ds.locs["data"])

        valid = ds.valid_range(spec)
        support = expected_support(ds, mask, spec)
        counts = {}
        with small_scan_chunk(4096):
            self.assertGreater(ds.locs["data"].size, Dataset.LOC_SCAN_CHUNK * 4,
                               "index array must span several chunks")
            for _ in range(3000):
                loc = tuple(ds._scan_location(valid))
                self.assertIn(loc, support)
                counts[loc] = counts.get(loc, 0) + 1
        self.assertEqual(set(counts), support, "scan must reach every location")
        exp = 3000 / len(support)
        sd = (3000 * (1 / len(support)) * (1 - 1 / len(support))) ** 0.5
        self.assertLess(max(abs(c - exp) for c in counts.values()) / sd, 5.0)


class TestAnisotropic(unittest.TestCase):
    """Cubic shapes hide axis transpositions; these do not."""

    def test_anisotropic_mask_and_fov(self):
        shape, lo, hi = (20, 34, 48), (3, 5, 7), (17, 29, 41)
        mask = np.zeros(shape, dtype="uint8")
        mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
        img = np.zeros(shape, dtype="float32")
        fov = (5, 9, 13)

        def build(force):
            ds = Dataset(tag="t")
            ds.add_data("input", img)
            ds.add_mask("m", mask, loc=True)
            spec = {"input": (1,) + fov, "m": (1,) + fov}
            ds.set_spec(spec)
            return ds, spec

        with force_box():
            ds_box, spec = build(True)
        self.assertIsNone(ds_box.locs["data"])
        self.assertEqual(tuple(ds_box.locs["box"].min()), lo)
        self.assertEqual(tuple(ds_box.locs["box"].max()), hi)

        ds_idx, _ = build(False)
        self.assertIsNotNone(ds_idx.locs["data"])

        support = expected_support(ds_box, mask, spec)
        for s_ in draw(ds_box, spec, 8000):
            self.assertIn(s_, support)
        for s_ in draw(ds_idx, spec, 8000, seed=SEED + 7):
            self.assertIn(s_, support)


class TestCrossRepresentation(unittest.TestCase):
    """Compare the two representations against each other, not just to uniform."""

    def test_frequencies_agree_between_paths(self):
        mask = box_mask((26,) * 3, 5, 19)
        mask[8:11, 8:11, 8:11] = 0                     # not a plain box
        with force_box():
            ds_box, spec = make_dataset(mask, 8)
        ds_idx, spec_idx = make_dataset(mask, 8)
        self.assertIsNone(ds_box.locs["data"])
        self.assertIsNotNone(ds_idx.locs["data"])

        n = 150000
        a = draw(ds_box, spec, n, seed=SEED)
        b = draw(ds_idx, spec_idx, n, seed=SEED + 1)
        ca, cb = {}, {}
        for s_ in a:
            ca[s_] = ca.get(s_, 0) + 1
        for s_ in b:
            cb[s_] = cb.get(s_, 0) + 1
        self.assertEqual(set(ca), set(cb), "the two paths cover different supports")

        # total variation distance should sit at the sampling-noise floor
        tv = 0.5 * sum(abs(ca.get(k, 0) - cb.get(k, 0)) for k in set(ca) | set(cb)) / n
        floor = (len(ca) / (np.pi * n)) ** 0.5
        self.assertLess(tv, 2.0 * floor,
                        f"TV {tv:.4f} vs chance floor {floor:.4f}")


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
        with self.assertRaises(ValueError):
            ds.add_mask("b", box_mask((20,) * 3, 4, 16), loc=True)
        self.assertNotIn("b", ds.data, "a rejected mask must leave nothing behind")


class TestUnionValidation(unittest.TestCase):
    def _ds(self):
        ds = Dataset(tag="t")
        ds.add_data("input", np.zeros((28,) * 3, dtype="float32"))
        ds.add_mask("a", box_mask((28,) * 3, 4, 16), loc=True)
        return ds

    def test_mismatched_offset_is_rejected(self):
        ds = self._ds()
        with self.assertRaises(ValueError):
            ds.add_mask("b", box_mask((28,) * 3, 4, 16), offset=(1, 0, 0), loc=True)
        self.assertNotIn("b", ds.data, "a rejected mask must leave nothing behind")

    def test_union_after_a_box_representation(self):
        """A box must convert to an index array when a second mask arrives."""
        a = box_mask((28,) * 3, 4, 16)
        b = box_mask((28,) * 3, 12, 24)
        with force_box():
            ds = Dataset(tag="t")
            ds.add_data("input", np.zeros(a.shape, dtype="float32"))
            ds.add_mask("a", a, loc=True)
            self.assertIsNone(ds.locs["data"], "first mask should be a box")
            ds.add_mask("b", b, loc=True)
        self.assertIsNone(ds.locs["box"], "the union cannot stay a box")
        expected = np.union1d(np.flatnonzero(a), np.flatnonzero(b))
        np.testing.assert_array_equal(ds.locs["data"], expected)
        self.assertEqual(ds.locs["keys"], ["a", "b"])


class TestChannelledMask(unittest.TestCase):
    def test_4d_mask_uses_the_index_array(self):
        mask = np.zeros((2,) + (24,) * 3, dtype="uint8")
        mask[:, 5:19, 5:19, 5:19] = 1
        img = np.zeros((24,) * 3, dtype="float32")
        with force_box():                       # even so, 4-D cannot be a box
            ds = Dataset(tag="t")
            ds.add_data("input", img)
            ds.add_mask("m", mask, loc=True)
        self.assertIsNone(ds.locs["box"])
        self.assertIsNotNone(ds.locs["data"])
        self.assertEqual(ds.locs["count"], int(np.count_nonzero(mask)))
        spec = {"input": (1, 8, 8, 8), "m": (1, 8, 8, 8)}
        ds.set_spec(spec)
        valid = ds.valid_range(spec)
        for loc in draw(ds, spec, 2000):
            self.assertTrue(valid.contains(Vec3d(loc)))


class TestTermination(unittest.TestCase):
    def test_disjoint_mask_raises_instead_of_hanging(self):
        """Previously an unbounded `while True`."""
        mask = np.zeros((40,) * 3, dtype="uint8")
        mask[0:3, 0:3, 0:3] = 1                # entirely inside the fov margin
        with force_box():
            ds, spec = make_dataset(mask, 16)
        self.assertIsNone(ds.locs["data"])
        valid = ds.valid_range(spec)
        self.assertFalse(valid.contains(Vec3d(2, 2, 2)))
        with self.assertRaises(Dataset.OutOfRangeError):
            ds._random_location(spec)

    def test_index_path_is_bounded(self):
        mask = np.zeros((40,) * 3, dtype="uint8")
        mask[0:3, 0:3, 0:3] = 1
        ds, spec = make_dataset(mask, 16)
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
