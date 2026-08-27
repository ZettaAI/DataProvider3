from collections import OrderedDict
import copy
import numpy as np

from .geometry import Box, Vec3d
from .tensor import TensorData
from . import utils


class Dataset(object):
    """Dataset for volumetric data.

    Attributes:
        spec (dict): mapping key to tensor's shape.
        data (dict): mapping key to TensorData.
        locs (dict): valid locations. Either a bounding box the sampler draws
            from and rejects against the mask ('box'), or an explicit array of
            flat nonzero indices ('data') for masks too sparse to hit by
            chance. See add_mask.
    """

    # A location mask is described one of two ways, decided once when the mask
    # is registered and never revised afterwards. Deciding eagerly is the whole
    # point: DataLoader workers fork after the Datasets are built, so anything
    # allocated later is private to a worker and multiplies by the worker count
    # -- the opposite of what this representation is for.
    #
    # The index array is used whenever it is small enough not to matter, and
    # for masks too sparse for rejection to converge. Otherwise the mask is
    # described by its bounding box, which is what keeps a dense mask over a
    # large volume from costing 8 bytes per nonzero voxel.
    LOC_INDEX_MAX_BYTES = 64 * 2**20
    REJECT_MIN_FILL = 0.05

    # Draw budgets. The box path only ever serves masks filling at least
    # REJECT_MIN_FILL of their bounding box, so exhausting it means the part of
    # the mask inside `valid` is empty, not merely rare.
    REJECT_LIMIT = 10000
    LOC_RETRY_LIMIT = 1000

    # Above this many nonzeros, resolving the valid subset exactly costs more
    # than it saves; see _scan_location.
    LOC_SCAN_MAX = 2**21

    def __init__(self, spec=None, tag=''):
        self.set_spec(spec)
        self.tag = tag
        self.data = dict()
        self.locs = None

    def __call__(self, spec=None):
        return self.random_sample(spec=spec)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += self.tag
        format_string += ')'
        return format_string

    def sanity_check(self, spec):
        return all([k in self.data for k in spec])

    def add_data(self, key, data, offset=(0,0,0)):
        self.data[key] = TensorData(data, offset=offset)

    def add_mask(self, key, data, offset=(0,0,0), loc=False):
        if loc and self.locs is not None:
            # A second location mask. Its indices are unraveled against the
            # first mask's dims and translated by the first mask's offset, so
            # both must match or the resulting coordinates are silently wrong.
            # Checked before add_data so a rejected mask leaves nothing behind.
            if data.shape != self.locs['dims']:
                raise ValueError(
                    "location masks must share a shape: "
                    f"{data.shape} vs {self.locs['dims']}")
            if Vec3d(offset) != self.locs['offset']:
                raise ValueError(
                    "location masks must share an offset: "
                    f"{tuple(Vec3d(offset))} vs {tuple(self.locs['offset'])}")

        self.add_data(key, data, offset=offset)

        if loc:
            if self.locs is None:
                self.locs = self._describe_locs(key, data, offset)
            else:
                # The union of two nonzero sets is not a box in general, so
                # drop to the explicit index array -- the only representation
                # that unions exactly.
                self._materialize_locs()
                self.locs['data'] = np.union1d(self.locs['data'],
                                               np.flatnonzero(data))
                self.locs['count'] = int(self.locs['data'].size)
                self.locs['keys'].append(key)

    def _describe_locs(self, key, data, offset):
        """Summarize a mask's sampleable region without listing its indices.

        Records the nonzero count and bounding box, both of which come from
        streaming reductions that allocate nothing of consequence. Masks too
        sparse for rejection sampling get the historical index array here.
        """
        locs = dict(keys=[key], array=data, dims=data.shape,
                    offset=Vec3d(offset), data=None, box=None, count=0)

        if data.ndim != 3:
            # Channelled masks are rare; keep the historical representation.
            locs['data'] = np.flatnonzero(data)
            locs['count'] = int(locs['data'].size)
            return locs

        count = int(np.count_nonzero(data))
        locs['count'] = count
        if count == 0:
            # No locations at all. Keep the empty index array so the sampling
            # path has something to report OutOfRangeError from.
            locs['data'] = np.flatnonzero(data)
            return locs

        bounds = []
        for axis in range(3):
            others = tuple(i for i in range(3) if i != axis)
            hits = np.flatnonzero(np.any(data, axis=others))
            bounds.append((int(hits[0]), int(hits[-1]) + 1))
        box = Box(Vec3d(*[lo for lo, _ in bounds]),
                  Vec3d(*[hi for _, hi in bounds]))

        small = count * 8 <= Dataset.LOC_INDEX_MAX_BYTES
        sparse = count < Dataset.REJECT_MIN_FILL * np.prod(box.size())
        if small or sparse:
            # Small enough not to matter, or too sparse for rejection to
            # converge. Either way the array is built here, before any fork,
            # so workers share it instead of each allocating its own. `sparse`
            # alone is not enough: 4.9% of a huge bounding box is still GiBs.
            locs['data'] = np.flatnonzero(data)
            return locs

        box.translate(locs['offset'])  # global coordinate system
        locs['box'] = box
        return locs

    def _materialize_locs(self):
        """Switch to the explicit index array, once."""
        if self.locs['data'] is None:
            self.locs['data'] = np.flatnonzero(self.locs['array'])
            self.locs['count'] = int(self.locs['data'].size)
            self.locs['box'] = None

    def set_spec(self, spec):
        self.spec = None
        if spec is not None:
            self.spec = dict(spec)

    def get_patch(self, key, pos, dim):
        """Extract a patch from the data tagged with `key`."""
        assert key in self.data
        assert len(pos)==3 and len(dim)==3
        return self.data[key].get_patch(pos, dim)

    def get_sample(self, pos, spec=None):
        """Extract a sample centered on pos."""
        spec = self._validate(spec)
        sample = dict()
        for key, dim in spec.items():
            patch = self.get_patch(key, pos, dim[-3:])
            if patch is None:
                raise Dataset.OutOfRangeError()
            sample[key] = patch
        return utils.sort(sample)

    def random_sample(self, spec=None):
        """Extract a random sample."""
        spec = self._validate(spec)
        try:
            pos = self._random_location(spec)
            ret = self.get_sample(pos, spec)
        except Dataset.OutOfRangeError:
            print("out-of-range error")
            raise
        except:
            raise
        return ret

    def num_samples(self, spec=None):
        try:
            if self.locs is None:
                spec = self._validate(spec)
                valid = self._valid_range(spec)
                num = np.prod(valid.size())
            else:
                num = self.locs['count']
        except Dataset.NoSpecError:
            nums = list()
            for k, v in self.data.items():
                nums.append(np.prod(v.dim()))
            num = min(nums)
        except:
            raise
        return num

    def valid_range(self, spec=None):
        spec = self._validate(spec)
        return self._valid_range(spec)

    ####################################################################
    ## Private Helper Methods.
    ####################################################################

    def _validate(self, spec):
        if spec is None:
            if self.spec is None:
                raise Dataset.NoSpecError()
            spec = dict(self.spec)
        assert all([k in self.data for k in spec])
        return spec

    def _random_location(self, spec):
        """Return a random valid location.

        Every branch yields a location drawn uniformly from
        `nonzero(mask) & valid` (or from `valid` alone when no location mask
        was registered) -- only the proposal differs.
        """
        valid = self._valid_range(spec)
        if self.locs is None:
            return self._uniform_location(valid)
        if self.locs['box'] is not None:
            return self._rejection_location(valid)
        return self._indexed_location(valid)

    def _uniform_location(self, box):
        """Uniform draw over a box, in the global coordinate system."""
        s = tuple(box.size())
        if any(v <= 0 for v in s):
            raise Dataset.OutOfRangeError()
        x = np.random.randint(0, s[-1])
        y = np.random.randint(0, s[-2])
        z = np.random.randint(0, s[-3])
        return Vec3d(z,y,x) + box.min()

    def _rejection_location(self, valid):
        """Draw from the mask's bounding box, keep it if the mask is nonzero.

        The box contains every nonzero voxel, so restricting the proposal to
        `box & valid` leaves the accepted set -- and therefore the sampling
        distribution -- unchanged, while removing the need for an index array.
        A solid-box mask fills its own bounding box, so nothing is ever
        rejected.
        """
        assert len(self.locs['keys']) == 1, (
            "the box describes a single mask; a union forces the index array")
        box = self.locs['box']
        region = box.intersect(valid)
        if region is None:
            raise Dataset.OutOfRangeError()
        mask = self.data[self.locs['keys'][0]]
        for _ in range(Dataset.REJECT_LIMIT):
            loc = self._uniform_location(region)
            if mask.is_nonzero_at(loc):
                return loc
        # This path only serves masks filling >= REJECT_MIN_FILL of their box,
        # so this many misses means the mask has (near enough) nothing inside
        # `valid`, not that we were unlucky.
        raise Dataset.OutOfRangeError()

    def _indexed_location(self, valid):
        """Draw a nonzero voxel, keep it if the whole patch fits."""
        data = self.locs['data']
        if data.size == 0:
            raise Dataset.OutOfRangeError()
        for _ in range(Dataset.LOC_RETRY_LIMIT):
            idx = np.random.choice(data, 1)
            loc = np.unravel_index(idx[0], self.locs['dims'])
            # Global coordinate system.
            loc = Vec3d(loc[-3:]) + self.locs['offset']
            if valid.contains(loc):
                return loc
        return self._scan_location(valid)

    def _scan_location(self, valid):
        """Resolve the locations inside `valid` exactly, in one pass.

        Random retries can miss a support that is real but rare, and simply
        giving up after a fixed number of draws would raise on inputs the
        unbounded loop this replaced always served. So settle it exactly rather
        than probabilistically: this either returns a location or proves none
        exists.
        """
        data = self.locs['data']
        if data.size > Dataset.LOC_SCAN_MAX:
            # Scanning would cost more than the draws already spent.
            raise Dataset.OutOfRangeError()
        coords = np.unravel_index(data, self.locs['dims'])
        offset, lo, hi = self.locs['offset'], valid.min(), valid.max()
        inside = np.ones(data.size, dtype=bool)
        for i in range(3):
            c = coords[i - 3] + offset[i]
            inside &= (c >= lo[i]) & (c < hi[i])
        hits = np.flatnonzero(inside)
        if hits.size == 0:
            raise Dataset.OutOfRangeError()
        pick = hits[np.random.randint(0, hits.size)]
        return Vec3d(*[int(coords[i - 3][pick]) + offset[i] for i in range(3)])

    def _valid_range(self, spec):
        """Compute the valid range, which is intersection of the valid range
        of each TensorData.
        """
        valid = None
        for key, dim in spec.items():
            assert key in self.data
            v = self.data[key].valid_range(dim[-3:])
            if v is None:
                raise Dataset.OutOfRangeError()
            valid = v if valid is None else valid.intersect(v)
        assert valid is not None
        return valid

    ####################################################################
    ## Exceptions.
    ####################################################################

    class OutOfRangeError(Exception):
        pass

    class NoSpecError(Exception):
        pass
