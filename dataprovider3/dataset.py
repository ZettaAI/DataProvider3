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

    # A mask whose nonzero voxels fill at least this fraction of its bounding
    # box is sampled by rejection: draw from the box, keep the draw if it lands
    # on a nonzero voxel. Expected draws is ~1/fill. Below the threshold we
    # store the nonzero indices instead -- which costs little precisely because
    # a sparse mask has few of them.
    REJECT_MIN_FILL = 0.05

    # `count` is measured over the whole mask while the draw comes from
    # box & valid, so the acceptance estimate can be optimistic. Give up after
    # this many draws and switch to the index array; falling back is sound.
    REJECT_LIMIT = 512

    # The index array has no further fallback, so it retries far longer before
    # declaring the location set unreachable.
    LOC_RETRY_LIMIT = 10000

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
        self.add_data(key, data, offset=offset)
        if loc:
            if self.locs is None:
                self.locs = self._describe_locs(key, data, offset)
            else:
                # A second location mask. The union of two nonzero sets is not
                # a box in general, so drop to the explicit index array -- the
                # only representation that unions exactly.
                assert data.shape == self.locs['dims'], (
                    "location masks must share a shape: "
                    f"{data.shape} vs {self.locs['dims']}")
                assert Vec3d(offset) == self.locs['offset'], (
                    "location masks must share an offset")
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
            return locs  # no locations at all; box stays None

        bounds = []
        for axis in range(3):
            others = tuple(i for i in range(3) if i != axis)
            hits = np.flatnonzero(np.any(data, axis=others))
            bounds.append((int(hits[0]), int(hits[-1]) + 1))
        box = Box(Vec3d(*[lo for lo, _ in bounds]),
                  Vec3d(*[hi for _, hi in bounds]))

        if count < Dataset.REJECT_MIN_FILL * np.prod(box.size()):
            # Sparse: rejection would need ~1/fill draws, and the index array
            # is cheap here anyway.
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
        if self.locs['data'] is None:
            loc = self._rejection_location(valid)
            if loc is not None:
                return loc
            # Not converging: the acceptance estimate was optimistic. Switch
            # representation permanently rather than keep spinning.
            self._materialize_locs()
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
        rejected. Returns None if the draws are not converging.
        """
        box = self.locs['box']
        if box is None:
            return None
        region = box.intersect(valid)
        if region is None:
            raise Dataset.OutOfRangeError()
        mask = self.data[self.locs['keys'][0]]
        for _ in range(Dataset.REJECT_LIMIT):
            loc = self._uniform_location(region)
            if mask.is_nonzero_at(loc):
                return loc
        return None

    def _indexed_location(self, valid):
        """Draw a nonzero voxel, keep it if the whole patch fits."""
        if self.locs['data'].size == 0:
            raise Dataset.OutOfRangeError()
        for _ in range(Dataset.LOC_RETRY_LIMIT):
            idx = np.random.choice(self.locs['data'], 1)
            loc = np.unravel_index(idx[0], self.locs['dims'])
            # Global coordinate system.
            loc = Vec3d(loc[-3:]) + self.locs['offset']
            if valid.contains(loc):
                return loc
        raise Dataset.OutOfRangeError()

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
