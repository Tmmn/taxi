import numpy as np
from random import shuffle

# special data types
from collections import deque
from scipy.interpolate import interp1d


class City:
    """
    Represents a grid on which taxis are moving.

    Request origin/destination distributions are configured as a mixture model.
    Each list entry is one mixture component with a relative `strength`.
    Strengths are normalized internally to probabilities via cumulative sums.

    Supported component schemas:
    - Gaussian component:
      {"location": [x0, y0], "sigma": float, "strength": float}
    - Radial component around `location`:
      {"location": [x0, y0], "x": [...], "y": [...], "strength": float}
      where `x` are radii and `y` are unnormalized radial density values.

    Internal keys are added for radial components during initialization:
    `cdf_inv`, `interp_min`, `interp_max`.
    """

    def __init__(self, **config):
        """
        Parameters
        ----------

        n : int
            width of grid

        m : int
            height of grid

        base_coords : [int,int]
            grid coordinates of the taxi base

        request_origin_distributions : list of dicts
            encodes the distribution of request origins on the grid

            each list element is one mixture component dict:
                if Gaussian:
                    each Gaussian is given by its location mu, standard deviation sigma and strength (mixture weight f)
                    {"location":[int,int], "sigma":float, "strength":float}
                if arbitrary distribution with rotational symmetry:
                    {"location":[int,int], "x":[],"y":[],"strength":float}
                    x and y define the shape of the desired density function that we rotate around the center
                    strength is the relative mixture weight f (normalized internally)

        Attributes
        ----------
        A : list[set]
            per-cell set of available taxi_ids; indexed by continuous grid coordinate c

        N : dict[int, list[int]]
            maps each continuous coordinate c to its list of neighboring continuous coordinates

        n : int
            width of grid

        m : int
            height of grid

        length : int
            length of coordstacks that speed up random origin and destination generation
        """
        # random number generator
        self.rng = np.random.default_rng()

        # grid dimensions
        self.n = config["n"]  # number of pixels in x direction
        self.m = config["m"]  # number of pixels in y direction

        self.base_coords = config.get("base_coords", [int(self.n / 2), int(self.m / 2)])

        # list that stores taxi_id of available taxis at the
        # specific position on the grid
        # we initialize this array with empty sets
        # it can be a list because of the continuous indexing scheme
        self.A = [set() for _ in range(self.n * self.m)]

        # storing neighbors in a dict for fast access
        self.N = {c: self.neighbors(c) for c in range(self.n * self.m)}

        # generating stacks for request coordinate choice

        self.request_p = deque([])

        # deprecated distribution around base
        if "base_sigma" in config:
            self.request_origin_distributions = [
                {"location": self.base_coords, "sigma": config["base_sigma"], "strength": 1}]

        # current distributions defined in the config file
        if 'request_origin_distributions' in config:
            # origins
            self.request_origin_distributions = config['request_origin_distributions']
        # one deque for each distribution the final distribution is composed of
        self.request_origin_coordstacks = \
            [deque([]) for _ in range(len(self.request_origin_distributions))]
        # Extract relative component weights from config.
        self.request_origin_strengths = \
            [distr["strength"] for distr in self.request_origin_distributions]
        # Convert weights to cumulative probabilities for np.digitize binning.
        self.request_origin_probabilities = \
            np.cumsum(np.array(self.request_origin_strengths) / sum(self.request_origin_strengths))

        # destinations, if different from origins
        if 'request_destination_distributions' in config:
            self.request_destination_distributions = \
                config['request_destination_distributions']
            self.request_destination_coordstacks = \
                [deque([]) for _ in range(len(self.request_destination_distributions))]
            self.request_destination_strengths = \
                [distr["strength"] for distr in self.request_destination_distributions]
            self.request_destination_probabilities = \
                np.cumsum(np.array(self.request_destination_strengths) / sum(self.request_destination_strengths))
        else:
            self.request_destination_distributions = self.request_origin_distributions
            self.request_destination_coordstacks = self.request_origin_coordstacks
            self.request_destination_probabilities = self.request_origin_probabilities

        # if taxis start from a random place
        self.taxi_home_coordstack = deque([])

        self.hard_limit = config.get("hard_limit", self.n + self.m)
        self.length = config.get("length", int(2e5))

        # pre-storing coordinates
        # dicts for converting between real positions and their labels
        self.coordinate_dict_ij_to_c = {}
        for i in range(self.m):
            for j in range(self.n):
                if i in self.coordinate_dict_ij_to_c:
                    self.coordinate_dict_ij_to_c[i][j] = self.ij_to_c(i, j)
                else:
                    self.coordinate_dict_ij_to_c[i] = {j: self.ij_to_c(i, j)}

        self.coordinate_dict_c_to_ij = {}
        for c in range(self.n * self.m):
            self.coordinate_dict_c_to_ij[c] = self.c_to_ij(c)

        # pre-storing BFS-trees until the depth self.hard_limit
        self.bfs_trees = {}
        for c in range(self.n * self.m):
            self.bfs_trees[c] = self.create_BFS_tree(c)

        # pre-storing inverse CDFs for arbitrary distributions
        for d in self.request_origin_distributions:
            if "x" in d:
                r = np.array(d["x"])
                fr_unnormed = np.array(d["y"])
                norm = np.sum(2 * np.pi * r * fr_unnormed)
                fr_latent = 2 * np.pi * r * fr_unnormed / norm
                i = np.cumsum(fr_latent)
                # Precompute inverse CDF helpers used later by generate_coords(**distr_spec).
                cdf_inv = interp1d(i, r)
                d["cdf_inv"] = cdf_inv
                d["interp_min"] = np.min(i)
                d["interp_max"] = np.max(i)

        for d in self.request_destination_distributions:
            if "x" in d:
                r = np.array(d["x"])
                fr_unnormed = np.array(d["y"])
                norm = np.sum(2 * np.pi * r * fr_unnormed)
                fr_latent = 2 * np.pi * r * fr_unnormed / norm
                i = np.cumsum(fr_latent)
                # Precompute inverse CDF helpers used later by generate_coords(**distr_spec).
                cdf_inv = interp1d(i, r)
                d["cdf_inv"] = cdf_inv
                d["interp_min"] = np.min(i)
                d["interp_max"] = np.max(i)

    def _get_coordinates(self, probabilities, coordstacks, distributions):
        """
        Draw one coordinate from a distribution mixture.

        Parameters
        ----------
        probabilities : array-like of shape (k,)
            Monotonic cumulative probabilities in (0, 1], typically created by
            `np.cumsum(strengths / sum(strengths))`.

            Interpretation:
            - component 0 is chosen when `p <= probabilities[0]`
            - component i is chosen when `probabilities[i-1] < p <= probabilities[i]`

            Expected properties:
            - non-decreasing
            - last element is close to 1.0

            Example for 3 components:
            `probabilities = np.array([0.2, 0.7, 1.0])`

        coordstacks : list[collections.deque[tuple[int, int]]], length k
            Per-component cache of already-generated integer grid coordinates.
            `coordstacks[i]` stores coordinates for mixture component `i`.
            When a deque is empty, it is refilled from
            `generate_coords(**distributions[i])`.

        distributions : list[dict], length k
            Per-component distribution specs. The index must align with
            `probabilities` and `coordstacks`.

            Supported dict schemas:
            - Gaussian component
              {
                "location": [x0, y0],
                "sigma": float,
                "strength": float
              }
            - Radial component
              {
                "location": [x0, y0],
                "x": sequence[float],
                "y": sequence[float],
                "strength": float,
                # added internally in __init__:
                "cdf_inv": scipy.interpolate.interp1d,
                "interp_min": float,
                "interp_max": float
              }

            Only one component spec is used per call: `distributions[ind]`.

        Returns
        -------
        tuple[int, int]
            One valid grid coordinate `(x, y)` sampled from the selected
            component and clipped by the grid-boundary filter in
            `generate_coords`.

        Notes
        -----
        Argument coupling invariant for mixture case:
            len(probabilities) == len(coordstacks) == len(distributions)

        Single-component shortcut:
            if `len(probabilities) <= 1`, component index `0` is used directly.
        """
        if len(probabilities) > 1:
            try:
                p = self.request_p.pop()
            except IndexError:
                self.request_p.extend(self.rng.random(self.length))
                p = self.request_p.pop()
            # Pick mixture component index according to cumulative probabilities.
            ind = np.digitize(p, probabilities)
        else:
            ind = 0

        try:
            x, y = coordstacks[ind].pop()
        except IndexError:
            coordstacks[ind].extend(
                self.generate_coords(**distributions[ind])
            )
            x, y = coordstacks[ind].pop()

        return x, y

    def create_one_request_coord(self):
        # _get_coordinates arguments:
        # 1) request_origin_probabilities: cumulative probs from normalized strengths
        # 2) request_origin_coordstacks: per-component cache of generated coordinates
        # 3) request_origin_distributions: per-component specs (location/sigma/x/y/...)
        ox, oy = self._get_coordinates(
            self.request_origin_probabilities,
            self.request_origin_coordstacks,
            self.request_origin_distributions
        )

        # Same contract for destination mixture components.
        dx, dy = self._get_coordinates(
            self.request_destination_probabilities,
            self.request_destination_coordstacks,
            self.request_destination_distributions
        )

        return ox, oy, dx, dy

    @staticmethod
    def measure_distance(source, destination):
        """
        Measure distance on the grid between two points.

        Returns
        -------
        Source coordinates are marked by *s*,
        destination coordinates are marked by *d*.

        The distance is the following integer:
        $$|x_s-x_d|+|y_s-y_d|$$
        """

        return np.dot(np.abs(np.array(destination) - np.array(source)), [1, 1])

    @staticmethod
    def create_path(source, destination):
        """
        Choose a random shortest path between source and destination.

        Parameters
        ----------

        source : [int,int]
            grid coordinates of the source

        destination : [int,int]
            grid coordinates of the destination


        Returns
        -------

        path : list of coordinate tuples
            coordinate list of a random path between source and destinaton

        """

        # distance along the x and the y axis
        d = dict(zip(['x', 'y'], np.array(destination) - np.array(source)))

        # create a sequence of "x"-es and "y"-s
        # we are going to shuffle this sequence
        # to get a random order of "x" and "y" direction steps
        sequence = ['x'] * int(np.abs(d['x'])) + ['y'] * int(np.abs(d['y']))
        shuffle(sequence)

        # source is included in the path
        path = [source]
        for item in sequence:
            # we add one step in the right direction based on the last position
            path.append([
                np.sign(d[item]) * int(item == "x") + path[-1][0],
                np.sign(d[item]) * int(item == "y") + path[-1][1]
            ])

        return path

    def neighbors(self, c):
        """
        Calculate the neighbors of a coordinate.
        On the edges of the simulation grid, there are no neighbors.
        (E.g. there are only 2 neighbors in the corners.)

        Parameters
        ----------

        c : int
            continuous grid index (as produced by ij_to_c)

        Returns
        -------

        ns : list of int
            continuous grid indices of all valid neighbors of c
        """

        coordinates = self.c_to_ij(c)

        ns = [(coordinates[0] + dx, coordinates[1] + dy) for dx, dy in [(1, 0), (0, 1), (-1, 0), (0, -1)]]
        ns = filter(lambda n: (0 <= n[0]) and (self.n > n[0]) and (0 <= n[1]) and (self.m > n[1]), ns)

        return [self.ij_to_c(x, y) for x, y in ns]

    def ij_to_c(self, i, j):
        # grid coordinates to continuous coordinates
        return self.n * i + j

    def c_to_ij(self, c):
        # continuous coordinates to grid coordinates
        return int(c / self.n), c % self.n

    # @profile
    def generate_coords(self, **distr_spec):
        """
        Generate a batch of request coordinates from one distribution component.

        Parameters
        ----------
        distr_spec : dict
            One component of `request_origin_distributions` or
            `request_destination_distributions`.

            Supported keys:
            - Gaussian branch:
              * `location` : list[int, int], center on the grid
              * `sigma` : float, standard deviation in grid units for x and y
              * `strength` : float, mixture weight (used by `_get_coordinates`,
                not by this function directly)
            - Radial branch:
              * `location` : list[int, int], center of radial distribution
              * `x` : sequence[float], radii support
              * `y` : sequence[float], unnormalized radial profile values
              * `strength` : float, mixture weight (used externally)
              * `cdf_inv`, `interp_min`, `interp_max` : internal precomputed
                interpolation helpers set in `__init__`

        Returns
        -------

        filter iterator of (int, int) tuples
            coordinate pairs sampled from the given distribution and filtered to lie within the n×m grid

        """
        if "sigma" in distr_spec:
            # Sample isotropic Gaussian offsets, then shift by configured location.
            temp = map(
                lambda t: (int(round(t[0] * distr_spec["sigma"], 0) + distr_spec["location"][0]),
                           int(round(t[1] * distr_spec["sigma"], 0)) + distr_spec["location"][1]),
                self.rng.normal(size=(self.length, 2))
            )
        else:
            # Sample radius from inverse CDF and angle uniformly to get radial points.
            u = self.rng.uniform(size=(self.length,))
            u = u[np.nonzero((distr_spec["interp_min"] < u) & (distr_spec["interp_max"] > u))]
            phi = 2 * np.pi * self.rng.uniform(size=np.size(u))
            x = distr_spec["cdf_inv"](u) * np.cos(phi)
            y = distr_spec["cdf_inv"](u) * np.sin(phi)

            temp = map(
                lambda t: (int(round(t[0], 0) + distr_spec["location"][0]),
                           int(round(t[1], 0) + distr_spec["location"][1])),
                zip(x, y)
            )

        temp = filter(lambda n: (0 <= n[0]) and (self.n > n[0]) and (0 <= n[1]) and (self.m > n[1]), temp)

        return temp

    def create_taxi_home_coords(self):
        """

        Returns
        -------

        Random coordinate pair from the grid.

        """
        try:
            hx, hy = self.taxi_home_coordstack.pop()
        except IndexError:
            temp = list(zip(self.rng.integers(0, self.n, 1000), self.rng.integers(0, self.m, 1000)))
            self.taxi_home_coordstack.extend(temp)
            hx, hy = self.taxi_home_coordstack.pop()
        return hx, hy

    # @profile
    def find_nearest_available_taxis(
            self,
            source,
            mode="nearest",
            radius=None):
        """
        This function lists the available taxis according to mode.


        Parameters
        ----------

        source : tuple, no default
            coordinates of the place from which we want to determine the nearest
            possible taxi

        mode : str, default "nearest"
            determines the mode of taxi listing
                * "nearest" lists only the nearest taxis, returns a list where there \
                are all taxis at the nearest possible distance from the source

                * "circle" lists all taxis within a certain distance of the source

        radius : int, optional
            if mode is "circle", gives the circle radius

        Returns
        -------
        list of int
            taxi IDs of available taxis matching the query
        """

        if mode == "nearest":
            radius = self.hard_limit

        # select BFS-tree
        tree = self.bfs_trees[source]

        # current depth storage
        depth = 0
        # list of available taxis
        p = set()

        while depth < radius:
            # take the next nodes
            ta = set.union(*[self.A[node] for node in tree[depth]])
            p = p.union(ta)

            if mode == "nearest" and len(ta) > 0:
                break

            depth += 1

        return list(p)

    def create_BFS_tree(
            self,
            source,
            max_depth=None
    ):
        """

        Parameters
        ----------
        source: int
            origin node as a continuous grid index (as produced by ij_to_c)
        max_depth: int
            depth of tree

        Returns
        -------

        dictionary storing BFS-tree of depth max depths around given source

        """

        if max_depth is None:
            max_depth = self.hard_limit + 1

        # BFS init
        # queue for BFS visit
        q = deque()
        q.append(source)
        # visited nodes with distance from the source node
        visited = {source: 0}
        # current depth storage
        depth = 0

        # while we still have nodes to visit
        while len(q) != 0 and depth < max_depth:
            # take the next node
            v = q.popleft()

            # visit the neighbors of v
            for n in self.N[v]:
                if n not in visited:
                    q.append(n)
                    # the depth of the neighbor is one more than that of its parent in the BFS tree
                    depth = visited[v] + 1
                    # if we surpass the search radius, quit BFS
                    if depth > max_depth:
                        break
                    # store node in the visited ones
                    visited[n] = depth

        bfs_tree = {}
        # reverse dict
        for k, v in visited.items():
            if v in bfs_tree:
                bfs_tree[v].append(k)
            else:
                bfs_tree[v] = [k]

        return bfs_tree
