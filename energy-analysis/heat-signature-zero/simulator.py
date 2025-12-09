import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import splu
from scipy.sparse import issparse


class Heat2D:
    """
    2D heat equation solver using the ADI (Alternating Direction Implicit)
    method for improved stability and efficiency. Uses robust Gaussian sources.
    """

    def __init__(self, Lx, Ly, nx, ny, kappa, bc="dirichlet"):
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        self.kappa = kappa

        supported_bc = ["dirichlet", "neumann"]
        if bc not in supported_bc:
            print(
                f"Warning: Boundary condition '{bc}' not supported. Defaulting to 'dirichlet'."
            )
            bc = "dirichlet"
        self.bc = bc

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Store 1D components and identities for ADI decomposition
        self.Lx_1D = self._lap1d(self.nx, self.dx)
        self.Ly_1D = self._lap1d(self.ny, self.dy)
        self.Ix_1D = eye(self.nx, format="csr")
        self.Iy_1D = eye(self.ny, format="csr")

        self.L = self._build_laplacian()
        self.I = eye(self.nx * self.ny, format="csr")

    def _lap1d(self, n, d):
        """Builds the 1D Laplacian operator (scaled by 1/d^2) for the given BC."""
        main = -2.0 * np.ones(n)
        off = 1.0 * np.ones(n - 1)
        L = diags([off, main, off], [-1, 0, 1], shape=(n, n)).tolil()

        if self.bc == "neumann":
            L[0, 0] = -2.0
            L[0, 1] = 2.0
            L[-1, -1] = -2.0
            L[-1, -2] = 2.0
        elif self.bc == "dirichlet":
            L[0, :] = 0.0
            L[-1, :] = 0.0
            L[0, 0] = 0.0
            L[-1, -1] = 0.0

        return (1.0 / d**2) * L.tocsr()

    def _build_laplacian(self):
        """Constructs the full 2D Laplacian."""
        return kron(self.Iy_1D, self.Lx_1D) + kron(self.Ly_1D, self.Ix_1D)

    def _vectorize(self, U):
        return U.reshape(self.nx * self.ny, order="C")

    def _devectorize(self, u):
        return u.reshape(self.nx, self.ny, order="C")

    def _source_field(self, t, sources):
        """Assembles the source field S (K/s) using Gaussian spreading."""
        S = np.zeros((self.nx, self.ny), dtype=float)
        if not sources:
            return S

        for s in sources:
            t0, t1 = s["on"]
            if not (t0 <= t <= t1):
                continue

            Q = s.get("Q", s.get("q", 0.0))
            if callable(Q):
                Q = Q(t)
            if Q == 0.0:
                continue

            x0 = float(s["x"])
            y0 = float(s["y"])
            sigma = s.get("sigma")

            # Sigma must be present since it's injected in solve()
            if sigma is None:
                continue

            S += self._deposit_gaussian(x0, y0, Q, sigma)

        return S

    def _deposit_gaussian(self, x0, y0, Q, sigma):
        """Deposits Q using a normalized 2D Gaussian profile (Grid-independent)."""
        r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        G = np.exp(-r_sq / (2.0 * sigma**2))
        Integral_G = np.sum(G) * self.dx * self.dy

        if Integral_G == 0:
            return np.zeros((self.nx, self.ny), dtype=float)

        # Normalize the field so that sum(S * dx * dy) = Q
        S_nodal = G * Q / Integral_G
        return S_nodal

    def solve(self, dt, nt, T0=None, sources=None, store_every=1):
        """
        Solves the heat equation using the ADI method.
        Pre-processes sources to handle default 'on' window and missing 'sigma'.
        """
        if sources is None:
            sources = []

        # --- MODIFICATION: Pre-process sources for default 'on' window AND add missing 'sigma' ---
        t_end = nt * dt

        # Calculate sigma based on current grid spacing (used if 'sigma' is missing)
        max_ds = max(self.dx, self.dy)
        SOURCE_SPREAD_FACTOR = 2.5
        auto_sigma = SOURCE_SPREAD_FACTOR * max_ds

        processed_sources = []
        for s in sources:
            s_copy = s.copy()

            # 1. Handle missing 'on'
            if "on" not in s_copy or s_copy["on"] is None:
                s_copy["on"] = (0.0, t_end)

            # 2. Handle missing 'sigma' (NEW LOGIC)
            if "sigma" not in s_copy or s_copy["sigma"] is None:
                s_copy["sigma"] = auto_sigma

            processed_sources.append(s_copy)
        sources = processed_sources
        # ----------------------------------------------------------------------------------------

        U = (
            np.zeros((self.nx, self.ny), dtype=float)
            if T0 is None
            else np.full((self.nx, self.ny), T0, dtype=float)
        )
        r = self.kappa * dt / 2.0  # Half time step factor

        Ax = (self.Ix_1D - r * self.Lx_1D).tocsc()
        Ay = (self.Iy_1D - r * self.Ly_1D).tocsc()
        Ax_lu = splu(Ax)
        Ay_lu = splu(Ay)

        times, Us = [], []

        def add_store(n, t, U):
            if n % store_every == 0:
                times.append(t)
                Us.append(U.copy())

        t = 0.0
        add_store(0, t, U)

        # Main ADI time stepping loop
        for n in range(nt):
            t_half = t + dt / 2.0
            S_field = (
                self._source_field(t_half, sources) * dt / 2.0
            )  # Source contribution per half-step

            # --- Step 1: Implicit in X, Explicit in Y (Solves for U_star) ---
            RHS_field_x = U.copy()
            for i in range(self.nx):
                RHS_field_x[i, :] += r * (self.Ly_1D @ U[i, :])
            RHS_field_x += S_field

            U_star = np.zeros_like(U)
            for j in range(self.ny):
                U_star[:, j] = Ax_lu.solve(RHS_field_x[:, j])

            # --- Step 2: Implicit in Y, Explicit in X (Solves for U_next) ---
            RHS_field_y = U_star.copy()
            for j in range(self.ny):
                RHS_field_y[:, j] += r * (self.Lx_1D @ U_star[:, j])
            RHS_field_y += S_field

            U_next = np.zeros_like(U)
            for i in range(self.nx):
                U_next[i, :] = Ay_lu.solve(RHS_field_y[i, :])

            U = U_next
            t += dt
            add_store(n + 1, t, U)

        return np.array(times), np.array(Us)

    def sample_sensors(self, U_field, sensor_xy):
        """Interpolates temperature field at sensor locations using bilinear interpolation."""
        m = sensor_xy.shape[0]
        out = np.zeros(m)
        for i, (sx, sy) in enumerate(sensor_xy):
            fx = sx / self.dx
            fy = sy / self.dy
            ix = np.clip(int(np.floor(fx)), 0, self.nx - 2)
            iy = np.clip(int(np.floor(fy)), 0, self.ny - 2)
            tx = fx - ix
            ty = fy - iy

            v = (
                (1 - tx) * (1 - ty) * U_field[ix, iy]
                + tx * (1 - ty) * U_field[ix + 1, iy]
                + (1 - tx) * ty * U_field[ix, iy + 1]
                + tx * ty * U_field[ix + 1, iy + 1]
            )
            out[i] = v
        return out
