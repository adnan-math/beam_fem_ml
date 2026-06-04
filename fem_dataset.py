import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
import ufl
from petsc4py import PETSc
from dolfinx.fem.petsc import LinearProblem
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells


class BeamDataset:
    def __init__(self,
            W=0.2,
            H=0.2,
            E=30e9,
            nu=0.2,
            traction=1000.0,
            nx=100, ny=20, nz=20,
            mesh_mode="relative"):   # relative or fixed

      # Geometry params
      self.W = W
      self.H = H

      # Material params
      self.E = E
      self.nu = nu
      self.traction = traction

      # Mesh mode
      self.mesh_mode = mesh_mode

      self.mu = E / (2 * (1 + nu))
      self.lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))

      # Mesh resolution
      self.nx = nx
      self.ny = ny
      self.nz = nz

    # --------------------------------------------------------
    # Create mesh for given L
    # --------------------------------------------------------
    def create_mesh(self, L):
        nx = int(self.nx * L / 1.0)

        domain = mesh.create_box( MPI.COMM_WORLD,
            [np.array([0.0, -self.W/2, -self.H/2]),
            np.array([L,  self.W/2,  self.H/2])],
            [nx, self.ny, self.nz],
            cell_type=mesh.CellType.hexahedron)
        return domain

    # --------------------------------------------------------
    # Solve FEM
    # --------------------------------------------------------
    def solve(self, L, traction):

        domain = self.create_mesh(L)
        fdim = domain.topology.dim - 1

        V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

        # Clamped boundary
        def left(x):
            return np.isclose(x[0], 0.0)

        left_facets = mesh.locate_entities_boundary(domain, fdim, left)
        left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)

        bc = fem.dirichletbc(np.array((0.0, 0.0, 0.0), dtype=PETSc.ScalarType),left_dofs, V)

        # Strain / stress
        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            return self.lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2.0 * self.mu * epsilon(u)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Top traction
        def top(x):
            return np.isclose(x[2], -self.H/2)

        top_facets = mesh.locate_entities_boundary(domain, fdim, top)

        tags = mesh.meshtags(domain,fdim, top_facets, np.full(len(top_facets), 1, dtype=np.int32))

        ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)

        T = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, -traction)))

        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        Lform = ufl.dot(T, v) * ds(1)

        problem = LinearProblem(a,Lform,bcs=[bc],petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
            petsc_options_prefix="beam_")

        uh = problem.solve()

        return domain, V, uh

    # --------------------------------------------------------
    # Extract centerline dataset
    # --------------------------------------------------------

    def sample_centerline(self, L, uh, domain, n_points=80, n_y=5, n_z=5):
    
        tdim = domain.topology.dim
        tree = bb_tree(domain, tdim)
    
        x_vals = np.linspace(0.0, L, n_points)
    
        y_vals = np.linspace(-self.W/2, self.W/2, n_y)
        z_vals = np.linspace(-self.H/2, self.H/2, n_z)
    
        data = []
    
        for x in x_vals:
    
            pts = []
    
            for y in y_vals:
                for z in z_vals:
    
                    X = np.array([x, y, z])
    
                    cells = compute_collisions_points(tree, X)
                    colliding = compute_colliding_cells(domain, cells, X)
    
                    if len(colliding.links(0)) == 0:
                        continue
    
                    cell = colliding.links(0)[0]
    
                    u = uh.eval(X, [cell])
    
                    pts.append(X + u)
    
            if len(pts) == 0:
                data.append([L, np.nan, np.nan])
                continue
    
            pts = np.array(pts)
    
            centroid = np.mean(pts, axis=0)
    
            data.append([L, centroid[0], centroid[2]])
    
        return np.array(data)
    # --------------------------------------------------------
    # Full pipeline for one beam
    # --------------------------------------------------------
    def generate(self, L, traction=None, n_points=100):

        if traction is None:
            traction = self.traction

        domain, V, uh = self.solve(L, traction)
        data = self.sample_centerline(L, uh, V, n_points)

        return data
