from data import *
from fenics import *

###### Define domain boundaries #####
def right(x, on_boundary): return near(x[0], xR)
def left(x, on_boundary): return near(x[0], xL)
def top(x, on_boundary): return near(x[1], yT)
def bottom(x, on_boundary): return near(x[1], yB)
def back(x, on_boundary): return near(x[2], zB)
def front(x, on_boundary): return near(x[2], zF)
def origin(x, on_boundary):
	if num_dims == 2:
		return near(x[0], 0.0) and near(x[1], 0.0)
	elif num_dims == 3:
		return near(x[0], 0.0) and near(x[1], 0.0) and near(x[2], 0.0)
def top_bottom(x, on_boundary): 
    return top(x, on_boundary) or bottom(x, on_boundary)
def front_back(x, on_boundary):
    return front(x, on_boundary) or back(x, on_boundary)
def left_right(x, on_boundary):
    return left(x, on_boundary) or right(x, on_boundary)

def get_ud(W_primal, primal_mesh, ut):
    """ Construct a function in the FEM space equal to BC on boundary and 0 in interior
        
        Args:
            W_primal -- Primal product function space
            primal_mesh -- primal_mesh
            ut -- True velocity
    """
    import util
    HO_ele = VectorElement("Lagrange", primal_mesh.ufl_cell(), 7)
    HO_space = FunctionSpace(primal_mesh, HO_ele)
    Boundary = AutoSubDomain(lambda x, on_bnd: on_bnd)
    Domain = AutoSubDomain(lambda x, on_bnd: True)
    if experiment_name == "Hartmann":
        ud = Function(HO_space)
        Zero = DirichletBC(HO_space, Constant((0.0, 0.0)), Boundary)
        Zero.apply(ud.vector())
        HO_bc = DirichletBC(HO_space, ut, Boundary)
        if use_lifting:
            HO_bc.apply(ud.vector())
    elif experiment_name == "lid_driven":
        if use_lifting:
            #small_mesh = RectangleMesh(Point(xL, yB), Point(xR, yT), 40, 40)
            small_mesh = primal_mesh
            # Chosen to approximate constant with \int_0^1 top_BC(x) dx = 1
            top_BC = Expression('30*(pow((x[0]+0.5), 2)*pow((x[0]-0.5), 2))', degree=4)
            HO_space_small_mesh = FunctionSpace(small_mesh, HO_ele)
            ud = Function(HO_space_small_mesh)
            vel_top = DirichletBC(HO_space_small_mesh.sub(0), top_BC, top)
            vel_top.apply(ud.vector())
            ud = interpolate(ud, HO_space)
        else:
            ud = Function(HO_space)
    return ud

class PeriodicBoundary(SubDomain):
    """ PeriodicBoundary in x """
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(abs(x[0] - xL) < DOLFIN_EPS and on_boundary)

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        y[0] = x[0] - (xR - xL) # xR - xL in case xL isn't zero
        y[1] = x[1]
        if num_dims == 3:
            y[2] = x[2] - (xR - xL)

def taylor_green(W_fluid, W_mag, ut):
    """" Boundary condition constructor for the Taylor Green fluid problem
        Args:
            [W_fuild, W_mag] -- Created by splitting W = (W_fluid, W_p, W_mag)
            ut -- True velocity
        Return:
            The BCs as an array
    """
    Zero = Constant(0.0)
    vel = DirichletBC(W_fluid.sub(0), ut, DomainBoundary())
    ### Manually set tangential components of B
    mag0 = DirichletBC(W_mag.sub(0), Zero, top)
    mag1 = DirichletBC(W_mag.sub(0), Zero, bottom)
    mag2 = DirichletBC(W_mag.sub(1), Zero, left)
    mag3 = DirichletBC(W_mag.sub(1), Zero, right)
    BCs = [vel, mag0, mag1, mag2, mag3]
    #BCs = [vel]
    return BCs

def hartmann_2D(W_fluid, W_mag, ut):
    """ Construct all Dirichlet BC's for u and B in 2D Hartmann
        Args:
            [W_fuild, W_mag] -- Created by splitting W = (W_fluid, W_p, W_mag)
            ut -- True velocity. Zero if lifting is in use.
        Return:
            The BCs as an array
    """
    One = Constant(1.0); Zero = Constant(0.0)
    if use_periodic:
        vel = DirichletBC(W_fluid.sub(0), ut, top_bottom)
        p0 = None
    else:
        vel = DirichletBC(W_fluid.sub(0), ut, DomainBoundary())
        #p0 = DirichletBC(W_fluid.sub(1), p, DomainBoundary())
    ### Manually set tangential components of B
    mag0 = DirichletBC(W_mag.sub(0), Zero, top)
    mag1 = DirichletBC(W_mag.sub(0), Zero, bottom)
    mag2 = DirichletBC(W_mag.sub(1), One, left)
    mag3 = DirichletBC(W_mag.sub(1), One, right)
    if use_periodic:
        BCs = [vel, mag0, mag1, mag2]#, mag2, mag3]
    else:
        BCs = [vel, mag0, mag1, mag2, mag3]#, p0]
    return BCs

def hartmann_3D(W_fluid, W_mag, u, B):
    """ UNDER DEVELOPMENT """
    Zero3 = Constant((0.0, 0.0, 0.0))
    # Series solution
    if do_series_solution:
        #vel0 = DirichletBC(W_fluid.sub(0), Zero3, top_bottom)
        #vel1 = DirichletBC(W_fluid.sub(0), Zero3, front_back)
        #mag0 = DirichletBC(W_mag, B, top_bottom)
        #mag1 = DirichletBC(W_mag, B, front_back)
        #BCs = [vel0, vel1, mag0, mag1]
        vel0 = DirichletBC(W_fluid.sub(0), u, DomainBoundary())
        mag0 = DirichletBC(W_mag, B, DomainBoundary())
        BCs = [vel0, mag0]
    # 1D profile 
    else:
        Zero = Constant(0.0); One = Constant(1.0)
        vel0 = DirichletBC(W_fluid.sub(0).sub(0), Zero, top_bottom)
        vel1 = DirichletBC(W_fluid.sub(0).sub(1), Zero, DomainBoundary())
        vel2 = DirichletBC(W_fluid.sub(0).sub(2), Zero, DomainBoundary())
        mag0 = DirichletBC(W_mag.sub(0), Zero, top_bottom)
        mag1 = DirichletBC(W_mag.sub(1), One, DomainBoundary())
        mag2 = DirichletBC(W_mag.sub(2), Zero, DomainBoundary())
        #p0 = bc_lower_fixed_point=DirichletBC(W_fluid.sub(1),Constant(0),origin,method='pointwise')
        BCs = [vel0, vel1, vel2, mag0, mag1, mag2]#, p0]
    return BCs

def lid_driven_BC(W_fluid, W_mag):
    """" Boundary condition constructor for the magnetic lid driven cavity
        Args:
            [W_fuild, W_mag] -- Created by splitting W = (W_fluid, W_p, W_mag)
        Return:
            The BCs as an array
    """
    B_field_strength = Constant(-1.0); Zero = Constant(0.0)
    # Integrates to 1, polynomial lid profile
    top_BC = Expression('30*(pow((x[0]+0.5), 2)*pow((x[0]-0.5), 2))', degree=4)
    ### Set ux to 0.0 on top, bottom, left. Flow on top
    vel0 = DirichletBC(W_fluid.sub(0).sub(0), Zero, left)
    vel1 = DirichletBC(W_fluid.sub(0).sub(0), Zero, right)
    vel2 = DirichletBC(W_fluid.sub(0).sub(0), Zero, bottom)
    if use_lifting: # Not used currently
        vel3 = DirichletBC(W_fluid.sub(0).sub(0), Zero, top)
    else:
        vel3 = DirichletBC(W_fluid.sub(0).sub(0), top_BC, top)
    vel4 = DirichletBC(W_fluid.sub(0).sub(1), Zero, DomainBoundary())
    ### Manually set tangential components of B
    mag0 = DirichletBC(W_mag.sub(0), B_field_strength, top)
    mag1 = DirichletBC(W_mag.sub(0), B_field_strength, bottom)
    mag2 = DirichletBC(W_mag.sub(1), Zero, left)
    mag3 = DirichletBC(W_mag.sub(1), Zero, right)
    # Collect boundary conditions
    #BCs = [vel0, vel1, vel2, vel3, vel4, mag0, mag1, mag2, mag3]
    BCs = [vel0, vel1, vel2, vel3, vel4, mag0, mag1, mag2, mag3]
    return BCs

def primal_BCs(W_fluid, W_mag, ut, Bt):
    """ Factory for creating the BCs based on the experiment being run.
        
        Args:
            [W_fuild, W_mag] -- Created by splitting W = (W_fluid, W_p, W_mag)
        Return:
            The BCs as an array
    """
    ### Note experiment_name comes from data.py
    if  experiment_name == "TaylorGreen":
        BCs = taylor_green(W_fluid, W_mag, ut)
    if  experiment_name == "Hartmann" and num_dims == 2:
        BCs = hartmann_2D(W_fluid, W_mag, ut)
    elif experiment_name == "Hartmann" and num_dims == 3:
        BCs = hartmann_3D(W_fluid, W_mag, ut, Bt)
    elif experiment_name == "lid_driven":
        BCs = lid_driven_BC(W_fluid, W_mag)
    return BCs

def adjoint_BCs(W):
    """" Large function for creating the adjoint BCs
   
        Args:
            W -- Monolithic function space
        Returns:
            The BCs as an array
    """
    W_fluid = W.sub(0); W_mag = W.sub(1)
    One = Constant(1.0); Zero = Constant(0.0)
    Zero2 = Constant((0.0, 0.0))
    if num_dims == 2:
        if use_periodic:
            vel = DirichletBC(W_fluid.sub(0), Zero2, top_bottom)
            ### Manually set tangential components of B
            mag0 = DirichletBC(W_mag.sub(0), Zero, top)
            mag1 = DirichletBC(W_mag.sub(0), Zero, bottom)
            mag2 = DirichletBC(W_mag.sub(1), Zero, left)
            BCs = [vel, mag0, mag1, mag2]
        else:
            vel = DirichletBC(W_fluid.sub(0), Zero2, DomainBoundary())
            mag0 = DirichletBC(W_mag.sub(0), Zero, top_bottom)
            mag1 = DirichletBC(W_mag.sub(1), Zero, left_right)
            BCs = [vel, mag0, mag1]#, p0]
    ############### UNDER DEVELOPMENT ###############
    elif num_dims == 3:
        Zero3 = Constant((0.0, 0.0, 0.0))
        W_fluid = W.sub(0); W_mag = W.sub(1)
        if do_series_solution:
            #vel = DirichletBC(W_fluid.sub(0), Zero3, DomainBoundary())
            vel = DirichletBC(W_fluid.sub(0), Zero3, top_bottom)
            vel = DirichletBC(W_fluid.sub(0), Zero3, front_back)
            #mag = DirichletBC(W_mag, Zero3, DomainBoundary())
            BCs = [vel, mag]
            #mag0 = DirichletBC(W_mag.sub(0), Zero, front_back)
            #mag1 = DirichletBC(W_mag.sub(1), Zero, front_back)
            #mag2 = DirichletBC(W_mag.sub(0), Zero, top_bottom)
            #mag3 = DirichletBC(W_mag.sub(2), Zero, top_bottom)
            #mag4 = DirichletBC(W_mag.sub(1), Zero, left_right)
            #mag5 = DirichletBC(W_mag.sub(2), Zero, left_right)
            #BCs = [vel, mag0, mag1, mag2, mag3, mag4, mag5]
        else: # 1D profile
            Zero = Constant(0.0); One = Constant(1.0)
            vel0 = DirichletBC(W_fluid.sub(0).sub(0), Zero, top_bottom)
            vel1 = DirichletBC(W_fluid.sub(0).sub(1), Zero, DomainBoundary())
            vel2 = DirichletBC(W_fluid.sub(0).sub(2), Zero, DomainBoundary())
            mag0 = DirichletBC(W_mag.sub(0), Zero, top_bottom)
            mag1 = DirichletBC(W_mag.sub(1), Zero, DomainBoundary())
            mag2 = DirichletBC(W_mag.sub(2), Zero, DomainBoundary())
            #p0 = bc_lower_fixed_point=DirichletBC(W_fluid.sub(1),Constant(0),origin,method='pointwise')
            BCs = [vel0, vel1, vel2, mag0, mag1, mag2]#, p0]

    # Homogeneous Dirichlet BC everywhere for now
    return BCs
