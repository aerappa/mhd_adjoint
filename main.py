from dolfin import *
import pickle
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
import sys
#set_log_level(50) # CRITICAL only

def get_qoi_psi(msh, dof):
    """ Return the characteristic function from a DG space.
    
        Args:
            msh -- The mesh to create on
            dof -- u or B
    """
    import util
    if dof == "u":
        psi_c = util.Psi_u(degree=0)
    elif dof == "B":
        psi_c = util.Psi_B(degree=0)
    dg0 = FiniteElement("DG", msh.ufl_cell(), 0)
    DG0 = FunctionSpace(msh, dg0)
    psi = Function(DG0)
    psi.interpolate(psi_c)
    return psi

def make_table_entry(T, effectivity, forcing_term):
    """ Given terms, effectivity, and err_est, create LaTeX table entry

        Args:
        T -- array containing the terms
        effectivity -- effectivity ratio (Psi, E) / Eta
        forcing_term -- (f, phi)
    """
    global ndofs, true_err
    import data
    E_mom = -(T[0]+T[1]+T[2]+T[3])
    E_con = -T[4]
    E_ind = -(T[5] + T[6])
    E_pen = -(T[7])
    if data.experiment_name == "Hartmann" and data.use_periodic == True:
        E_mom = forcing_term + E_mom
    E_F = E_mom + E_con
    E_M = E_ind + E_pen
    err_file = open('tables/' +str(data.num_dims) + 'D_' + data.experiment_name+'-'+ data.spaces + '.txt', 'a+')
    table_str = '{0} & {1:.2e} & {2:.2f} & {3:.2e} & {4:.2e} &{5:.2e}\\\\\n\\hline\n'\
    .format(primal_ele, true_err, effectivity, E_mom, E_con, E_M)
    err_file.write(table_str)
    err_file.close()

def print_adjoint_estimates(u, p, B, phi, pi, beta, adjoint_mesh, RHS):
    """ Given U_ho, Phi, print to screen RHS and effectivity, then call make_table_entry

        Args:
        [u, p, B] -- high order primal state
        [phi, pi, beta] -- high order adjoint state
        adjoint_mesh -- same as primal mesh by default
        RHS -- forcing term for momentum f
    """
    global true_err, primal_ele
    import util, data, forms
    T = forms.assemble_terms(u, p, B, phi, pi, beta, adjoint_mesh)
    forcing_term = assemble(inner(RHS, phi)*dx)
    err_est = forcing_term
    #util.save(phi, pi, beta, 'Phi', 'pi', 'beta', 'plots/')
    if rank == 0:
        print('RHS={}'.format(err_est))
    for i in range(len(T)):
        err_est -= T[i]
    if rank == 0:
        print("Adjoint error estimate: {}".format(err_est))
        effectivity = err_est/true_err
        print("Effectivity: {}".format(effectivity))
        make_table_entry(T, effectivity, forcing_term)

def solve_adjoint_problem(W_adjoint, adjoint_mesh, u, p, B, psi_u, psi_B, RHS):
    """ Solve the adjoint problem linearized about U = [u, p, B]

        Args:
        W_adjoint -- adjoint product space
        adjoint_mesh -- same as primal mesh by default
        [u, p, B] -- high order primal state
        [psi_u, psi_B] -- respreseters of QoI Q(U) = (psi_u, u) + (psi_B, B)
        RHS -- Forcing term for momentum f
    """
    import adjoint_mhd as adjoint
    import data, util
    Phi = adjoint.get_greens_function(W_adjoint, adjoint_mesh, u, B, psi_u, psi_B)
    (Phi1, beta) = Phi.split()
    (phi, pi)  = Phi1.split()
    if data.do_plotting and rank == 0:
        print('Plotting adjoint variables..')
        util.plot_soln(phi, pi, beta)
    print_adjoint_estimates(u, p, B, phi, pi, beta, adjoint_mesh, RHS)

def setup_spaces(primal_mesh, adjoint_mesh, lo_u, lo_p, lo_B, ho_u, ho_p, ho_B):
    """ Create the spaces for the problem
        
        Args:
            [primal_mesh, adjoint_mesh] -- By default these are the same
            [lo_u, lo_p, lo_B] -- Dimensions for low order product space
            [ho_u, ho_p, ho_B] -- Dimensions for high order product space
        
        Return:
            [W_primal, W_adjoint] -- Product spaces
    """
    global ndofs
    import data, util
    import primal_newton as primal
    import BC_factory
    # Build function space
    Pu = VectorElement("Lagrange", primal_mesh.ufl_cell(), lo_u)
    Pp = FiniteElement("Lagrange", primal_mesh.ufl_cell(), lo_p)
    PB = VectorElement("Lagrange", primal_mesh.ufl_cell(), lo_B)
    TH = Pu * Pp * PB
    if data.use_periodic:
        W_primal = FunctionSpace(primal_mesh, TH, constrained_domain=BC_factory.PeriodicBoundary())
    else: W_primal = FunctionSpace(primal_mesh, TH)
    if data.do_adjoint_analysis:
        # Build adjoint function space
        Pu = VectorElement("Lagrange", adjoint_mesh.ufl_cell(), ho_u)
        Pp = FiniteElement("Lagrange", adjoint_mesh.ufl_cell(), ho_p)
        PB = VectorElement("Lagrange", adjoint_mesh.ufl_cell(), ho_B)
        TH = Pu * Pp * PB
        if data.use_periodic:
            W_adjoint = FunctionSpace(adjoint_mesh, TH, constrained_domain=BC_factory.PeriodicBoundary())
        else: W_adjoint = FunctionSpace(adjoint_mesh, TH)
        ndofs = W_primal.dim()
        return W_primal, W_adjoint
    else:
        return W_primal, None

def setup_meshes(primal_xele, primal_yele, primal_zele, mult):
    """ Return meshes. Square in 2D, box in 3D """
    import data
    if data.num_dims == 3:
        primal_mesh = BoxMesh(Point(data.xL, data.yB, data.zB), Point(data.xR, data.yT, data.zF),
                              primal_xele, primal_yele, primal_zele)
        adjoint_mesh = BoxMesh(Point(data.xL, data.yB, data.zB), Point(data.xR, data.yT, data.zF),
                              mult*primal_xele, mult*primal_yele, mult*primal_zele)
    else:
        primal_mesh = RectangleMesh(Point(data.xL, data.yB), Point(data.xR, data.yT), primal_xele, primal_yele)
        adjoint_mesh = RectangleMesh(Point(data.xL, data.yB), Point(data.xR, data.yT),
                                     mult*primal_xele, mult*primal_yele)
    return primal_mesh, primal_mesh

def run_problem(primal_xele, primal_yele, primal_zele):
    """ Workhouse for running mesh refinement studies
        
        Args:
            [primal_xele, yele, zele] -- Numbers of elements. Primal_zele == nan in 2D
    """
    global true_err, computed_QoI
    import data
    import primal_newton as primal
    import util
    import BC_factory
    primal_mesh, adjoint_mesh = setup_meshes(primal_xele, primal_yele, primal_zele, data.mult)
    W_primal, W_adjoint = setup_spaces(primal_mesh, adjoint_mesh, data.lo_u, data.lo_p, data.lo_B,
                                       data.ho_u, data.ho_p, data.ho_B)
    # True solution
    ut = util.HartmannVelocity(degree=6)
    # For splitting, uDirichlet
    if data.use_lifting:
        ud = BC_factory.get_ud(W_primal, primal_mesh, ut)
    else:
        ud = None
    # True soltution
    Bt = util.HartmannBField(degree=6)
    if data.experiment_name == "TaylorGreen":
        RHS = util.TaylorGreenRHS(degree=7)
    else:
        RHS = util.HartmannRHS(degree=6)
    (u0, p0, B0) = primal.solve_exact_penalty(W_primal, RHS, ut, ud, Bt, primal_xele, primal_yele)
    psi_u = get_qoi_psi(primal_mesh, "u")
    psi_B = get_qoi_psi(primal_mesh, "B")
    computed_QoI, true_err = util.verify(u0, ud, p0, B0, psi_u, psi_B, primal_mesh, rank)
    if data.do_adjoint_analysis:
        if data.use_lifting:
            solve_adjoint_problem(W_adjoint, adjoint_mesh, u0+ud, p0, B0, psi_u, psi_B, RHS)
        else:
            solve_adjoint_problem(W_adjoint, adjoint_mesh, u0, p0, B0, psi_u, psi_B, RHS)

def driver():
    """ Either run problem or create pickle data file """
    global primal_ele, dofs, k
    arg = sys.argv[1]
    if arg == 'pickle': # Save the pickle file with current specs
       k = sys.argv[2]
       import pickler
       pickler.run(k)
       exit()
    else: # Run specified pickle file
        k = arg
    filename = 'pickles/'  + k + '.p'
    file = open(filename, 'rb')
    my_data = pickle.load(file)
    import data
    data.set_globals(my_data)
    if data.do_adjoint_analysis:
        err_file = open('tables/' +str(data.num_dims) + 'D_' + data.experiment_name+'-'+ data.spaces + '.txt', 'w')
        err_file.write('P' + str(data.lo_u) + 'P' + str(data.lo_p) + 'P' + str(data.lo_B) + '\n')
        err_file.write('use_psi_B=' + str(data.use_psi_B) + ', use_psi_u=' + str(data.use_psi_u) + '\n')
        err_file.write('Re=' + str(data.Re) + ', Rm=' + str(data.Rm) + ', S= ' + str(data.S) + '\n')
        err_file.write('use_lifting=' + str(data.use_lifting) + ', use_periodic=' + str(data.use_periodic)\
                       + ' ,use_true_soln=' + str(data.use_true_soln) + '\n')
        err_file.write('chi_xL={}, chi_xR={}, chi_yB={}, chi_yT={}\n'.format(data.chi_xL, data.chi_xR, data.chi_yB, data.chi_yT))
        err_file.close()
    for (nx, ny, nz) in data.nvals:
        primal_ele = str(nx*ny) # Restricted to 2D problems for now
        run_problem(nx, ny, nz)
    import util
    if data.experiment_name == "Hartmann":
        if rank == 0:
            print("Writing convergences to file")
        util.write_convs_to_file(data.nvals)

if __name__ == "__main__":
    driver()
