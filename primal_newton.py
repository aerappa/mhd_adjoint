from dolfin import *
from pathlib import Path
from data import *
import forms
import BC_factory
import numpy as np

# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()


def solve_exact_penalty(W, RHS, ut, ud, Bt, xele, yele):
    """
    Assemble and then solve the primal problem

        Args:
            W -- The primal monolithic product space
            RHS -- The momentum source term, f
            ut -- The true velocity to be used for Dirichlet BCs without lifting
            ud -- The known Dirichlet BC function for lifting
            Bt -- the the magnetic field for Dirichlet BCs
            [xele, yele] -- Used to print the saved state for lid driven homotopy
    """
    W_fluid = W.sub(0); W_mag = W.sub(1)
    zero = Constant((0.0, 0.0))
    if use_lifting:
        # Plug in zero for BCs on u
        BCs = BC_factory.primal_BCs(W_fluid, W_mag, zero, Bt)
    else:
        BCs = BC_factory.primal_BCs(W_fluid, W_mag, ut, Bt)
    # The block is done to load the saved state as a starting guess
    # for the nonlinear solver
    if experiment_name == "lid_driven":
        saved_state = 'init_guesses/P' + str(lo_u) + 'P' + str(lo_p) + 'P' + str(lo_B)\
                    + '_Rm=' + str(Rm) + '_' + str(xele) + '-' + str(yele) + '.xml'
        my_file = Path(saved_state)
        try: 
            # Hack to generate an exception if the file doesn't exist
            my_file.resolve(strict=True)
            U = Function(W, saved_state)
            print(saved_state + " found!")
        except FileNotFoundError:
            print(saved_state + " not found")
            U = Function(W)
    else:
        U = Function(W)
    V = TestFunction(W)
    (U1, B) = split(U)
    (u, p) = split(U1)
    if use_lifting:
        N = forms.assemble_full_form(u+ud, p, B, V) # Add the known Dirichlet function
    else:
        N = forms.assemble_full_form(u, p, B, V)
    (V1, C) = split(V)
    (v, q) = split(V1)
    N -= inner(RHS,v)*dx
    solve(N == Constant(0.0), U, BCs, solver_parameters={"newton_solver": {"relative_tolerance": rel_tol,\
            "krylov_solver":{"absolute_tolerance": 1e-8}}})
    if experiment_name == "lid_driven":
        file = File(saved_state)
        file << U
    (U1, B) = U.split()
    (u, p) = U1.split()
    return (u, p, B)
