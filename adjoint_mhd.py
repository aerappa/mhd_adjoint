from dolfin import *
import numpy as np
import ufl
import BC_factory
from data import *
from util import *

def C_star(u, uh, phi):
    """" Adjoint term for dC/du where C(u) = u.grad(u) """
    if use_true_soln:
        return 0.5*(ufl.transpose(grad(u+uh))*phi - grad(phi)*(u+uh) - div((u+uh))*phi)
    else:
        return ufl.transpose(grad(uh))*phi - grad(phi)*uh - div(uh)*phi

def ZB_star(beta, u, uh, C):
    """ Adjoint term for dZ/dB where Z(u,B) = curl(u x B) """
    if use_true_soln:
        return 0.5*curl_cross_dot(beta, (u+uh), C) 
    else:
        return curl_cross_dot(beta, uh, C) 
        
def Zu_star(beta, B, Bh, v):
    """ Adjoint term for dZ/du where Z(u,B) = curl(u x B) """
    if use_true_soln:
        return -0.5*curl_cross_dot(beta, (B+Bh), v)
    else:
        return -curl_cross_dot(beta, Bh, v)

def Y_star(phi, B, Bh, C):
    """ Adjoint term for dY/dB where Y(B) = curl(B) x B """
    if use_true_soln:
        return 0.5*(-curl_cross_dot((B+Bh), phi, C) + cross_curl_dot((B+Bh), phi, C))
    else:
        return -curl_cross_dot(Bh, phi, C) + cross_curl_dot(Bh, phi, C)

def get_greens_function(W, adjoint_mesh, uh, Bh, psi_u, psi_B):
    """ Solve for the generalized Green's function in the linearized adjoint problem

        Args:
            W -- Adjoint product space (higher order than primal)
            adjoint_mesh -- Same as primal by default
            uh -- Computed velocity to linearize around
            Bh -- Computed magnetic field to linearize around
            [psi_u, psi_B] -- Representer for Q(U) = (Psi, U)
    """
    # Define variational problem
    BCs = BC_factory.adjoint_BCs(W)
    (Phi, beta) = TrialFunctions(W)
    (phi, pi) = split(Phi)
    (V, C) = TestFunctions(W)
    (v, q) = split(V)
    ### True solution on this mesh
    U = HartmannTrue(degree=20)
    U = interpolate(U, W)
    (U1, B) = U.split()
    (u, p) = U1.split()
    # Momentum equation
    a = 1/Re*inner(grad(phi), grad(v))*dx
    a += div(v)*pi*dx 
    a += inner(C_star(u, uh, phi), v)*dx
    a -= S*Zu_star(beta, B, Bh, v)*dx
    # Continuity
    a -= q*div(phi)*dx
    # Induction
    a -= S*ZB_star(beta, u, uh, C)*dx
    #print("adjoint_mesh={}".format(adjoint_mesh.ufl_cell()))
    a -= S*Y_star(phi, B, Bh, C)*dx
    a += S/Rm * inner(my_curl(beta), my_curl(C))*dx
    ## Exact penalty
    a += S/Rm * inner(div(beta), div(C))*dx
    if num_dims == 2:
        Cx, Cy = split(C)
        vx, vy = split(v)
    else:
        Cx, Cy, Cz = split(C)
        vx, vy, Cz = split(v)
    # We want induced B field, By
    RHS = inner(psi_B, Cy)*dx
    # Nonconstant flow direction
    RHS += inner(psi_u, vx)*dx
    Phi = Function(W)
    solve(a == RHS, Phi, bcs=BCs)
    return Phi
