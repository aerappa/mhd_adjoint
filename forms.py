from data import *
from fenics import *
from util import *

def T0(u, v):
    return inner(grad(u)*u,v)*dx

def T1(u, v):
    return 1/Re*inner(grad(u), grad(v))*dx

def T2(p, v):
    return -p*div(v)*dx

def T3(B, v):
    return -S*curl_cross_dot(B, B, v)*dx

def T4(u, q):
    return q*div(u)*dx

def T5(B, C):
    return S/Rm * inner(my_curl(B), my_curl(C))*dx

def T6(u, B, C):
    return -S*cross_curl_dot(u, B, C)*dx

def T7(B, C):
    return S/Rm * inner(div(B), div(C))*dx

def assemble_terms(u, p, B, phi, pi, beta, adjoint_mesh):
    #plot_div_error(u, pi, adjoint_mesh)
    _T0 = assemble(T0(u, phi))
    _T1 = assemble(T1(u, phi))
    _T2 = assemble(T2(p, phi))
    _T3 = assemble(T3(B, phi))
    _T4 = assemble(T4(u, pi))
    _T5 = assemble(T5(B, beta))
    _T6 = assemble(T6(u, B, beta))
    _T7 = assemble(T7(B, beta))
    terms = {0:_T0, 1:_T1, 2:_T2, 3:_T3, 4:_T4, 5:_T5, 6:_T6, 7:_T7}
    return terms

def assemble_full_form(u, p, B, V):
    """
    Construct the full nonlinear form for the MHD equations
    with the exact penalty stabilization.
    """
    (V1, C) = split(V)
    (v, q) = split(V1)
    ### Momentum equation
    N = assemble_fluid(u, v, p, q, B)
    N += assemble_maxwell(u, B, C)
    return N

def assemble_fluid(u, v, p, q, B):
    NS = assemble_momentum(u, v, p, B)
    NS += assemble_continuity(u, q)
    return NS

def assemble_navier(u, p, V):
    (v, q) = split(V)
    return T0(u, v) + T1(u, v) + T2(p, v) + T4(u, q)

def assemble_momentum(u, v, p, B):
    return T0(u, v) + T1(u, v) + T2(p, v) + T3(B, v)

def assemble_continuity(u, q):
    ### Continuity equation
    return T4(u, q)

def assemble_maxwell(u, B, C):
    M = assemble_induction(u, B, C)
    M += assemble_penalty(B, C)
    return M

def assemble_induction(u, B, C):
    return T5(B, C) + T6(u, B, C)

def assemble_penalty(B, C):
    return T7(B, C)
