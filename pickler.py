import numpy as np
import pickle
import sys
from numpy import sinh, cosh, nan, pi

num_dims = 2
# Flags
do_adjoint_analysis = False
use_psi_B = True
use_psi_u = False
use_true_soln = False
use_periodic = False
use_lifting = True
do_plotting = False
do_checking = False
do_series_solution = False

experiment_name = "lid_driven"
### Width of domain
d = 0.5
solver_type = 'Newton'
xL = -d; xR = d
yB = -d; yT = d
zB = -d; zF = d
chi_xL = -0.25; chi_xR = 0.5
chi_yB = -0.5; chi_yT = 0.25
### Characteristic function for QoI
chi_zB = zB; chi_zF = zF
### Reynolds number
Re = 16.0
### Magnetic Reynolds number
Rm = 16.0
### Coupling coefficient
S = 1.0
### Hartmann number
Ha = np.sqrt(S * Re * Rm)
### Pressure drop, G = -dp/dx
if num_dims == 2:
    G = 2*Ha*sinh(Ha/2)/(Re*(cosh(Ha/2) - 1)) # Normalized so that max ||u|| = 1
else:
    G = 1.0
### Degrees of mixed low order space
lo_u = 2; lo_p = 1; lo_B = 1;
### Degrees of mixed high order space
ho_u = 3; ho_p = 2; ho_B = 2;
### Tolerance for nonlinear solver
rel_tol = 1e-10
#nvals = [(32, 32, nan), (64, 64, nan), (128, 128, nan), (256, 256, nan)]
nvals = [(40, 40, nan), (80, 80, nan), (120,120, nan), (160, 160, nan)]
mult = 1 # For adjoint mesh. Not used in parallel
#nvals = [(128, 128, nan)]
if num_dims == 3:
    #nvals = [(16, 16, 16)]
    nvals = [(8, 8, 8)]
spaces = "P{}_P{}_P{}".format(lo_u, lo_p, lo_B)
if experiment_name == "lid_driven":
    if do_adjoint_analysis:
        nvals = [(40, 40, nan), (60, 60, nan), (80,80, nan), (100, 100, nan)]
    else:
        nvals = [(128, 128, nan)]
    use_lifting = False
    use_true_soln = False
    use_periodic = False
    do_checking = False
    Re = 5000
    #Rm = np.finfo(float).eps
    Rm = 5.0
    S = 1.0
    chi_xL = -0.25; chi_xR = 0.25
    chi_yB = 0.0; chi_yT = 0.5
if experiment_name == "TaylorGreen":
    h = 2*pi
    xL = 0; xR = h
    yB = 0; yT = h
    use_psi_u = True
    use_psi_B = False
    chi_xL = 0.25*h; chi_xR = 1.0*h
    chi_yB = 0.2*h; chi_yT = 0.5*h
#######################################################################
my_data ={
    'num_dims':num_dims,
    'd':d,
    'solver_type':'Newton',
    'xL':xL, 'xR':xR,
    'yB':yB, 'yT':yT,
    'zB':zB, 'zF':zF,
    'chi_xL':chi_xL, 'chi_xR':chi_xR,
    'chi_yB':chi_yB, 'chi_yT':chi_yT,
    'chi_zB':chi_zB, 'chi_zF':chi_zF,
    'Re':Re, 'Rm':Rm, 'S':S, 'Ha':Ha, 'G':G,
    'lo_u':lo_u, 'lo_p':lo_p, 'lo_B':lo_B,
    'ho_u':ho_u, 'ho_p':ho_p, 'ho_B':ho_B,
    'rel_tol':rel_tol,
    'do_adjoint_analysis':do_adjoint_analysis,
    'use_psi_B':use_psi_B, 'use_psi_u':use_psi_u,
    'use_true_soln':use_true_soln,
    'use_periodic':use_periodic,
    'use_lifting':use_lifting,
    'do_plotting':do_plotting, 'do_checking':do_checking,
    'do_series_solution':do_series_solution,
    'nvals':nvals,
    'mult':mult,
    'spaces':spaces,
    'experiment_name':experiment_name,
    }

def run(k):
    print('Running pickler')
    #print(my_data['num_dims'])
    pickle_name = 'pickles/' + k + '.p'
    pickle_file = open(pickle_name, 'wb')
    pickle.dump(my_data, pickle_file)
