import numpy as np
import pickle
from numpy import sinh, cosh

MACH_EPS = np.finfo(float).eps

def set_globals(my_data):
    global num_dims, d, xL, xR, yB, yT, zB, zF, chi_xL, chi_xR,\
        chi_yB, chi_yT, chi_zB, chi_zF, Re, Rm, S, Ha,\
        G, lo_u, lo_p, lo_B, ho_u, ho_p, ho_B, use_psi_B,\
        use_psi_u, use_periodic, use_lifting, experiment_name, spaces,\
        do_checking, do_adjoint_analysis, rel_tol, nvals,\
        do_plotting, use_true_soln, do_series_solution,\
        mult
    num_dims = my_data['num_dims']
    d = my_data['d']
    xL = my_data['xL']; xR = my_data['xR']
    yB = my_data['yB']; yT = my_data['yT']
    zB = my_data['zB']; zF = my_data['zF']
    chi_xL = my_data['chi_xL']; chi_xR = my_data['chi_xR']
    chi_yB = my_data['chi_yB']; chi_yT = my_data['chi_yT']
    chi_zB = my_data['chi_zB']; chi_zF = my_data['chi_zF']
    Re = my_data['Re']
    Rm = my_data['Rm']
    S = my_data['S']
    Ha = my_data['Ha']
    G = my_data['G']
    lo_u = my_data['lo_u']; lo_p = my_data['lo_p']; lo_B = my_data['lo_B']
    ho_u = my_data['ho_u']; ho_p = my_data['ho_p']; ho_B = my_data['ho_B']
    do_adjoint_analysis = my_data['do_adjoint_analysis']
    use_psi_B = my_data['use_psi_B']
    use_psi_u = my_data['use_psi_u']
    use_true_soln = my_data['use_true_soln']
    use_periodic = my_data['use_periodic']
    use_lifting = my_data['use_lifting']
    solver_type = my_data['solver_type']
    experiment_name = my_data['experiment_name']
    spaces = my_data['spaces']
    do_checking = my_data['do_checking']
    do_plotting = my_data['do_plotting']
    do_series_solution = my_data['do_series_solution']
    do_adjoint_analysis = my_data['do_adjoint_analysis']
    rel_tol = my_data['rel_tol']
    nvals = my_data['nvals']
    mult = my_data['mult']
