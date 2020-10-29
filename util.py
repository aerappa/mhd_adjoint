from fenics import *
from data import *
from numpy import cosh, sinh, inf, nan
import scipy.integrate as integrate

def my_curl(v):
    if num_dims == 2:
        return curl2D(v)
    elif num_dims == 3:
        return curl(v)

def curl_cross_dot(B, C, v):
    if num_dims == 2:
        return curl_cross_dot2D(B, C, v)
    elif num_dims == 3:
        return inner(cross(curl(B), C), v)

def cross_curl_dot(u, B, C):
    if num_dims == 2:
        return cross_curl_dot2D(u, B, C)
    elif num_dims == 3:
        return inner(curl(cross(u, B)), C)

def cross2D(v,w):
    """
    Computes v x w in 2d.
    """
    vx = v[0]; vy = v[1]
    wx = w[0]; wy = w[1]
    return (vx*wy - vy*wx)

def curl2D(v):
    """
    Computed curl(v) in 2d.
    """
    vx = v[0]; vy = v[1]
    return (vy.dx(0) - vx.dx(1))

def curl_cross_dot2D(B, C, v):
    """
    Computes inner((curl B) x C, v) in 2d.
    """
    By = B[1]; Bx = B[0]
    Cy = C[1]; Cx = C[0]
    vy = v[1]; vx = v[0]
    alpha = curl2D(B)
    Tx = -inner(Cy * alpha, vx)
    Ty = inner(Cx*alpha, vy)
    return Tx + Ty

def cross_curl_dot2D(u, B, C):
    """
    Computes inner(curl(u x B), C) in 2d.
    """
    uy = u[1]; ux = u[0]
    By = B[1]; Bx = B[0]
    Cy = C[1]; Cx = C[0]
    alpha = cross2D(u, B)
    Tx = inner(alpha.dx(1), Cx)
    Ty = -inner(alpha.dx(0), Cy)
    return Tx + Ty

class HartmannRHS(UserExpression):
    def eval(self, values, x):
        y = x[1]
        x = x[0]
        if num_dims == 2:
            if use_periodic:
                values[0] = G
            else: values[0] = 0.0
            values[1] = 0.0
        elif num_dims == 3:
            values[0] = 1.0
            values[1] = 0.0
            values[2] = 0.0
    def value_shape(self):
        return num_dims,

class TaylorGreenRHS(UserExpression):
    def eval(self, values, x):
        y = x[1]
        x = x[0]
        values[0] = -cos(x)*(sin(x) + 2*sin(y) - 2/Re*sin(y))
        values[1] = -cos(y)*(2*sin(x) + sin(y) + 2/Re*sin(x))
    def value_shape(self):
        return 2,

class HartmannTrue(UserExpression):
    def eval(self, values, x):
        if num_dims == 3:
            z = x[2]
        else: z = nan
        y = x[1]
        x = x[0]
        if num_dims == 2:
            values[0] = exact_ux(x,y,z)
            values[1] = exact_uy(x,y,z)
            values[2] = exact_pressure(x,y,z)
            values[3] = exact_Bx(x,y,z)
            values[4] = exact_By(x,y,z)
        elif num_dims == 3:
            values[0] = 1.0
            values[1] = 0.0
            values[2] = 0.0
    def value_shape(self):
        return (num_dims*2)+1,

class HartmannVelocity(UserExpression):
    def eval(self, values, x):
        if num_dims == 3:
            values[2] = 0.0
            z = x[2]
        else: z = nan
        y = x[1]
        x = x[0]
        values[0] = exact_ux(x,y,z)
        values[1] = exact_uy(x,y,z)
    def value_shape(self):
        return num_dims,

class HartmannBField(UserExpression):
    def eval(self, values, x):
        if num_dims == 3:
            values[2] = 0.0
            z = x[2]
        else: z = nan
        y = x[1]
        x = x[0]
        values[0] = exact_Bx(x,y,z)
        values[1] = exact_By(x,y,z)
    def value_shape(self):
        return num_dims,

class ExactPressure(UserExpression):
    def eval(self, values, x):
        if num_dims == 3:
            z = x[2]
        else: z = nan
        y = x[1]
        x = x[0]
        values[0] = exact_pressure(x, y, z)
    #def value_shape(self):
        #return 1,

class ExactUX(UserExpression):
    def eval(self, values, x):
        if num_dims == 3:
            z = x[2] 
        else:
            z = nan 
        y = x[1]
        x = x[0]
        values[0] = exact_ux(x,y,z)

class ExactUY(UserExpression):
    def eval(self, values, x):
        if num_dims == 3:
            z = x[2]
        else:
            z = nan
        y = x[1]
        x = x[0]
        values[0] = exact_uy(x,y,z)

class ExactBX(UserExpression):
    def eval(self, values, x):
        if num_dims == 3:
            z = x[2]
        else:
            z = nan
        y = x[1]
        x = x[0]
        values[0] = exact_Bx(x,y,z)

class ExactBY(UserExpression):
    def eval(self, values, x):
        if num_dims == 3:
            z = x[2]
        else:
            z = nan
        y = x[1]
        x = x[0]
        values[0] = exact_By(x,y,z)

def exact_ux(x, y, z):
    return Re*G*(cosh(Ha/2) - cosh(Ha*y))/(2*Ha*sinh(Ha/2))

def exact_uy(x, y, z):
    return 0.0

def exact_Bx(x, y, z):
    return G*(sinh(Ha*y) - 2*sinh(Ha/2)*y)/(2*S*sinh(Ha/2))

def exact_By(x, y, z):
    return 1.0

def exact_pressure(x, y, z):
    if use_periodic:
        return S/2*exact_Bx(x, y, z)**2
    else: return -G*x - S/2*exact_Bx(x, y, z)**2

labels = {'ux' : ExactUX, 'uy' : ExactUY, 'Bx' : ExactBX, 'By' : ExactBY, 'p' : ExactPressure}
def check_exact(computed, label, mesh, rank):
    exact = labels[label](degree=5)
    error = errornorm(exact, computed, mesh=mesh)
    if (rank == 0):
        print('{} error={}'.format(label, error))
    return error

def save(u, B, p, uname, Bname, pname, prefix):
    file = File(prefix + uname + '.pvd')
    file << u
    file = File(prefix + Bname + '.pvd')
    file << B
    file = File(prefix + pname + '.pvd')
    file << p

def get_QoI_err(rank, computed_QofI):
    integrand = lambda x, y, z: psi_ux(x,y,z)*exact_ux(x,y,z) + psi_By(x,y,z)*exact_By(x,y,z)
    if rank == 0:
        print('chi_xL={}, chi_xR={}, chi_yB={}, chi_yT={}'.format( chi_xL, chi_xR, chi_yB, chi_yT))
        if num_dims == 3:
            print('chi_zB={}, chi_zF={}'.format(chi_zB, chi_zF))
    if experiment_name == "lid_driven":
        #chi_xL = -0.25; chi_xR = 0.25
        #chi_yB = -0.5; chi_yT = 0.0
        #true_QofI = -0.008480907953120835 # Re=2000, Rm=0.4 on 400x400 mesh
        #true_QofI = -0.020243275698143375 # Same as above but using Lee stabilization
        if use_psi_u:
            if Re == 2000:
                #true_QofI = -0.020242268611636232 # Same as above but using lifting
                true_QofI = -0.020243205706664956 # Same as above but using P3-P2 for (u,p)
            elif Re == 1000:
                #true_QofI = -0.020242268611636232 # Same as above but using lifting
                true_QofI = -0.02569267778513431 # Same as above but using P3-P2 for (u,p)
            elif Re == 500:
                true_QofI = -0.02943596424436935#Same as above but Re=500
            elif Re == 200:
                true_QofI = -0.03125233683009955 #500x500 mesh
            elif Re == 50:
                true_QofI =-0.03217675059104161
            else:
                true_QofI = nan
        elif use_psi_B:
            if Re == 2000:
                #true_QofI = -0.2498434418116289 # bx
                true_QofI = 0.002937493117393345 # by
            if Re == 1000:
                #true_QofI = -0.24978066334180693 # bx
                true_QofI = 0.003920416098212947 # by
    elif num_dims == 2:
        integrand2 = lambda x, y: integrand(x,y,nan)
        true_QofI = integrate.dblquad(integrand2,xL, xR, yB, yT)[0]
    elif num_dims == 3:
        #true_QofI = integrate.tplquad(integrand, xL, xR, yB, yT, zB, zF, epsabs=1.49e-8, epsrel=1.49e-8)[0]
        true_QofI = -0.00390559523907167
    computed_err = true_QofI - computed_QofI
    if rank == 0:
        print('true QofI={}'.format(true_QofI))
        print('error against reference QofI={}'.format(computed_err))
    return computed_err

def write_convs_to_file(nvals):
    err_file = open('convergences/' + str(num_dims)  + 'D' + experiment_name + spaces + '.txt', 'w')
    err_file.write(str(nvals)+'\n')
    err_file.write('Re=' + str(Re) + ', Rm=' + str(Rm) + ',S=' + str(S) + '\n')
    err_file.write('use_psi_B=' + str(use_psi_B) + ', use_psi_u=' + str(use_psi_u) + '\n')
    err_file.write('ux =' + str(ux_errs)+';\n')
    err_file.write('uy =' + str(uy_errs)+';\n')
    err_file.write('bx =' + str(Bx_errs)+';\n')
    err_file.write('by =' + str(By_errs)+';\n')
    err_file.write('p =' +  str(p_errs)+';\n')

ux_errs = []; uy_errs = []; Bx_errs = []; By_errs = []; p_errs = []
def verify(u0, ud, p0, B0, psi_u, psi_B, mesh, rank):
    if num_dims == 2:
        ux, uy = u0.split()
        Bx, By = B0.split()
    elif num_dims == 3:
        ux, uy, uz = u0.split()
        Bx, By, Bz = B0.split()
    if rank == 0 and do_plotting:
        print('plotting')
        plot_soln(u0, B0, p0)
    save(u0, B0, p0, 'u', 'B', 'p', 'plots/')
    if use_lifting:
        udx, udy = ud.split()
        ux = ux + udx
        computed_QoI = assemble(inner(ux, psi_u)*dx + inner(By, psi_B)*dx)
        ux = project(ux) # For L2 error
    else:
        computed_QoI = assemble(inner(ux, psi_u)*dx + inner(By, psi_B)*dx)
    if do_checking:
        ux_errs.append(check_exact(ux, 'ux', mesh, rank))
        uy_errs.append(check_exact(uy, 'uy', mesh, rank))
        Bx_errs.append(check_exact(Bx, 'Bx', mesh, rank))
        By_errs.append(check_exact(By, 'By', mesh, rank))
    if rank == 0:
        print('computed QofI={}'.format(computed_QoI))
    if do_adjoint_analysis:
        QoI_err = get_QoI_err(rank, computed_QoI)
        return computed_QoI, QoI_err
    else: return None, None

def plot_var(u, u_name):
    import matplotlib.pyplot as plt
    plot(u)
    plt.title(u_name)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.figure()

def plot_soln(u, B, p):
    plot_var(u, 'U')
    plot_var(p, 'P')
    plot_var(B, 'B')
    import matplotlib.pyplot as plt
    plt.show()

def chi(x, y, z):
    """
    Characteristic function on [chi_xL, chi_xR] x [chi_yB, chi_yT] (x [chi_yB, chi_yT] in 3D)
    """
    if (x >= (chi_xL + MACH_EPS) and x <= (chi_xR + MACH_EPS)) and \
       (y >= (chi_yB + MACH_EPS) and y <= (chi_yT + MACH_EPS)):
        ###### Need to check for 3D
        if num_dims == 3:
            if (z >= (chi_zB + MACH_EPS) and z <= (chi_zF + MACH_EPS)):
                return 1.0
            else: return 0.0
        else:
            return 1.0
    else:
        return 0.0

def psi_ux(x, y, z):
    if use_psi_u:
        return chi(x, y, z)
    else:
        return 0.0

def psi_uy(x, y, z):
    return 0.0

def psi_Bx(x, y,z):
    return 0.0

def psi_By(x, y, z):
    if use_psi_B:
        return chi(x, y, z)
    else:
        return 0.0

def psi_p(x, y, z):
    return 0.0

# Data for QoI in velocity
class Psi_u(UserExpression):
    def eval(self, values, x):
        if num_dims == 3:
            z = x[2]
        else:
            z = nan
        y = x[1]
        x = x[0]
        values[0] = psi_ux(x, y, z)
        #values[1] = psi_uy(x, y, z)
    def value_shape(self):
        return ()

# Data for QoI in B field
class Psi_B(UserExpression):
    def eval(self, values, x):
        if num_dims == 3:
            z = x[2]
        else:
            z = nan
        y = x[1]
        x = x[0]
        values[0] = psi_By(x, y, z)
    def value_shape(self):
        return ()
