"""Supply code for

B. Amirian, H. Jafarzadeh, B. E. Abali, A. Reali, and J. D. Hogan. 

Thermodynamically-consistent derivation and computation of twinning and fracture in 
brittle materials by means of phase-field approaches in the finite element method. 

In: International Journal of Solids and Structures 252 (2022), p. 111789

"""
__author__ = "B. Amirian"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

from dolfin import *
from ufl import indices
import numpy as np
import matplotlib.pyplot as plt

# mesh = UnitSquareMesh(100,100)
# F = m a
# 10^9 nN = 10^24 (zg) * 10^9 nm/10^24 ps^2
# 10^9 GPa = 10^9 nN / nm^2


L = 77
H = 55.
Nx = 400
Ny = 400
mesh = RectangleMesh(Point(0., 0.), Point(L, H), Nx, Ny)

# GL constants
thta = Constant(0.0)
gamma_0 = Constant(0.1295)
AA = Constant(1.404)

Dt = 0.1
t = 0.0
tMax = 500

lmbda = Constant(24.0)             # GPa = nN/nm^2
mu = Constant(19.4)           	   # GPa = nN/nm^2
kappa = Constant(0.0878)           # nJ/m
kappaT = as_tensor([[2*kappa, 0], [0, kappa/2.]])

ss = as_vector(( cos(thta), sin(thta)))
mm = as_vector((-sin(thta), cos(thta)))

# BoundaryU = Expression(('gam*x[1]*t/tMax + 0.08'), tMax = tMax, gam = 0.092, t = 0, degree = 1)

BoundaryU = Expression(('gam*x[1] + 1e-3*t'), gam = 0.07, t = 0, degree = 1)
# BoundaryU = Expression(('gam*x[1]*t/tMax'), gam = 0.8, t = 0, tMax = tMax, degree = 1)

# Function Spaces
V_u = VectorElement('CG', mesh.ufl_cell(), 1) # displacement finite element
V_eta = FiniteElement('CG', mesh.ufl_cell(), 1) # damage finite element
V = FunctionSpace(mesh, MixedElement([V_u, V_eta]))

# Mechanical Boundary Conditions
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)

right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 77.0)

top = CompiledSubDomain("near(x[1], side) && on_boundary", side = 55.0)

bot= CompiledSubDomain("near(x[1], side) && on_boundary", side = 0.0)

bc = []

facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
top.mark(facets, 1)
bot.mark(facets, 2)
left.mark(facets, 3)
right.mark(facets, 4)
ds = Measure("ds", domain = mesh, subdomain_data=facets)

bc.append(DirichletBC(V.sub(0).sub(0), BoundaryU, facets, 1))
bc.append(DirichletBC(V.sub(0).sub(0), BoundaryU, facets, 2))
bc.append(DirichletBC(V.sub(0).sub(0), BoundaryU, facets, 3))
bc.append(DirichletBC(V.sub(0).sub(0), BoundaryU, facets, 4))
bc.append(DirichletBC(V.sub(0).sub(1), Constant(0.0), facets, 1)) 
bc.append(DirichletBC(V.sub(0).sub(1), Constant(0.0), facets, 2))
bc.append(DirichletBC(V.sub(0).sub(1), Constant(0.0), facets, 3))
bc.append(DirichletBC(V.sub(0).sub(1), Constant(0.0), facets, 4))

bc.append(DirichletBC(V.sub(1), Constant(0.0), bot))
bc.append(DirichletBC(V.sub(1), Constant(0.0), top))
bc.append(DirichletBC(V.sub(1), Constant(0.0), right)) 
bc.append(DirichletBC(V.sub(1), Constant(0.0), left)) 

##########################################################
U_ = TestFunction(V)
(u_, eta_) = split(U_)

U_tr = TrialFunction(V)

dU = Function(V)
(du, deta) = split(dU)

Uold = Function(V)
(uold, eta_old) = split(Uold)

delta = Identity(2)
i, j, k, l, p, q, r, s = indices(8)

# Interpolation Function
phi_eta = deta**2*(3-2*deta)

# Reorientation matrix
Q = as_tensor(2*mm[i]*mm[j] - delta[i, j], (i, j))

# Elastic coefficients of the parent phase
C_0 = as_tensor(lmbda * delta[i, j] * delta[k, l] + mu * delta[i, k] *
	delta[j, l] + mu * delta[i, l] * delta[j, k], (i, j, k, l))

# Elastic coefficients of the fully twinned crystals
C_1 = as_tensor(Q[p, i]*Q[q, j]*Q[r, k]*Q[s, l]*C_0[i, j, k, l], (p, q, r, s))

# Elastic coefficients in interfacial regions
C_ = as_tensor(C_0[i, j, k, l] + (C_1[i, j, k, l] - C_0[i, j, k, l])*phi_eta , (i,j,k,l))

# Total strain
eps = as_tensor(1.0/2.0*(du[i].dx(j) + du[j].dx(i)), (i, j))
# eps = sym(grad(du))

# Stress-free strain associated with the twinning
eps_eta = as_tensor(gamma_0*phi_eta*1.0/2.0*(ss[i]*mm[j] + mm[i]*ss[j]), (i, j))

# Elastic strain
eps_el = as_tensor(eps[i, j] - eps_eta[i, j], (i, j))

# Cauchy stress tensor (Anisotropic and Isotropic elasticity)
# sigma = as_tensor(C_[i, j, k, l]*eps_el[k, l], (i, j))
sigma = as_tensor(lmbda*eps[k, k]*delta[i, j]+2.0*mu*eps_el[i, j], (i, j))

# Forming the weak form for order parameter
df0_detta1 = 2*AA*deta - 6*AA*deta**2 + 4*AA*deta**3
df0_detta2 = as_tensor(-1*du[i].dx(j)*(ss[i]*mm[j] + mm[i]*ss[j]), ())
df0_detta3 = gamma_0*phi_eta
df0_detta4 = 6*deta*(1-deta)

df0_detta = df0_detta1 + mu*gamma_0*(df0_detta3+df0_detta2)*df0_detta4

# dF = (df0_detta*eta_ + 2.0*kappa*dot(grad(deta), grad(eta_)))*dx
# 2*kappaT[i,j]*deta.dx(i)*eta_dx(j)
# 2*kappaT[i,j]*deta.dx(j)*eta_dx(i)
# dF = (df0_detta*eta_ + 2.0*inner(kappaT, outer(grad(deta), grad(eta_))))*dx

# dF = (df0_detta*eta_ + 2.0*(kappaT[i, j]*deta.dx(i)*eta_.dx(j)))*dx
dF = (df0_detta*eta_ + 2.0*(kappa*deta.dx(i)*eta_.dx(i)))*dx

# The weak for of the elasticity equation
mech_form = (sigma[i, j]*u_[i].dx(j))*dx

detta_dt = (deta - eta_old)/Dt

res = (detta_dt*eta_)*dx + 4*(dF + mech_form)
Gain = derivative(res, dU, U_tr)

# res = dF + mech_form


# U_init1 = Expression(('0.', '0.', '(x[0] - 38.5)*(x[0] - 38.5) + (x[1] - 27.5)*(x[1] - 27.5) < 4.1*4.1 - tol ? 1.0 : 0'), tol = 1e-3, degree = 2)
x = SpatialCoordinate(mesh)
U_init1 = Expression(('0.', '0.', '35 < (x[0] - 0.001) && (x[0] - 0.001) < 42 && 25.35 < (x[1] - 0.001) && (x[1] - 0.001) < 29.65 ? 1.0 : 0'), tol = 1e-3, degree = 2)
U_init2 = Expression(('0.', '0.', '50 < (x[0] - 0.001) && (x[0] - 0.001) < 57 && 40.0 < (x[1] - 0.001) && (x[1] - 0.001) < 44.3 ? 1.0 : 0'), tol = 1e-3, degree = 2)
# U_init2 = Expression(('0.', '0.', '(x[0] - 5)*(x[0] - 5) + (x[1] - 5)*(x[1] - 5) < 4.0*4.0 - tol ? 1.0 : 0'), tol = 1e-3, degree = 2)
U_init = U_init1 + U_init2

Uold.assign(project(U_init,V))
dU.assign(Uold)

etaFile = File("resultSI11/eta.pvd")
uFile = File("resultSI11/u.pvd")
sigma_file = File("resultSI11/sig.pvd")

while t < tMax:
	uFile << dU.split()[0]
	etaFile << dU.split()[1]
	BoundaryU.t = t
	solve(res==0, dU, bc, J=Gain, solver_parameters = {"newton_solver":{"linear_solver": \
		"cg", "preconditioner": "ilu","relative_tolerance": 1e-6, "absolute_tolerance": 1e-7} }, \
		form_compiler_parameters = {"cpp_optimize": True, \
		"quadrature_degree": 2})
	sigma_project = project(sigma, TensorFunctionSpace(mesh, 'P', 1))
	sigma_file << sigma_project
	Uold.assign(dU)
	t += Dt
