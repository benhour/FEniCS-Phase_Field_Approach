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

mesh = Mesh('..../mesh.xml')


Dt = 0.1
t = 0.0
tMax = 500


# GL constants
thta = Constant(0.75)  # in radian
gamma_0 = Constant(0.13)
AA = Constant(1.404)
kk = Constant(15.0)

lmbda = Constant(24.0)             # GPa = nN/nm^2
mu = Constant(19.4)           	   # GPa = nN/nm^2
kappa = Constant(0.0878)           # nJ/m
kappaT = as_tensor([[2*kappa, 0], [0, kappa/2.]])

ss = as_vector(( cos(thta), sin(thta)))
mm = as_vector((-sin(thta), cos(thta)))

# BoundaryU = Expression(('t'), t = 0, degree = 1)

BoundaryU = Expression(('gam*x[1] + 1e-3*t'), gam = 0.1, t = 0, degree = 1)

# Twin = Expression('(x[0]-0.5)*(x[0]-0.5) + (x[1])*(x[1]) < 0.8*0.8 ?? 1.0 : 0.0', degree = 1)


def Twin(x):
	return (x[0]-0.5)*(x[0]-0.5) + (x[1])*(x[1]) < 0.8*0.8
	# abs(x[1]) < 1e-03 and x[0] <= 0.0

# Function Spaces
V_u = VectorElement('CG', mesh.ufl_cell(), 2) # displacement finite element
V_eta = FiniteElement('CG', mesh.ufl_cell(), 1) # damage finite element
V = FunctionSpace(mesh, MixedElement([V_u, V_eta]))

# Mechanical Boundary Conditions
top = CompiledSubDomain("near(x[1], side) && on_boundary", side = 50.0)

bot= CompiledSubDomain("near(x[1], side) && on_boundary", side = -50.0)

left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = -50.0)

right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 50.0)

# Twin= CompiledSubDomain("(x[0]-0.5)*(x[0]-0.5) + (x[1])*(x[1]) < 0.8*0.8")


bc = []

facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
top.mark(facets, 1)
bot.mark(facets, 2)
left.mark(facets, 3)
right.mark(facets, 4)

ds = Measure("ds", domain = mesh, subdomain_data=facets)

bc.append(DirichletBC(V.sub(0).sub(1), BoundaryU, facets, 1))
bc.append(DirichletBC(V.sub(0).sub(1), BoundaryU, facets, 2))
#bc.append(DirichletBC(V.sub(0).sub(1), BoundaryU, facets, 3))
#bc.append(DirichletBC(V.sub(0).sub(1), BoundaryU, facets, 4))
bc.append(DirichletBC(V.sub(0).sub(0), Constant(0.0), facets, 1)) 
bc.append(DirichletBC(V.sub(0).sub(0), Constant(0.0), facets, 2))
#bc.append(DirichletBC(V.sub(0).sub(0), Constant(0.0), facets, 3))
#bc.append(DirichletBC(V.sub(0).sub(0), Constant(0.0), facets, 4))

zero_v1 = project(Constant(1.0), V.sub(1).collapse())

# bc.append(DirichletBC(V.sub(0).sub(0), BoundaryU, top))
# bc.append(DirichletBC(V.sub(0).sub(1), Constant(0.0), bot)) 
# bc.append(DirichletBC(V.sub(0).sub(0), Constant(0.0), bot))

# bc.append(DirichletBC(V.sub(0).sub(0), BoundaryU, facets, 1))
# bc.append(DirichletBC(V.sub(0).sub(0), BoundaryU, facets, 2))
# bc.append(DirichletBC(V.sub(0).sub(0), Constant(0.0), facets, 1)) 
# bc.append(DirichletBC(V.sub(0).sub(0), Constant(0.0), facets, 2))
# bc.append(DirichletBC(V.sub(0).sub(1), Constant(0.0), facets, 3))
# bc.append(DirichletBC(V.sub(0).sub(1), Constant(0.0), facets, 4))

bc.append(DirichletBC(V.sub(1), zero_v1, Twin))
# bc.append(DirichletBC(V.sub(1), 1.0, Twin))

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
# phi_eta = 1./(1+exp(-2.0*kk*(deta-0.5)))

# Reorientation matrix
Q = as_tensor(2*mm[i]*mm[j] - delta[i, j], (i, j))

# Elastic coefficients of the parent phase
C_0 = as_tensor(lmbda * delta[i, j] * delta[k, l] + mu * delta[i, k] *
	delta[j, l] + mu * delta[i, l] * delta[j, k], (i, j, k, l))

# Elastic coefficients of the fully twinned crystals
C_1 = as_tensor(Q[p, i]*Q[q, j]*Q[r, k]*Q[s, l]*C_0[i, j, k, l], (p, q, r, s))

# Elastic coefficients in interfacial regions
C_ = as_tensor(C_0[i, j, k, l] + (C_1[i, j, k, l] - C_0[i, j, k, l]) * phi_eta, (i,j,k,l))

# Deformation gradient
F = as_tensor(du[i].dx(j) + delta[i, j], (i, j))

# Stress-free twinning shear
F_eta = as_tensor(delta[i, j] + gamma_0*phi_eta*ss[i]*mm[j], (i, j))

# Elastic deformation
F_E = as_tensor(F[i, j] * inv(F_eta)[j, k], [i, k])

# Symemtric Elastic deformation tensor
C_E = as_tensor(F_E[i, k]*F_E[i, j], (k, j))

# J=def(F)=det(F_E)*det(F_eta)=(J_E)*(J_eta)=(J_E)=(det(C_E))**0.5
J = (det(C_E))**0.5

# Elastic strain tensor
E_E = as_tensor(1./2.*(C_E[k, j] - delta[k, j]), (k, j))

# The symmetric elastic second Piola-Kirchhoff stress tensor
# by considering neo-Hookean material for elastic strain energy density
# S = as_tensor(mu*delta[i, j] + (lmbda*ln(J)-mu)*inv(C_E)[i, j], (i, j))
S = as_tensor(C_[i, j, k, l] * E_E[k, l], (i, j))

# The first Piola-Kirchhoff stress tensor
P = as_tensor(F[i, j]*S[j, k], (k, i))

# Driving (resolved shear stress for twinning)
tau = as_tensor(S[i, j] * C_E[i, k] * ss[k] * mm[l] * inv(F_eta)[l, j], ())

# Surface gradient term (Isotropic and Anisotropic case)
SurGrad = as_tensor(2*kappa*deta.dx(i)*eta_.dx(i), ())
# SurGrad = as_tensor(2*kappaT[i, j]*deta.dx(i)*eta_.dx(j), ())

# The -*RHS of Eq. (40)
Ext = as_tensor(1./2.*E_E[i, j]*(C_1[i, j, k, l] - C_0[i, j, k, l])*E_E[k, l] - gamma_0*tau, ())

# Forming the weak form for order parameter
df0_detta1 = 2*AA*deta - 6*AA*deta**2 + 4*AA*deta**3
df0_detta2 = Ext
df0_detta3 = 6*deta*(1-deta)
# df0_detta3 = 2*kk*exp(-2*kk*(deta - 0.5))/(1+exp(-2*kk*(deta - 0.5)))**2
df0_detta4 = df0_detta2*df0_detta3
df0_detta = df0_detta1 + df0_detta4

dF = (df0_detta*eta_ + SurGrad)*dx

# The weak for of the elasticity equation
mech_form = P[k, i]*u_[i].dx(k)*dx

# The time derivative of order parameter
detta_dt = (deta - eta_old)/Dt

# The total residual form
res = (detta_dt*eta_)*dx + 1.0*(dF + mech_form)

Gain = derivative(res, dU, U_tr)


etaFile = File("resultsLA-TrueMg-ModeI/eta.pvd")
uFile = File("resultsLA-TrueMg-ModeI/u.pvd")
sigma_file = File("resultsLA-TrueMg-ModeI/sig.pvd")

# while t<=1.0:
# 	t+=deltaT
# 	if t>=0.7:
# 		deltaT = 0.0001
# 	BoundaryU.t = t*u_r
# 	uFile << dU.split()[0]
# 	etaFile << dU.split()[1]
# 	solve(res==0, dU, bc, J=Gain, solver_parameters = {"newton_solver":{"linear_solver": \
# 		"cg", "preconditioner": "ilu","relative_tolerance": 1e-6, "absolute_tolerance": 1e-7} }, \
# 		form_compiler_parameters = {"cpp_optimize": True, \
# 		"quadrature_degree": 2})
# 	sigma = as_tensor(1./J*F[j, k]*P[k, i], (j, i))
# 	sigma_project = project(sigma, TensorFunctionSpace(mesh, 'P', 1), solver_type="cg", form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2} )
# 	sigma_file << sigma_project
# 	assign(Uold, dU) # For parallel computing

while t < tMax:
	uFile << dU.split()[0]
	etaFile << dU.split()[1]
	BoundaryU.t = t
	solve(res==0, dU, bc, J=Gain, solver_parameters = {"newton_solver":{"linear_solver": \
		"cg", "preconditioner": "ilu","relative_tolerance": 1e-6, "absolute_tolerance": 1e-7} }, \
		form_compiler_parameters = {"cpp_optimize": True, \
		"quadrature_degree": 2})
	sigma = as_tensor(1./J*F[j, k]*P[k, i], (j, i))
	sigma_project = project(sigma, TensorFunctionSpace(mesh, 'P', 1), solver_type="cg", form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": 2} )
	sigma_file << sigma_project
	assign(Uold, dU) # For parallel computing
	t += Dt
