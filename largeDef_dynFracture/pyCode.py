"""Supply code for

B. Amirian, B. E. Abali, and J. D. Hogan. 

The study of diffuse interface propagation of dynamic failure in advanced ceramics using the phase-field approach 

In: Computer Methods in Applied Mechanics and Engineering 405 (2023), p. 115862

"""
__author__ = "B. Amirian"
__license__  = "GNU GPL Version 3.0 or later"
#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

import time as comp_time
start_time = comp_time.time()

from dolfin import *
from ufl import indices
import numpy as np
import matplotlib.pyplot as plt

processID =  MPI.comm_world.Get_rank()
set_log_level(50) #50, 40, 30, 20, 16, 13, 10

# mesh = UnitSquareMesh(100,100)
# F = m a
# 10^9 nN = 10^24 (zg) * 10^9 nm/10^24 ps^2
# 10^9 GPa = 10^9 nN / nm^2

mesh = Mesh('.../old_mesh_new_refine.xml')
###############

Dt = 1e-5
t = 0.0
tMax = 0.8

###########################
thta = Constant(0.0)
mu0 = Constant(197.0)         # GPa
rs = Constant(10e-5)          # Residual stiffness
B = Constant(6.54)            # GPa
omega0 = Constant(1.635)       # nN
k0 = Constant(248.0)          # GPa
lmbda0 = k0 - (2./3.)*mu0     # GPa
bta = Constant(100)         # Cleavage anisotropy
G = Constant(10.0)
LL = Constant(1.5)
rho0 = Constant(2.51e-24)
##################################################
V_u = VectorElement('CG', mesh.ufl_cell(), 1) # displacement finite element
V_psi = FiniteElement('CG', mesh.ufl_cell(), 1) # damage finite element
V = FunctionSpace(mesh, MixedElement([V_u, V_psi]))
###################################################

BoundaryU = Expression('t', t = 0, degree = 1)
BoundaryD = Expression('-t', t = 0, degree = 1)
trac = Expression(('0.', '1.3'), degree = 1)
###############################################################################

top = CompiledSubDomain("near(x[1], side) && on_boundary", side = 20.0)

bot= CompiledSubDomain("near(x[1], side) && on_boundary", side = -20.0)

left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = -50.0)

right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 50.0)

bc = []

facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
top.mark(facets, 1)
bot.mark(facets, 2)
left.mark(facets, 3)
right.mark(facets, 4)

ds = Measure("ds", domain = mesh, subdomain_data=facets)
#####################################################################################
U_ = TestFunction(V)
(u_, psi_) = split(U_)

U_tr = TrialFunction(V)

dU = Function(V)
(du, dpsi) = split(dU)

Uold = Function(V)
(uold, psi_old) = split(Uold)

Uoldold = Function(V)
(uoldold, psioldold) = split(Uoldold)

delta = Identity(2)
i, j, k, l, p, q, r, s = indices(8)

###################################################
l_psi = rs+(1-rs)*(1-dpsi)**2
l_psi_prime = -2*(1-rs)*(1-dpsi)
mu = mu0*l_psi
K = k0*l_psi
lmbda = K - (2./3.)*mu

C_0 = as_tensor(lmbda0 * delta[i, j] * delta[k, l] + mu0 * delta[i, k] *
	delta[j, l] + mu0 * delta[i, l] * delta[j, k], (i, j, k, l))

C_ = C_0*l_psi

C_prime = C_0*l_psi_prime

###################################################################

F = as_tensor(delta[i,j] + du[i].dx(j), (i,j))
J = det(F)

# Symemtric Elastic deformation tensor
CC = as_tensor(F[k, i]*F[k, j], (i,j))

E = as_tensor((1.0/2.0)*(CC[i,j] - delta[i,j]), (i,j))

S = as_tensor(C_[i, j, k, l] * E[k, l], (i,j))

P = as_tensor(F[j,l]*S[i,l], (i,j))

##################################################################
mm = as_vector((-sin(thta), cos(thta)))
rhs1 = LL*6*B*dpsi*(1-dpsi)
rhs2 = LL*2.0*omega0*dpsi.dx(i)*psi_.dx(i)
rhs_33 = as_tensor((1.0/2.0)*E[i,j]*C_prime[i,j,k,l]*E[k,l], ())
rhsgT = rhs2
rhsngT = rhs1 + rhs_33
dF_psi = (rhsngT*psi_ + rhsgT)*dx

###################################################################
mech_form = (rho0/Dt/Dt*(du-2.*uold+uoldold)[i]*u_[i])*dx + P[k, i]*u_[i].dx(k)*dx \
- trac[i]*u_[i]*ds(1) + trac[i]*u_[i]*ds(2)

#####################################################################
dpsi_dt = (dpsi - psi_old)/Dt

###############################################################
Form = (dpsi_dt*psi_)*dx + dF_psi + mech_form
Gain = derivative(Form, dU, U_tr)

#############################################################

# bc.append(DirichletBC(V.sub(0).sub(1), BoundaryU, top))
# bc.append(DirichletBC(V.sub(0).sub(1), BoundaryD, bot))


# bc.append(DirichletBC(V.sub(0).sub(1), Constant(0.0), bot))

#Initial condition for eta
U_init = Expression(('0.', '0.', '0.01'), degree = 1)
assign(Uold, project(U_init,V))
assign(Uoldold, Uold)
assign(dU, Uold)

##########################################################
psiFile = File(".../psi.pvd")
uFile = File(".../u.pvd")
u_r = 15.0      #nm

while t < tMax:
	uFile << dU.split()[0]
	psiFile << dU.split()[1]
	BoundaryU.t = t*u_r
	BoundaryD.t = t*u_r
	solve(Form==0, dU, J=Gain, solver_parameters = {"newton_solver":{"linear_solver": \
		"cg", "preconditioner": "jacobi","relative_tolerance": 1e-6, "absolute_tolerance": 1e-7} }, \
		form_compiler_parameters = {"cpp_optimize": True, \
		"quadrature_degree": 2})
	assign(Uoldold, Uold) # For parallel computing
	assign(Uold, dU)
	t += Dt
