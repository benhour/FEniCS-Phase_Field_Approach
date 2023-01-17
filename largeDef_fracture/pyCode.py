"""Supply code for

B. Amirian, H. Jafarzadeh, B. E. Abali, A. Reali, and J. D. Hogan. 

Thermodynamically-consistent derivation and computation of twinning and fracture in 
brittle materials by means of phase-field approaches in the finite element method. 

In: International Journal of Solids and Structures 252 (2022), p. 111789

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


parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
#parameters["allow_extrapolation"] = True
parameters["mesh_partitioner"] = "SCOTCH"
parameters["ghost_mode"] = 'shared_vertex' #'shared_vertex'  |  'shared_facet'

krylov_params = parameters["krylov_solver"]
krylov_params["relative_tolerance"] = 1E-7
krylov_params["absolute_tolerance"] = 1E-8
#krylov_params["divergence_limit"] = 1E-8
krylov_params["maximum_iterations"] = 150000
krylov_params["nonzero_initial_guess"] = False
krylov_params["monitor_convergence"] = False
krylov_params["error_on_nonconvergence"] = True
krylov_params["report"] = True

#convergence: incremental, residual
solver_parameters = {"nonlinear_solver": "newton", "symmetric": False,
	"newton_solver":{"linear_solver": "mumps", 
						"convergence_criterion": "incremental",
						"relative_tolerance": 1E-4,
						"absolute_tolerance": 1E-5, 
						"krylov_solver": krylov_params, 
						"relaxation_parameter":1.0, 
						"maximum_iterations":100, 
						"error_on_nonconvergence": True
					} 
	}


# mesh = UnitSquareMesh(100,100)
#mesh = Mesh('/home/bamirian/Downloads/benhour/symmetry.xml')
mesh = Mesh('double_notch.xml')
Dim = mesh.geometric_dimension()
mesh_xmin  = 0.
mesh_xmax  = 100.
mesh_ymin  = 0.
mesh_ymax  = 100.

# units : nm, s, g
# derived units: nN, GPa 


###############
Dt = 0.0005
t = 0.0
tMax = 0.3

###########################
thta = 0.0
mu0 = 197.0         # GPa
rs = 10e-5          # Residual stiffness
B = 3.27           # GPa
omega0 = 3.27      # nN
k0 = 248.0          # GPa
lmbda0 = k0 - (2./3.)*mu0     # GPa
betta = 0.0         # Cleavage anisotropy
G = 10.0
LL = 1.0
##################################################
V_u = VectorElement('CG', mesh.ufl_cell(), 1) # displacement finite element
V_psi = FiniteElement('CG', mesh.ufl_cell(), 1) # damage finite element
V = FunctionSpace(mesh, MixedElement([V_u, V_psi]))
dofs = V.dim()

###################################################


facets = MeshFunction("size_t", mesh, Dim-1)
cells = MeshFunction("size_t", mesh, Dim-1)

dA = Measure('ds', domain=mesh, subdomain_data=facets, metadata={'quadrature_degree': 2, "quadrature_scheme": "uflacs"})
dV = Measure('dx', domain=mesh, subdomain_data=cells, metadata={'quadrature_degree': 2, "quadrature_scheme": "uflacs"})

top = CompiledSubDomain("near(x[1], side) && on_boundary", side = mesh_ymax)
bot= CompiledSubDomain("near(x[1], side) && on_boundary", side = mesh_ymin)
left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = mesh_xmin)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = mesh_xmax)

facets.set_all(0)
top.mark(facets, 1)
bot.mark(facets, 2)
left.mark(facets, 3)
right.mark(facets, 4)

bc_list = []
BoundaryU = Expression('t', t = 0, degree = 1)
bc_list.append(DirichletBC(V.sub(0).sub(1), BoundaryU, top))
bc_list.append(DirichletBC(V.sub(0).sub(1), Constant(0.0), bot))

del_Unk = TestFunction(V)
(del_u, del_psi) = split(del_Unk)

dUnk = TrialFunction(V)

Unk = Function(V)
(u, psi) = split(Unk)

Uold = Function(V)
(u_old, psi_old) = split(Uold)

delta = Identity(Dim)
i, j, k, l, p, q, r, s = indices(8)

###################################################
l_psi = rs+(1-rs)*(1-psi)**2
l_psi_prime = -2*(1-rs)*(1-psi)
#mu = mu0*l_psi
#K = k0*l_psi
#lmbda = K - (2./3.)*mu

C_0 = as_tensor(lmbda0 * delta[i, j] * delta[k, l] + mu0 * delta[i, k] * delta[j, l] + mu0 * delta[i, l] * delta[j, k], (i, j, k, l))


'''
F = delta + grad(u)
CC = as_tensor(F[k,i]*F[k,j], (i,j))
E = as_tensor(1./2.*(CC[i,j] - delta[i,j]), (i,j))
S = as_tensor(C_0[i, j, k, l] * E[k, l], (i,j))
P = as_tensor(F[j,l]*S[i,l], (i,j))

W0 = as_tensor(1./2.*E[i,j]*C_0[i,j,k,l]*E[k,l] , ())
#W = W0*l_psi
W_prime = W0*l_psi_prime

Form_u = l_psi*P[k,i]*del_u[i].dx(k)*dV
'''

#stored energy, St.Venant material
def W0(g_u):
	F = delta + g_u
	#J = det(F)
	CC = as_tensor(F[k, i]*F[k, j], (i,j))
	E = as_tensor((1.0/2.0)*(CC[i,j] - delta[i,j]), (i,j))

	return as_tensor(1./2.*E[i,j]*C_0[i,j,k,l]*E[k,l] , ())

W = W0(grad(u))*l_psi
W_prime = W0(grad(u))*l_psi_prime

grad_u = variable(grad(u))
stored_energy = W0(grad_u)

Form_u = l_psi*diff(stored_energy,grad_u)[k,i]*del_u[i].dx(k)*dV


##################################################################

Form_psi = ( (psi - psi_old)/Dt * del_psi + LL*6.0*B*psi*(1.0-psi) * del_psi + W_prime *del_psi  + LL*2.0*omega0*psi.dx(i)*del_psi.dx(i) )*dV

###################################################################

###############################################################
Form = Form_psi + Form_u
Gain = derivative(Form, Unk, dUnk)

problem = NonlinearVariationalProblem(Form, Unk, bcs=bc_list, J=Gain)
solver  = NonlinearVariationalSolver(problem)
solver.parameters.update(solver_parameters)


#Initial condition for eta
U_init = Expression(('0.', '0.', '0.01'), degree = 1)
assign(Uold, project(U_init,V))
assign(Unk, Uold)

##########################################################
psiFile = File("results/psi.pvd")
uFile = File("results/u.pvd")
sigma_file = File("results/sig.pvd")
u_r = 15.0      #nm

while t < tMax:
	u_out, psi_out = Unk.split(deepcopy=True)
	uFile << (u_out,t)
	psiFile << (psi_out,t)
	BoundaryU.t = t*u_r
	
	iternr, converged = solver.solve()
	
	assign(Uold, Unk) # For parallel computing
	t += Dt
	
	elapsed = int(comp_time.time() - start_time)
	e_h, e_m, e_s = int(elapsed/3600), int(elapsed % 3600 / 60), int( ( elapsed % 3600 ) % 60 )
	if processID == 0: print('time: %.4f s with %.0f DOFs in %.0f iterations taking %.0f h %.0f min %.0f s' % (t, dofs, iternr, e_h, e_m, e_s)  )

