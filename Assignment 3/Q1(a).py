import numpy as np
import gurobipy as gp
from ReadData import *
import time

# Check for relatively complete recourse


# Start Calculating Time
start = time.time()

# Global Parameters:
tol = 1e-4

# Build Master Problem:
MP = gp.Model("Master Problem")
MP.Params.outputFlag = 0  

x = MP.addVars(cities, vtype=gp.GRB.CONTINUOUS, name='x')
eta = MP.addVars(scenarios, name='eta')
MP.addConstr(x.sum() <= I)
obj_expr_mp = 0
for n in cities:
    obj_expr_mp += x[n] * theta[n]
for s in scenarios:
    obj_expr_mp += prob * eta[s]
MP.setObjective(obj_expr_mp, sense=gp.GRB.MINIMIZE)

# Initialise Sub-problem
SP = gp.Model("Sub-problem")
u = SP.addVars(cities, name='u')
v = SP.addVars(cities, name='v')
z = SP.addVars(cities, name='z')
s = SP.addVars(cities, name='s')
constr_0 = SP.addConstr(u.sum() - v.sum() >= 0)
constrs = SP.addConstrs(v[n] + s[n] - z[n] - u[n] == 0 for n in cities)
obj_expr_sp = 0
for n in cities:
    obj_expr_sp += theta_s[n] * (u[n] + v[n]) + h * z[n] + g * s[n]
SP.setObjective(obj_expr_sp, sense=gp.GRB.MINIMIZE)
SP.Params.outputFlag = 0


# Build/Update and Solve Sub-problem for each scenario:
def subproblem_update(s, x_sol, eta_sol):
    constr_0.rhs = sum(x_sol.values()) - I
    for n in cities:
        constrs[n].rhs = demand[n, s] - x_sol[n] - Yn[n]
    SP.update()
    SP.optimize()
    pi_0 = constr_0.Pi
    pi = {n:constrs[n].Pi for n in cities}
    obj_val = SP.objVal

    # Check whether a violated Benders cut is found
    cut_found = False
    if eta_sol[s] < obj_val - tol: # Found Benders cut is violated at the current master solution
        cut_found = True

    return obj_val, cut_found, pi_0, pi

# Main Loop
cut_found = True
iter = 0
best_upper = gp.GRB.INFINITY
best_lower = -gp.GRB.INFINITY
n_cuts = 0
while(cut_found):
    iter += 1
    cut_found = False
    MP.update()
    MP.optimize()
    obj_mp = MP.objVal
    x_sol = {n:x[n].x for n in cities}
    eta_sol = {s:eta[s].x for s in scenarios}
    UB = sum([theta[n] * x_sol[n] for n in cities])
    
    for s in scenarios:
        Q_val, cut_found_s, pi_0_sol, pi_sol = subproblem_update(s, x_sol, eta_sol)
        UB += prob * Q_val
        if(cut_found_s):
            cut_found = True
            n_cuts += 1
            MP.addConstr(eta[s] - (x.sum() - I) * pi_0_sol - gp.quicksum((demand[n, s] - x[n] - Yn[n]) * pi_sol[n] for n in cities) >= 0)
    print('Lower & Upper Bound at Iteration ' + str(iter) + ':', [obj_mp, UB])
    if UB < best_upper:
        best_upper = UB
    if obj_mp > best_lower:
        best_lower = obj_mp
    if iter > 200:
        print("Unable to finish within 200 iterations")
        print("Best Upper Bound:", best_upper)
        print('Best Lower Bound:', best_lower)

print('\nOptimal Value: %g' % obj_mp)
print('Optimal Solution (first-stage):')
print("x_sol: " + str(x_sol))
print('Total Number of Cuts Added:', n_cuts)
print("No of Iterations: " + str(iter))
print("Total Time:", time.time() - start)

