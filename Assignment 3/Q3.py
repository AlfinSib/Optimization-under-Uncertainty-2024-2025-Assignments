import numpy as np
import gurobipy as gp

n_scenarios = 100
n_rep = 10
n_eva = 10000
xi = np.random.poisson(0.5, n_scenarios)

V = np.zeros((n_scenarios, n_scenarios))
x_sols = []

F = lambda x, v: -0.75 * x + (-v[0] + 3 * v[1] + v[2] + v[3])

for i in range(n_scenarios):
    OC_P = gp.Model('Opportunity Cost Matrix Generation - 1')
    OC_P.params.OutputFlag = 0
    x = OC_P.addVar(lb = 0, ub = 5, name = 'x')
    v = OC_P.addVars(4, name = 'v')
    OC_P.addConstr(-v[0] + v[1] - v[2] + v[3] == xi[i] + (0.5 * x))
    OC_P.addConstr(-v[0] + v[1] + v[2] - v[3] == 1 + xi[i] + (0.25 * x))
    obj_expr = -0.75 * x + (-v[0] + 3 * v[1] + v[2] + v[3])
    OC_P.setObjective(obj_expr, sense=gp.GRB.MINIMIZE)
    OC_P.optimize()
    x_sols.append(x.X)

for i in range(n_scenarios):
    for j in range(n_scenarios):
        OC_Q = gp.Model('Opportunity Cost Matrix Generation - 2')
        OC_Q.params.OutputFlag = 0
        v = OC_Q.addVars(4, name = 'v')    
        OC_Q.addConstr(-v[0] + v[1] - v[2] + v[3] == xi[j] + (0.5 * x_sols[i]))
        OC_Q.addConstr(-v[0] + v[1] + v[2] - v[3] == 1 + xi[j] + (0.25 * x_sols[i]))
        obj_expr = -0.75 * x_sols[i] + (-v[0] + 3 * v[1] + v[2] + v[3])
        OC_Q.setObjective(obj_expr, sense=gp.GRB.MINIMIZE)
        OC_Q.optimize()
        V[i, j] = OC_Q.ObjVal

CP = gp.Model('Clustering')
CP.params.OutputFlag = 0
t = CP.addVars(n_scenarios, name = 't')
x = CP.addVars(n_scenarios, n_scenarios, vtype=gp.GRB.BINARY, name = 'x')
u = CP.addVars(n_scenarios, vtype=gp.GRB.BINARY, name = 'u')
CP.addConstrs(t[j] >= gp.quicksum(x[i,j] * V[j, i] for i in range(n_scenarios)) - gp.quicksum(x[i,j] * V[j, j] for i in range(n_scenarios)) 
              for j in range(n_scenarios))
CP.addConstrs(t[j] >= gp.quicksum(x[i,j] * V[j, j] for i in range(n_scenarios)) - gp.quicksum(x[i,j] * V[j, i] for i in range(n_scenarios)) \
              for j in range(n_scenarios))
CP.addConstrs(x[i, j] <= u[j] for i in range(n_scenarios) for j in range(n_scenarios))
CP.addConstrs(x[j, j] <= u[j] for j in range(n_scenarios))
CP.addConstr(u.sum() == n_rep)
CP.addConstrs(x.sum(i, '*') == 1 for i in range(n_scenarios))
CP.setObjective(t.sum()/n_scenarios, sense=gp.GRB.MINIMIZE)
CP.optimize()
u_sols = np.array([u[n].X for n in range(n_scenarios)])
xi_rep = xi[np.where(u_sols == 1)[0]]

P = gp.Model('Model using full scenario set')
P.Params.OutputFlag = 0
x = P.addVar(lb = 0, ub = 5, name = 'x')
v = P.addVars(n_scenarios, 4, name = 'v')
P.addConstrs(-v[n, 0] + v[n, 1] - v[n, 2] + v[n, 3] == xi[n] + (0.5 * x) for n in range(n_scenarios))
P.addConstrs(-v[n, 0] + v[n, 1] + v[n, 2] - v[n, 3] == 1 + xi[n] + (0.25 * x) for n in range(n_scenarios))
obj_expr = -0.75 * x + gp.quicksum((1/n_scenarios) * (-v[n, 0] + 3 * v[n, 1] + v[n, 2] + v[n, 3]) for n in range(n_scenarios))
P.setObjective(obj_expr, sense = gp.GRB.MINIMIZE)
P.optimize()
x_sol_full = x.X

P_rep = gp.Model('Model using representative scenario set')
P_rep.Params.OutputFlag = 0
x = P_rep.addVar(lb = 0, ub = 5, name = 'x')
v = P_rep.addVars(n_rep, 4, name = 'v')
P_rep.addConstrs(-v[n, 0] + v[n, 1] - v[n, 2] + v[n, 3] == xi_rep[n] + (0.5 * x) for n in range(n_rep))
P_rep.addConstrs(-v[n, 0] + v[n, 1] + v[n, 2] - v[n, 3] == 1 + xi_rep[n] + (0.25 * x) for n in range(n_rep))
obj_expr = -0.75 * x + gp.quicksum((1/n_rep) * (-v[n, 0] + 3 * v[n, 1] + v[n, 2] + v[n, 3]) for n in range(n_rep))
P_rep.setObjective(obj_expr, sense = gp.GRB.MINIMIZE)
P_rep.optimize()
x_sol_rep = x.X


xi_eva = np.random.poisson(0.5, n_eva)
EP_vals = []
for x in [x_sol_full, x_sol_rep]:
    EP = gp.Model('Evaluation Problem')
    EP.Params.OutputFlag = 0
    v = EP.addVars(n_eva, 4, name = 'v')
    EP.addConstrs(-v[n, 0] + v[n, 1] - v[n, 2] + v[n, 3] == xi_eva[n] + (0.5 * x) for n in range(n_eva))
    EP.addConstrs(-v[n, 0] + v[n, 1] + v[n, 2] - v[n, 3] == 1 + xi_eva[n] + (0.25 * x) for n in range(n_eva))
    obj_expr = -0.75 * x + gp.quicksum((1/n_eva) * (-v[n, 0] + 3 * v[n, 1] + v[n, 2] + v[n, 3]) for n in range(n_eva))
    EP.setObjective(obj_expr, sense = gp.GRB.MINIMIZE)
    EP.optimize()
    EP_vals.append(EP.ObjVal)

print(xi_eva[0], v[0,0], v[0,1], v[0,2], v[0,3], x_sol_rep)

print('Objective Value using full scenarios:', EP_vals[0])
print('Objective Value using representative scenarios:', EP_vals[1])


