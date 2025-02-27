import numpy as np
import gurobipy as gp

seat_type = np.array(['Eco', 'Business', 'FstClass'])
scenarios = range(4)
prob = [0.4, 0.3, 0.2, 0.1]
demand = np.array([[200, 60, 25],
                   [180, 40, 20],
                   [175, 25, 10],
                   [150, 10, 5]])
demand_mean = prob[0] * demand[0,:] + prob[1] * demand[1,:] + prob[2] * demand[2,:] + prob[3] * demand[3,:]

P = gp.Model('Seat Allocation 2SP')
n = P.addVars(seat_type, name = 'n', vtype=gp.GRB.INTEGER)
s = P.addVars(scenarios, seat_type, name = 's', vtype=gp.GRB.INTEGER)

P.addConstr(n['Eco'] + 1.5 * n['Business'] + 2 * n['FstClass'] <= 200)
for o in scenarios:
    P.addConstrs(s[o, i] <= n[i] for i in seat_type)
    P.addConstrs(s[o, i] <= demand[o, np.where(seat_type == i)[0]] for i in seat_type)

obj_expr = 0
for o in scenarios:
    obj_expr += prob[o] * (s[o, 'Eco'] + 2 * s[o, 'Business'] + 3 * s[o, 'FstClass'])

P.setObjective(obj_expr, sense=gp.GRB.MAXIMIZE)

P.optimize()

P_mean = gp.Model('Mean Value Model')
n_mean = P_mean.addVars(seat_type, name = 'n_mean', vtype=gp.GRB.INTEGER)
s_mean = P_mean.addVars(seat_type, name = 's_mean', vtype=gp.GRB.INTEGER)
P_mean.addConstr(n_mean['Eco'] + 1.5 * n_mean['Business'] + 2 * n_mean['FstClass'] <= 200)
P_mean.addConstrs(s_mean[i] <= n_mean[i] for i in seat_type)
P_mean.addConstrs(s_mean[i] <= demand_mean[np.where(seat_type == i)[0]] for i in seat_type)
P_mean.setObjective(s_mean['Eco'] + 2 * s_mean['Business'] + 3 * s_mean['FstClass'], sense=gp.GRB.MAXIMIZE)
P_mean.optimize()

P_mean_fst = gp.Model('Model using MV First-stage Solution')
s_mean_fst = P_mean_fst.addVars(scenarios, seat_type, name = 's_mean_fst', vtype=gp.GRB.INTEGER)
for o in scenarios:
    P_mean_fst.addConstrs(s_mean_fst[o, i] <= n_mean[i].X for i in seat_type)
    P_mean_fst.addConstrs(s_mean_fst[o, i] <= demand[o, np.where(seat_type == i)[0]] for i in seat_type)

obj_expr_mean_fst = 0
for o in scenarios:
    obj_expr_mean_fst += prob[o] * (s_mean_fst[o, 'Eco'] + 2 * s_mean_fst[o, 'Business'] + 3 * s_mean_fst[o, 'FstClass'])

P_mean_fst.setObjective(obj_expr_mean_fst, sense=gp.GRB.MAXIMIZE)
P_mean_fst.optimize()

EVPI = 0
for o in scenarios:
    P_o = gp.Model('Single Scenario Model')
    n_o = P_o.addVars(seat_type, name = 'n_o', vtype=gp.GRB.INTEGER)
    s_o = P_o.addVars(seat_type, name = 's_o', vtype=gp.GRB.INTEGER)
    P_o.addConstr(n_o['Eco'] + 1.5 * n_o['Business'] + 2 * n_o['FstClass'] <= 200)
    P_o.addConstrs(s_o[i] <= n_o[i] for i in seat_type)
    P_o.addConstrs(s_o[i] <= demand[o, np.where(seat_type == i)[0]] for i in seat_type)
    P_o.setObjective(s_o['Eco'] + 2 * s_o['Business'] + 3 * s_o['FstClass'], sense=gp.GRB.MAXIMIZE)
    P_o.optimize()
    EVPI += prob[o] * P_o.ObjVal
EVPI += - P.ObjVal

print("Optimal Value:", P.ObjVal)
print("Solution Order: Eco | Business |FstClass")
print("Optimal Solution (first-stage):", [n[i].X for i in seat_type])
print("Optimal Solution (second-stage):")
for o in scenarios:
    print('Scenario '+str(o)+':', [s[o, i].X for i in seat_type])
print('VSS:', P.ObjVal - P_mean_fst.ObjVal)
print('EVPI:', EVPI)

    
