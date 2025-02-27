import numpy as np
import gurobipy as gp
from ReadData import *

P = gp.Model('City Supply Chain Model')
x = P.addVars(cities, name='x')
u = P.addVars(scenarios, cities, name='u')
v = P.addVars(scenarios, cities, name='v')
z = P.addVars(scenarios, cities, name='z')
s = P.addVars(scenarios, cities, name='s')

P.addConstr(x.sum() <= I)
P.addConstrs(I + u.sum(o, '*') <= x.sum() + v.sum(o, '*') for o in scenarios)
P.addConstrs(Yn[n] + x[n] + v[o, n] + s[o, n] == demand[n, o] + z[o, n] + u[o, n] 
            for n in cities
            for o in scenarios)
fst_stage_obj = 0
for n in cities:
    fst_stage_obj += theta[n] * x[n]
snd_stage_obj = 0
for o in scenarios:
    Q_k = 0
    for n in cities:
        Q_k += theta_s[n] * (u[o, n] + v[o, n]) + h * z[o, n] + g * s[o, n]
    snd_stage_obj += prob * Q_k

P.setObjective(fst_stage_obj + snd_stage_obj, gp.GRB.MINIMIZE)

P.optimize()

x_list = ['x_' + n + ' = ' + str(x[n].X) for n in cities]

print("Optimal Value:", P.ObjVal)
print("Optimal Solution (first-stage):", x_list)
print("Solving Time:", P.Runtime)