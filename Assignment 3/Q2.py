import numpy as np
import gurobipy as gp
import scipy as sp

# Compute the CI for the lower bound
sample_size_lower = 30
batch_size = 10
xi_lower = np.random.poisson(0.5, (batch_size, sample_size_lower))
SAA_vals_lower = np.zeros(batch_size)
SAA_sols_lower = []

for i in range(batch_size):
    SAA_problem_lower = gp.Model('SAA')
    SAA_problem_lower.Params.outputFlag = 0  
    x = SAA_problem_lower.addVar(lb = 0, ub = 5, name = 'x')
    v = SAA_problem_lower.addVars(sample_size_lower, 4, name = 'v')
    SAA_problem_lower.addConstrs(-v[n, 0] + v[n, 1] - v[n, 2] + v[n, 3] == xi_lower[i, n] + (0.5 * x) for n in range(sample_size_lower))
    SAA_problem_lower.addConstrs(-v[n, 0] + v[n, 1] + v[n, 2] - v[n, 3] == 1 + xi_lower[i, n] + (0.25 * x) for n in range(sample_size_lower))
    obj_expr = -0.75 * x + gp.quicksum((1/sample_size_lower) * (-v[n, 0] + 3 * v[n, 1] + v[n, 2] + v[n, 3]) for n in range(sample_size_lower))
    SAA_problem_lower.setObjective(obj_expr, sense = gp.GRB.MINIMIZE)
    SAA_problem_lower.optimize()
    SAA_vals_lower[i] = SAA_problem_lower.ObjVal
    SAA_sols_lower.append(x.X)

lower_avg = np.mean(SAA_vals_lower)
estimated_var = np.sum(np.square(SAA_vals_lower - lower_avg))/(batch_size - 1)
estimated_std = np.sqrt(estimated_var)
t_stat = sp.stats.t.ppf(0.975, batch_size - 1)
LB_L = lower_avg - ((t_stat * estimated_std)/np.sqrt(batch_size))
UB_L = lower_avg + ((t_stat * estimated_std)/np.sqrt(batch_size))
print('95% CI for the lower bound:', [LB_L, UB_L])

# Compute the CI for the upper bound
sample_size_upper = 500
x_best = SAA_sols_lower[np.argmin(SAA_vals_lower)]
xi_upper = np.random.poisson(0.5, sample_size_upper)
SAA_vals_upper = []
for xi in xi_upper:
    SAA_problem_upper = gp.Model('SAA-u')
    SAA_problem_upper.Params.outputFlag = 0
    v = SAA_problem_upper.addVars(4, name = 'v')
    SAA_problem_upper.addConstr(-v[0] + v[1] - v[2] + v[3] == xi + (0.5 * x_best))
    SAA_problem_upper.addConstr(-v[0] + v[1] + v[2] - v[3] == 1 + xi + (0.25 * x_best))
    obj_expr = -0.75 * x_best + (-v[0] + 3 * v[1] + v[2] + v[3])
    SAA_problem_upper.setObjective(obj_expr, sense = gp.GRB.MINIMIZE)
    SAA_problem_upper.optimize()   
    SAA_vals_upper.append(SAA_problem_upper.ObjVal)

SAA_vals_upper = np.array(SAA_vals_upper)
upper_avg = np.mean(SAA_vals_upper)
estimated_var = np.sum(np.square(SAA_vals_upper - upper_avg))/(sample_size_upper - 1)
estimated_std = np.sqrt(estimated_var)
z_stat= sp.stats.norm.ppf(0.975)
LB_U = upper_avg - ((z_stat * estimated_std)/np.sqrt(sample_size_upper))
UB_U = upper_avg + ((z_stat * estimated_std)/np.sqrt(sample_size_upper))
print('95% CI for the upper bound:', [LB_U, UB_U])
print('Optimality Gap:', [LB_L, UB_U])
