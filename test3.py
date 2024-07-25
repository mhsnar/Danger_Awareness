import mosek
import cvxpy as cp

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()

with mosek.Env() as env:
    env.set_Stream(mosek.streamtype.log, streamprinter)
    # Set the license path
    env.set_licensedir(r'C:\Users\mohsen.amiri\mosek\mosek.lic')

# Your optimization problem setup
x = cp.Variable(10, integer=True)
constraints = [0 <= x, x <= 1]
objective = cp.Maximize(cp.sum(x))
problem = cp.Problem(objective, constraints)

# Solve the problem using MOSEK
problem.solve(solver=cp.MOSEK, verbose=True)
print("Solution status:", problem.status)
print("Optimal value:", problem.value)
print("Optimal solution:", x.value)
