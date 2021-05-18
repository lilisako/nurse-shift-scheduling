from mip import (CONTINUOUS, Model, OptimizationStatus, maximize, minimize)

m = Model()
x = m.add_var("x")
y = m.add_var("y")

m.objective = maximize(100 * x + 100 * y)
m += x + 2 * y <= 16
m += 3 * x + y <= 18
m.optimize()
print(x.x,y.x)