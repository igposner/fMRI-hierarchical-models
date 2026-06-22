# %%
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

# %% 
# ---- 1) Your data / knots (input here, like you wanted) ----
x_knot = np.array([0., 8., 30., 75., 100.1], dtype=float)

# IMPORTANT: y_knot must match length of x_knot
# (Your earlier snippet had 6 y values; that won't work.)
y_knot = np.array([1., -1.8, 0., 0., -0.4], dtype=float)

# sort knots (Stan requires increasing)
order = np.argsort(x_knot)
x_knot = x_knot[order]
y_knot = y_knot[order]

# ---- 2) Compile Stan model ----
model = CmdStanModel(stan_file="splineCode.stan")

# ---- 3) Run Stan in fixed_param mode (no sampling; just compute outputs once) ----
stan_data = {
    "K": len(x_knot),
    "x_knot": x_knot,
    "y_knot": y_knot,
    "N_plot": 200,
}

fit = model.sample(
    data=stan_data,
    chains=1,
    iter_sampling=1,
    iter_warmup=0,
    fixed_param=True,
    adapt_engaged=False

)

# ---- 4) Extract results ----
coeffs = fit.stan_variable("coeffs")[0]   # shape (4, K-1)
x_plot = fit.stan_variable("x_plot")[0]
y_plot = fit.stan_variable("y_plot")[0]

print("coeffs (rows: a,b,c,d; cols: segments):")
print(coeffs)

# ---- 5) Plot knots + spline ----
plt.figure()
plt.scatter(x_knot, y_knot, label="knots")
plt.plot(x_plot, y_plot, label="not-a-knot spline")
plt.legend()
plt.show()



# %%
