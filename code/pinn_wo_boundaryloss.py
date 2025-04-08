import torch
from torch import nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')  # Add backend configuration before pyplot import
import matplotlib.pyplot as plt
from matplotlib import cm


x_train = torch.tensor([[1.1437e-04],
        [1.4676e-01],
        [3.0233e-01],
        [4.1702e-01],
        [7.2032e-01]], dtype=torch.float32)
y_train = torch.tensor([[1.0000],
        [1.0141],
        [1.0456],
        [1.0753],
        [1.1565]], dtype=torch.float32)

R = 1.0
ft0 = 1.0
domain = [0.0, 1.5] #consider interval of t


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_fn=nn.Tanh()):
        super(NN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        out = self.layers(x)
        return out

def df(f, x, order=1):
    df_value = f(x)
    for _ in range(order):
        df_value = torch.autograd.grad(
            df_value,
            x,
            grad_outputs=torch.ones_like(x),
            create_graph=True,
            retain_graph=True,
        )[0]

    return df_value

t = torch.linspace(domain[0], domain[1], steps=10, requires_grad=True).reshape(-1, 1)

def compute_loss(nn, t, x, y):
    pde_loss = df(nn, t) - R * t * (1 - t)
    pde_loss = pde_loss.pow(2).mean()

    boundary = torch.Tensor([0.0])
    boundary.requires_grad = True
    # bc_loss = nn(boundary) - ft0
    # bc_loss = bc_loss.pow(2)

    mse_loss = torch.nn.MSELoss()(nn(x), y)

    tot_loss = mse_loss + pde_loss  # Only keep PDE and data losses

    return tot_loss

model = NN(1, 32, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Train
for ep in range(2000):

    loss = compute_loss(model, t, x_train, y_train)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ep % 100 == 0:
        print(f"epoch: {ep}, loss: {loss.item():>7f}")
        
x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
def logistic_eq_fn(x, y):
    return R * x * (1 - x)

numeric_solution = solve_ivp(
    logistic_eq_fn, domain, [ft0], t_eval=x_eval.squeeze().detach().numpy()
)

f_colloc = solve_ivp(
    logistic_eq_fn, domain, [ft0], t_eval=t.squeeze().detach().numpy()
).y.T


f_eval = model(x_eval)

# Existing plotting section remains unchanged:
# plotting
fig, ax = plt.subplots(figsize=(15, 7))
ax.scatter(t.detach().numpy(), f_colloc, label="Collocation points", color="magenta", alpha=0.75)
ax.scatter(x_train.detach().numpy(), y_train.detach().numpy(), label="Observation data", color="blue")
ax.plot(x_eval.detach().numpy(), f_eval.detach().numpy(), label="NN solution", color="black")
ax.plot(x_eval.detach().numpy(), numeric_solution.y.T,
        label="Analytic solution", color="magenta", alpha=0.75)
ax.set(title="Logistic equation solved with NNs", xlabel="t", ylabel="f(t)")
ax.legend()
# Save before showing
fig.savefig('../figures/pinn_wo_boundaryloss.png', dpi=300, bbox_inches='tight')