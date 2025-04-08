import torch
from torch import nn
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('TkAgg')  # Add backend configuration before pyplot import
import matplotlib.pyplot as plt
from matplotlib import cm

# Training data
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

# Parameters
R = 1.0  
ft0 = 1.0 
domain = [0.0, 1.5] #consider interval of t

# Improved neural network with more layers and dropout
class ImprovedNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act_fn=nn.Tanh(), dropout_rate=0.1):
        super(ImprovedNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Dropout(dropout_rate),
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

# Adaptive sampling for collocation points
def adaptive_sampling(model, domain, n_points, n_iterations=5):
    # Start with uniform sampling
    t = torch.linspace(domain[0], domain[1], steps=n_points, requires_grad=True).reshape(-1, 1)
    
    for _ in range(n_iterations):
        # Compute PDE residual
        with torch.no_grad():
            # Forward pass to get predictions
            f_t = model(t)
            
            # Compute derivatives manually for residual
            dt = 0.01 * (domain[1] - domain[0])
            t_plus = t + dt
            t_minus = t - dt
            f_t_plus = model(t_plus)
            f_t_minus = model(t_minus)
            
            # Central difference approximation for derivative
            df_dt = (f_t_plus - f_t_minus) / (2 * dt)
            
            # Compute residual
            pde_residual = df_dt - R * t * (1 - t)
            residual_norm = pde_residual.pow(2).mean(dim=1)
            
            # Sample more points where residual is high
            indices = torch.multinomial(residual_norm, n_points, replacement=True)
            t = t[indices]
            
            # Add some noise to avoid clustering
            t = t + torch.randn_like(t) * 0.01 * (domain[1] - domain[0])
            t = torch.clamp(t, domain[0], domain[1])
            t.requires_grad_(True)
    
    return t

# Compute loss with adaptive weights
def compute_loss(nn, t, x, y, bc_weight=1.0, pde_weight=1.0, data_weight=1.0):
    # PDE loss
    pde_loss = df(nn, t) - R * t * (1 - t)
    pde_loss = pde_loss.pow(2).mean()
    
    # Boundary condition loss
    bc_loss = (nn(torch.tensor([[domain[0]]], dtype=torch.float32)) - ft0).pow(2)
    
    # Data loss
    data_loss = torch.nn.MSELoss()(nn(x), y)
    
    # Total loss with weights
    tot_loss = pde_weight * pde_loss + bc_weight * bc_loss + data_weight * data_loss
    
    return tot_loss, pde_loss, bc_loss, data_loss

# Train with learning rate scheduling
def train_model(model, x_train, y_train, domain, n_epochs=2000, n_collocation=100):
    # Initialize optimizer with learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100, verbose=True
    )
    
    # Initial collocation points
    t = torch.linspace(domain[0], domain[1], steps=n_collocation, requires_grad=True).reshape(-1, 1)
    
    # Training history
    history = {
        'total_loss': [], 'pde_loss': [], 'bc_loss': [], 'data_loss': []
    }
    
    # Train
    for ep in range(n_epochs):
        # Adaptive sampling every 100 epochs
        if ep % 100 == 0 and ep > 0:
            t = adaptive_sampling(model, domain, n_collocation)
        
        # Compute loss
        tot_loss, pde_loss, bc_loss, data_loss = compute_loss(
            model, t, x_train, y_train, 
            bc_weight=1.0, 
            pde_weight=1.0, 
            data_weight=1.0
        )
        
        # Backpropagation
        optimizer.zero_grad()
        tot_loss.backward()
        optimizer.step()
        
        # Update learning rate
        scheduler.step(tot_loss)
        
        # Record history
        history['total_loss'].append(tot_loss.item())
        history['pde_loss'].append(pde_loss.item())
        history['bc_loss'].append(bc_loss.item())
        history['data_loss'].append(data_loss.item())
        
        if ep % 100 == 0:
            print(f"epoch: {ep}, total loss: {tot_loss.item():>7f}, "
                  f"pde loss: {pde_loss.item():>7f}, "
                  f"bc loss: {bc_loss.item():>7f}, "
                  f"data loss: {data_loss.item():>7f}")
    
    return history

# Evaluate model
def evaluate_model(model, domain):
    x_eval = torch.linspace(domain[0], domain[1], steps=100).reshape(-1, 1)
    
    # Analytic solution
    def logistic_eq_fn(x, y):
        return R * x * (1 - x)
    
    numeric_solution = solve_ivp(
        logistic_eq_fn, domain, [ft0], t_eval=x_eval.squeeze().detach().numpy()
    )
    
    # Neural network solution
    f_eval = model(x_eval)
    
    return x_eval, f_eval, numeric_solution

# Plot results
def plot_results(x_eval, f_eval, numeric_solution, x_train, y_train, title, filename):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.scatter(x_train.detach().numpy(), y_train.detach().numpy(), label="Observation data", color="blue")
    ax.plot(x_eval.detach().numpy(), f_eval.detach().numpy(), label="NN solution", color="black")
    ax.plot(x_eval.detach().numpy(), numeric_solution.y.T,
            label="Analytic solution", color="magenta", alpha=0.75)
    ax.set(title=title, xlabel="t", ylabel="f(t)")
    ax.legend()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Plot training history
def plot_history(history, filename):
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(history['total_loss'], label='Total Loss')
    ax.plot(history['pde_loss'], label='PDE Loss')
    ax.plot(history['bc_loss'], label='BC Loss')
    ax.plot(history['data_loss'], label='Data Loss')
    ax.set(title='Training History', xlabel='Epoch', ylabel='Loss')
    ax.legend()
    ax.set_yscale('log')
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# Main function
def main():
    # Create improved model
    model = ImprovedNN(1, 64, 1, dropout_rate=0.1)
    
    # Train model
    history = train_model(model, x_train, y_train, domain, n_epochs=2000, n_collocation=100)
    
    # Evaluate model
    x_eval, f_eval, numeric_solution = evaluate_model(model, domain)
    
    # Plot results
    plot_results(x_eval, f_eval, numeric_solution, x_train, y_train, 
                "Logistic equation solved with improved PINN", 'pinn_improved.png')
    
    # Plot training history
    plot_history(history, 'pinn_improved_history.png')
    
    # Print final losses
    print(f"Final total loss: {history['total_loss'][-1]:>7f}")
    print(f"Final PDE loss: {history['pde_loss'][-1]:>7f}")
    print(f"Final BC loss: {history['bc_loss'][-1]:>7f}")
    print(f"Final data loss: {history['data_loss'][-1]:>7f}")

if __name__ == "__main__":
    main() 