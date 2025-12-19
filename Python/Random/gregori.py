import numpy as np
import matplotlib.pyplot as plt

def plot_domains(N_values):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    for N in N_values:
        # Define domain boundaries for D_A^(A)
        x_A = [1/N, 1, 1, 1/N, 1/N]
        y_A = [1/N, 1/N, 2, 2, 1/N]
        
        # Define domain boundaries for D_A^(B)
        x_B = [1/N, 1, 1, 1/N, 1/N]
        y_B = [1/N**2, 1/N**2, 2/N**2, 2/N**2, 1/N**2]
        
        # Plot D_A^(A)
        axes[0].plot(x_A, y_A, label=f'N = {N}')
        axes[0].fill(x_A, y_A, alpha=0.3)
        
        # Plot D_A^(B)
        axes[1].plot(x_B, y_B, label=f'N = {N}')
        axes[1].fill(x_B, y_B, alpha=0.3)
    
    # Formatting
    axes[0].set_xlim(0, 1.1)
    axes[0].set_ylim(0, 0.1)
    axes[0].set_title(r'$D_A^{(A)}$')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlim(0, 1.1)
    axes[1].set_ylim(0, 0.1)
    axes[1].set_title(r'$D_A^{(B)}$')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
N_values = [5, 10, 15]  # You can change these values
plot_domains(N_values)
