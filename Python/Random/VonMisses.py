import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises_fisher
from matplotlib.colors import Normalize
n_grid = 100
u = np.linspace(0, np.pi, n_grid)
v = np.linspace(0, 2 * np.pi, n_grid)
u_grid, v_grid = np.meshgrid(u, v)
vertices = np.stack([np.cos(v_grid) * np.sin(u_grid),
                     np.sin(v_grid) * np.sin(u_grid),
                     np.cos(u_grid)],
                    axis=2)
x = np.outer(np.cos(v), np.sin(u))
y = np.outer(np.sin(v), np.sin(u))
z = np.outer(np.ones_like(u), np.cos(u))
def plot_vmf_density(ax, x, y, z, vertices, mu, kappa):
    vmf = vonmises_fisher(mu, kappa)
    pdf_values = vmf.pdf(vertices)
    # print(np.shape(vertices))
    pdfnorm = Normalize(vmin=pdf_values.min(), vmax=pdf_values.max())
    # print(pdfnorm(pdf_values),np.shape(pdfnorm(pdf_values)),len(x[:,0]))
    ax.plot_surface(x, y, z, rstride=1, cstride=1,
                    facecolors=plt.cm.viridis(pdfnorm(pdf_values)),
                    linewidth=0)
    ax.set_aspect('equal')
    ax.view_init(azim=-130, elev=0)
    ax.axis('off')
    ax.set_title(rf"$\kappa={kappa}$")
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(9, 4),
                         subplot_kw={"projection": "3d"})
left, middle, right = axes
mu = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])
plot_vmf_density(left, x, y, z, vertices, mu, 5)
plot_vmf_density(middle, x, y, z, vertices, mu, 20)
plot_vmf_density(right, x, y, z, vertices, mu, 100)
plt.subplots_adjust(top=1, bottom=0.0, left=0.0, right=1.0, wspace=0.)
# plt.show()

rng = np.random.default_rng()
mu = np.array([0, 0, 1])
samples = vonmises_fisher(mu, 20).rvs(5, random_state=rng)
# print(samples)

def plot_vmf_samples(ax, x, y, z, mu, kappa):
    vmf = vonmises_fisher(mu, kappa)
    samples = vmf.rvs(20)
    ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0,
                    alpha=0.2)
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='k', s=5)
    ax.scatter(mu[0], mu[1], mu[2], c='r', s=30)
    ax.set_aspect('equal')
    ax.view_init(azim=-130, elev=0)
    ax.axis('off')
    ax.set_title(rf"$\kappa={kappa}$")
mu = np.array([-np.sqrt(0.5), -np.sqrt(0.5), 0])
fig, axes = plt.subplots(nrows=1, ncols=3,
                         subplot_kw={"projection": "3d"},
                         figsize=(9, 4))
left, middle, right = axes

