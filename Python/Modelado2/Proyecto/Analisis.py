import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import kstest, norm
from scipy.stats import ortho_group
from scipy.special import iv
import warnings
import os
from sphere.distribution import fb8
import  Distributions as distri

warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

def save_all_plots():
    plots_dir = "analysis_plots"
    os.makedirs(plots_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 15))

    plt.subplot(3, 4, 1)
    plt.hist(s1, bins=50, alpha=0.7, density=True)
    plt.title('Distribución de Stokes 1')
    plt.xlabel('Stokes 1')
    plt.ylabel('Densidad')

    plt.subplot(3, 4, 2)
    plt.hist(s2, bins=50, alpha=0.7, density=True)
    plt.title('Distribución de Stokes 2')
    plt.xlabel('Stokes 2')

    plt.subplot(3, 4, 3)
    plt.hist(s3, bins=50, alpha=0.7, density=True)
    plt.title('Distribución de Stokes 3')
    plt.xlabel('Stokes 3')

    plt.subplot(3, 4, 4)
    plt.hist(df['theta'], bins=50, alpha=0.7, density=True)
    plt.title('Distribución de Theta')
    plt.xlabel('Theta (rad)')

    plt.subplot(3, 4, 5)
    plt.hist(df['phi'], bins=50, alpha=0.7, density=True)
    plt.title('Distribución de Phi')
    plt.xlabel('Phi (rad)')

    plt.subplot(3, 4, 6)
    plt.hist(np.sqrt(s1**2+s2**2+s3**2), bins=50, alpha=0.7, density=True)
    plt.title('Distribución del DOP')
    plt.xlabel('DOP (%)')

    from mpl_toolkits.mplot3d import Axes3D
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    ax = fig.add_subplot(3, 4, 7, projection='3d')
    ax.scatter(x, y, z, alpha=0.6, s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Esfera de Poincaré')

    plt.subplot(3, 4, 8)
    plt.scatter(s3,df['Azimuth [�]'], alpha=0.5, s=1)
    plt.xlabel('Azimuth (°)')
    plt.ylabel('Ellipticidad (°)')
    plt.title('Azimuth vs Ellipticidad')

    plt.subplot(3, 4, 9)
    plt.plot(times, s1, alpha=0.7, label='Stokes 1')
    plt.plot(times, s2, alpha=0.7, label='Stokes 2')
    plt.plot(times, s3, alpha=0.7, label='Stokes 3')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Valor Stokes')
    plt.title('Evolución Temporal')
    plt.legend()

    plt.subplot(3, 4, 10)
    stats.probplot(s1, dist="norm", plot=plt)
    plt.title('QQ-plot Stokes 1')

    plt.subplot(3, 4, 11)
    stats.probplot(df['theta'], dist="norm", plot=plt)
    plt.title('QQ-plot Theta')

    plt.subplot(3, 4, 12)
    stats.probplot(df['phi'], dist="norm", plot=plt)
    plt.title('QQ-plot Phi')

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/descriptive_analysis.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    corr_matrix = df[['Time Stamp [s]','Stokes 1', 'Stokes 2', 'Stokes 3', 'Azimuth [�]']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Matriz de Correlación')
    plt.savefig(f"{plots_dir}/correlation_matrix.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].hist(kappa_bootstrap, bins=30, alpha=0.7, density=True)
    axes[0, 0].axvline(kappa_mean, color='red', linestyle='--', label=f'Media: {kappa_mean:.3f}')
    axes[0, 0].axvline(ci_lower_k, color='orange', linestyle=':', label=f'IC95%: [{ci_lower_k:.3f}, {ci_upper_k:.3f}]')
    axes[0, 0].axvline(ci_upper_k, color='orange', linestyle=':')
    axes[0, 0].set_xlabel('Kappa')
    axes[0, 0].set_ylabel('Densidad')
    axes[0, 0].set_title('Distribución Bootstrap de Kappa')
    axes[0, 0].legend()

    axes[0, 1].plot(segment_df['segment'], segment_df['mean_R'], 'o-', markersize=8)
    axes[0, 1].set_xlabel('Segmento Temporal')
    axes[0, 1].set_ylabel('R (Vector Resultante)')
    axes[0, 1].set_title('Evolución de la Concentración')
    axes[0, 1].grid(True, alpha=0.3)

    residuals_s1 = s1 - np.mean(s1)
    axes[1, 0].hist(residuals_s1, bins=50, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Residuales Stokes 1')
    axes[1, 0].set_ylabel('Densidad')
    axes[1, 0].set_title('Distribución de Residuales')

    theta_deg = np.degrees(df['theta'])
    phi_deg = np.degrees(df['phi']) % 360
    ax_polar = fig.add_subplot(2, 2, 4, projection='polar')
    scatter = ax_polar.scatter(phi_deg * np.pi / 180, theta_deg, c=df['DOP [%]'], alpha=0.6, cmap='viridis', s=2)
    ax_polar.set_title('Distribución Angular')
    plt.colorbar(scatter, ax=ax_polar, label='DOP (%)')

    plt.tight_layout()
    plt.savefig(f"{plots_dir}/summary_plots.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    return plots_dir

def save_results_to_txt():
    results_file = "analysis_results.txt"

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("""
                         ██████╗  ██████╗ ████████╗███████╗    ██╗   ██╗██╗███████╗
                        ██╔════╝ ██╔═══██╗╚══██╔══╝██╔════╝    ██║   ██║██║██╔════╝
                        ██║  ███╗██║   ██║   ██║   ███████╗    ██║   ██║██║███████╗
                        ██║   ██║██║   ██║   ██║   ╚════██║    ██║   ██║██║╚════██║
                        ╚██████╔╝╚██████╔╝   ██║   ███████║    ╚██████╔╝██║███████║
                         ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝     ╚═════╝ ╚═╝╚══════╝
                        """)
        f.write("=== RESULTADOS DEL ANÁLISIS DE POLARIZACIÓN ===\n\n")

        f.write("1. INFORMACIÓN GENERAL\n")
        f.write(f"Número de mediciones: {len(df)}\n")
        f.write(f"Duración del experimento: {df['Time Stamp [s]'].iloc[-1]:.1f} segundos\n\n")

        f.write("2. ESTADÍSTICAS DESCRIPTIVAS\n")
        f.write(f"Radio medio (DOP): {np.mean(r):.6f}\n")
        f.write(f"Theta medio: {np.mean(theta):.6f} rad\n")
        f.write(f"Phi medio: {np.mean(phi):.6f} rad\n")
        f.write(f"DOP promedio: {np.mean(df['DOP [%]']):.4f}%\n\n")

        f.write("3. PARÁMETROS ESTIMADOS\n")
        if 'estimated_params' in locals():
            f.write(f"Kappa (concentración): {estimated_params[0]:.4f}\n")
            f.write(f"Beta1: {estimated_params[1]:.4f}\n")
            f.write(f"Beta2: {estimated_params[2]:.4f}\n")
            f.write(f"Dirección media: [{estimated_params[3]:.4f}, {estimated_params[4]:.4f}, {estimated_params[5]:.4f}]\n")
        f.write(f"Kappa (bootstrap): {kappa_mean:.4f}\n")
        f.write(f"IC95% Kappa: [{ci_lower_k:.4f}, {ci_upper_k:.4f}]\n\n")

        f.write("4. PRUEBAS DE BONDAD DE AJUSTE\n")
        f.write(f"Theta - KS: {ks_theta.statistic:.4f}, p-value: {ks_theta.pvalue:.4f}\n")
        f.write(f"Phi - KS: {ks_phi.statistic:.4f}, p-value: {ks_phi.pvalue:.4f}\n")
        f.write(f"DOP - KS: {ks_dop.statistic:.4f}, p-value: {ks_dop.pvalue:.4f}\n")
        f.write(f"Stokes 1 - Shapiro: {sw_stokes1[0]:.4f}, p-value: {sw_stokes1[1]:.4f}\n")
        f.write(f"Theta - Shapiro: {sw_theta[0]:.4f}, p-value: {sw_theta[1]:.4f}\n\n")

        f.write("5. INTERVALOS DE CONFIANZA (BOOTSTRAP)\n")
        f.write(f"Stokes 1 - Media: {mean_stokes1:.6f}, IC95%: [{ci_lower_s1:.6f}, {ci_upper_s1:.6f}]\n")
        f.write(f"DOP - Media: {mean_dop:.4f}%, IC95%: [{ci_lower_dop:.4f}, {ci_upper_dop:.4f}]\n\n")

        f.write("6. PRUEBAS DE HIPÓTESIS\n")
        f.write(f"Estadístico de Rayleigh: {rayleigh_stat:.4f}\n")
        f.write(f"R (vector resultante): {R:.4f}\n")
        f.write(f"Prueba de isotropía: χ² = {chi2_stat:.4f}, p-value = {p_value_isotropy:.6f}\n")
        f.write(f"Conclusión isotropía: {'NO isotrópica' if p_value_isotropy < 0.05 else 'Posiblemente isotrópica'}\n\n")

        f.write("7. DISTRIBUCIÓN KENT\n")
        if 'kappa_est' in locals():
            f.write(f"Kappa estimado: {kappa_est:.4f}\n")
            f.write(f"Beta estimado: {beta_est:.4f}\n")
            f.write(f"R² total: {fit_metrics['total_r2']:.6f}\n")
            f.write(f"R² media: {fit_metrics['mean_fit']:.6f}\n")
            f.write(f"R² varianza: {fit_metrics['variance_fit']:.6f}\n")
            f.write(f"Ratio elipticidad observado: {fit_metrics['ellipticity_ratio']:.4f}\n")
            f.write(f"Ratio elipticidad teórico: {theoretical_ellipticity:.4f}\n\n")

        f.write("8. INTERPRETACIÓN FÍSICA\n")
        if kappa_mean > 1:
            f.write("• Existe dirección preferencial en la polarización\n")
            f.write("• El sistema muestra memoria de polarización\n")
        else:
            f.write("• La polarización es esencialmente aleatoria\n")
            f.write("• El LED se comporta como fuente no polarizada\n")

        if 'estimated_params' in locals() and estimated_params[0] > 0.5:
            f.write("• Los datos muestran características compatibles con Fisher-Bingham\n")
            f.write("• Existe concentración direccional significativa\n")
        else:
            f.write("• Los datos no muestran fuerte concentración direccional\n")
            f.write("• Podría aproximarse mejor con distribución uniforme esférica\n")

    return results_file

print("=== ANÁLISIS DE DATOS DE POLARIZACIÓN ===\n")

df = pd.read_csv('DiodoLedLVL1ntensity.csv', skiprows=22)
time_df = pd.read_csv('times.csv')
times = time_df['Time'].values

print("1. ESTRUCTURA DE LOS DATOS:")
print(f"Número de mediciones: {len(df)}")
print(f"Columnas disponibles: {list(df.columns)}")
print("\nPrimeras 5 filas:")
print(df.head())

print("\n" + "=" * 50)
print("2. ESTADÍSTICAS DESCRIPTIVAS")

stats_desc = df[['Time Stamp [s]','Stokes 1', 'Stokes 2', 'Stokes 3', 'Azimuth [�]', 'Ellipticity [�]']].describe()
print(stats_desc)

print("\n" + "=" * 50)
print("3. TRANSFORMACIÓN A COORDENADAS ESFÉRICAS")

def stokes_to_spherical(s1, s2, s3):
    r = np.sqrt(s1 ** 2 + s2 ** 2 + s3 ** 2)
    theta = np.arccos(s3 / r)
    phi = np.arctan2(s2, s1)
    return r, theta, phi

s1 = df['Time Stamp [s]'].values
s2 = df['Stokes 1'].values
s3 = df['Stokes 2'].values
print('Parámetro de Stokes S3 = ',s3)
r, theta, phi = stokes_to_spherical(s1, s2, s3)

df['r'] = r
df['theta'] = theta
df['phi'] = phi

print(f"Radio medio (DOP): {np.mean(r):.6f}")
print(f"Theta medio: {np.mean(theta):.6f} rad")
print(f"Phi medio: {np.mean(phi):.6f} rad")

print("\n" + "=" * 50)
print("4. ANÁLISIS VISUAL DESCRIPTIVO")

fig = plt.figure(figsize=(20, 15))

plt.subplot(3, 4, 1)
plt.hist(s1, bins=50, alpha=0.7, density=True)
plt.title('Distribución de Stokes 1')
plt.xlabel('Stokes 1')
plt.ylabel('Densidad')

plt.subplot(3, 4, 2)
plt.hist(s2, bins=50, alpha=0.7, density=True)
plt.title('Distribución de Stokes 2')
plt.xlabel('Stokes 2')

plt.subplot(3, 4, 3)
plt.hist(s3, bins=50, alpha=0.7, density=True)
plt.title('Distribución de Stokes 3')
plt.xlabel('Stokes 3')

plt.subplot(3, 4, 4)
plt.hist(df['theta'], bins=50, alpha=0.7, density=True)
plt.title('Distribución de Theta')
plt.xlabel('Theta (rad)')

plt.subplot(3, 4, 5)
plt.hist(df['phi'], bins=50, alpha=0.7, density=True)
plt.title('Distribución de Phi')
plt.xlabel('Phi (rad)')

plt.subplot(3, 4, 6)
plt.hist(np.sqrt(s1**2 + s2**2 + s3**2), bins=50, alpha=0.7, density=True)
plt.title('Distribución del DOP')
plt.xlabel('DOP (%)')

from mpl_toolkits.mplot3d import Axes3D
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

ax = fig.add_subplot(3, 4, 7, projection='3d')
ax.scatter(x, y, z, alpha=0.6, s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Esfera de Poincaré')

plt.subplot(3, 4, 8)
plt.scatter(df['Stokes 3'], df['Azimuth [�]'], alpha=0.5, s=1)
plt.xlabel('Azimuth (°)')
plt.ylabel('Elipticidad (°)')
plt.title('Azimuth vs Elipticidad')

plt.subplot(3, 4, 9)
plt.plot(times, s1, alpha=0.7, label='Stokes 1')
plt.plot(times, s2, alpha=0.7, label='Stokes 2')
plt.plot(times, s3, alpha=0.7, label='Stokes 3')
plt.xlabel('Tiempo (s)')
plt.ylabel('Valor Stokes')
plt.title('Evolución Temporal')
plt.legend()

plt.subplot(3, 4, 10)
stats.probplot(s1, dist="norm", plot=plt)
plt.title('QQ-plot Stokes 1')

plt.subplot(3, 4, 11)
stats.probplot(df['theta'], dist="norm", plot=plt)
plt.title('QQ-plot Theta')

plt.subplot(3, 4, 12)
stats.probplot(df['phi'], dist="norm", plot=plt)
plt.title('QQ-plot Phi')

plt.tight_layout()
plt.show()

print("\n" + "=" * 50)
print("5. ANÁLISIS DE CORRELACIÓN")

corr_matrix = df[['Time Stamp [s]','Stokes 1', 'Stokes 2', 'Stokes 3', 'Azimuth [�]']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlación')
plt.show()

print("\n" + "=" * 50)
print("6. ESTIMACIÓN DE PARÁMETROS - DISTRIBUCIÓN FISHER-BINGHAM")

def fisher_bingham_log_likelihood(params, data):
    kappa, beta1, beta2, mu1, mu2, mu3 = params
    mu = np.array([mu1, mu2, mu3])
    mu = mu / np.linalg.norm(mu)
    concentration_term = kappa * np.dot(data, mu)
    log_likelihood = np.sum(concentration_term)
    return -log_likelihood

unit_vectors = np.column_stack([x / r, y / r, z / r])

initial_params = [1.0, 0.1, 0.1, np.mean(unit_vectors[:, 0]), np.mean(unit_vectors[:, 1]), np.mean(unit_vectors[:, 2])]

try:
    result = minimize(fisher_bingham_log_likelihood, initial_params, args=(unit_vectors,), method='L-BFGS-B', bounds=[(0.1, 10), (-1, 1), (-1, 1), (-1, 1), (-1, 1), (-1, 1)])
    estimated_params = result.x
    print(f"Parámetros estimados Fisher-Bingham:")
    print(f"  Kappa: {estimated_params[0]:.4f}")
    print(f"  Beta1: {estimated_params[1]:.4f}")
    print(f"  Beta2: {estimated_params[2]:.4f}")
    print(f"  Dirección media: [{estimated_params[3]:.4f}, {estimated_params[4]:.4f}, {estimated_params[5]:.4f}]")
except Exception as e:
    print(f"Error: {e}")
    mean_vector = np.mean(unit_vectors, axis=0)
    kappa_hat = (3 * np.linalg.norm(mean_vector) - np.linalg.norm(mean_vector) ** 3) / (1 - np.linalg.norm(mean_vector) ** 2)
    print(f"Kappa: {kappa_hat:.4f}")
    print(f"Dirección media: {mean_vector}")

print("\n" + "=" * 50)
print("7. PRUEBAS DE BONDAD DE AJUSTE")

print("\nPrueba de Kolmogorov-Smirnov:")
ks_theta = kstest(df['theta'], 'norm', args=(np.mean(df['theta']), np.std(df['theta'])))
print(f"Theta - KS: {ks_theta.statistic:.4f}, p-value: {ks_theta.pvalue:.4f}")

phi_normalized = df['phi'] % (2 * np.pi)
ks_phi = kstest(phi_normalized, 'uniform', args=(0, 2 * np.pi))
print(f"Phi - KS: {ks_phi.statistic:.4f}, p-value: {ks_phi.pvalue:.4f}")

ks_dop = kstest(np.sqrt(s1**2 + s2**2 + s3**3), 'norm', args=(np.mean(np.sqrt(s1**2+s2**2+s3**2)), np.std(np.sqrt(s1**2+s2**2+s3**2))))
print(f"DOP - KS: {ks_dop.statistic:.4f}, p-value: {ks_dop.pvalue:.4f}")

print("\nPrueba de Shapiro-Wilk:")
sw_stokes1 = stats.shapiro(s1)
print(f"Stokes 1 - Shapiro: {sw_stokes1[0]:.4f}, p-value: {sw_stokes1[1]:.4f}")

sw_theta = stats.shapiro(df['theta'])
print(f"Theta - Shapiro: {sw_theta[0]:.4f}, p-value: {sw_theta[1]:.4f}")

print("\n" + "=" * 50)
print("8. ANÁLISIS DE REMUESTREO BOOTSTRAP")

def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=1000, ci=95):
    bootstrap_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_sample = data[indices]
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)
    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrap_stats, alpha)
    upper = np.percentile(bootstrap_stats, 100 - alpha)
    return np.mean(bootstrap_stats), lower, upper, bootstrap_stats

print("\nIntervalos de confianza al 95%:")
mean_stokes1, ci_lower_s1, ci_upper_s1, _ = bootstrap_confidence_interval(s1, np.mean)
print(f"Stokes 1 - Media: {mean_stokes1:.6f}, IC95%: [{ci_lower_s1:.6f}, {ci_upper_s1:.6f}]")

mean_dop, ci_lower_dop, ci_upper_dop, _ = bootstrap_confidence_interval(np.sqrt(s1**2+s2**2+s3**2), np.mean)
print(f"DOP - Media: {mean_dop:.4f}%, IC95%: [{ci_lower_dop:.4f}, {ci_upper_dop:.4f}]")

def estimate_kappa(vectors):
    mean_vec = np.mean(vectors, axis=0)
    R = np.linalg.norm(mean_vec)
    if R < 0.97:
        kappa = (3 * R - R ** 3) / (1 - R ** 2)
    else:
        kappa = 1 / (1 - R)
    return kappa

unit_vectors = np.column_stack([x / r, y / r, z / r])
kappa_mean, ci_lower_k, ci_upper_k, kappa_bootstrap = bootstrap_confidence_interval(unit_vectors, estimate_kappa)
print(f"Kappa - Media: {kappa_mean:.4f}, IC95%: [{ci_lower_k:.4f}, {ci_upper_k:.4f}]")

print("\n" + "=" * 50)
print("9. PRUEBAS DE HIPÓTESIS")

print("\nHipótesis: Distribución isotrópica vs direccional")
R = np.linalg.norm(np.mean(unit_vectors, axis=0))
n = len(unit_vectors)
rayleigh_stat = n * R ** 2
print(f"Estadístico de Rayleigh: {rayleigh_stat:.4f}")
print(f"R: {R:.4f}")

chi2_stat = 3 * n * R ** 2
p_value_isotropy = 1 - stats.chi2.cdf(chi2_stat, 3)
print(f"Prueba de isotropía: χ² = {chi2_stat:.4f}, p-value = {p_value_isotropy:.6f}")

if p_value_isotropy < 0.05:
    print("→ Distribución NO isotrópica")
else:
    print("→ Distribución posiblemente isotrópica")

print("\n" + "=" * 50)
print("10. ANÁLISIS DE ESTABILIDAD TEMPORAL")

n_segments = 4
segment_size = len(df) // n_segments
segment_stats = []
for i in range(n_segments):
    start_idx = i * segment_size
    end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(df)
    segment = df.iloc[start_idx:end_idx]
    segment_vector = np.mean(unit_vectors[start_idx:end_idx], axis=0)
    segment_R = np.linalg.norm(segment_vector)
    segment_stats.append({
        'segment': i + 1,
        'time_start': times[0],
        'time_end': times[-1],
        'mean_R': segment_R,
        'mean_theta': np.mean(segment['theta']),
        'mean_phi': np.mean(segment['phi']),
        'mean_dop': np.mean(np.sqrt(s1**2+s2**2+s3**2))
    })

segment_df = pd.DataFrame(segment_stats)
print("\nEstadísticas por segmento temporal:")
print(segment_df.round(4))

print("\n" + "=" * 50)
print("11. CONCLUSIONES Y RESUMEN")

print(f"\nRESUMEN:")
print(f"• Mediciones: {len(df)}")
print(f"• Duración: {times[-1]:.1f} s")
print(f"• DOP promedio: {np.mean(np.sqrt(s1**2+s2**2+s3**2)):.4f}%")
print(f"• Kappa: {kappa_mean:.4f}")

print(f"\nHALLADOS:")
print(f"1. Distribución: {'NO isotrópica' if p_value_isotropy < 0.05 else 'posible isotrópica'}")
print(f"2. Normalidad:")
print(f"   - Stokes 1: {'normal' if sw_stokes1[1] > 0.05 else 'no normal'}")
print(f"   - Theta: {'normal' if sw_theta[1] > 0.05 else 'no normal'}")

print(f"\nINTERPRETACIÓN FÍSICA:")
if kappa_mean > 1:
    print("• Dirección preferencial en polarización")
    print("• Sistema con memoria de polarización")
else:
    print("• Polarización aleatoria")
    print("• LED como fuente no polarizada")

print(f"\nAJUSTE A FISHER-BINGHAM:")
if 'estimated_params' in locals():
    if estimated_params[0] > 0.5:
        print("• Compatible con Fisher-Bingham")
        print("• Concentración direccional significativa")
    else:
        print("• Baja concentración direccional")
        print("• Posible distribución uniforme")
else:
    print("• Baja concentración")
    print("• Distribución cercana a uniforme")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].hist(kappa_bootstrap, bins=30, alpha=0.7, density=True)
axes[0, 0].axvline(kappa_mean, color='red', linestyle='--', label=f'Media: {kappa_mean:.3f}')
axes[0, 0].axvline(ci_lower_k, color='orange', linestyle=':', label=f'IC95%: [{ci_lower_k:.3f}, {ci_upper_k:.3f}]')
axes[0, 0].axvline(ci_upper_k, color='orange', linestyle=':')
axes[0, 0].set_xlabel('Kappa')
axes[0, 0].set_ylabel('Densidad')
axes[0, 0].set_title('Bootstrap Kappa')
axes[0, 0].legend()

axes[0, 1].plot(segment_df['segment'], segment_df['mean_R'], 'o-', markersize=8)
axes[0, 1].set_xlabel('Segmento')
axes[0, 1].set_ylabel('R')
axes[0, 1].set_title('Concentración Temporal')
axes[0, 1].grid(True, alpha=0.3)

residuals_s1 = s1- np.mean(s1)
axes[1, 0].hist(residuals_s1, bins=50, alpha=0.7, density=True)
axes[1, 0].set_xlabel('Residuales S1')
axes[1, 0].set_ylabel('Densidad')
axes[1, 0].set_title('Residuales Stokes 1')

theta_deg = np.degrees(df['theta'])
phi_deg = np.degrees(df['phi']) % 360
ax_polar = fig.add_subplot(2, 2, 4, projection='polar')
scatter = ax_polar.scatter(phi_deg * np.pi / 180, theta_deg, c=np.sqrt(s1**2+s2**2+s3**2), alpha=0.6, cmap='viridis', s=2)
ax_polar.set_title('Distribución Angular')
plt.colorbar(scatter, ax=ax_polar, label='DOP (%)')

plt.tight_layout()
plt.show()

def kent_constant_approximation(kappa, beta):
    if kappa <= 0:
        return 1.0
    term1 = 2 * np.pi * np.exp(kappa) / kappa
    term2 = (1 - np.exp(-2 * kappa))**(-1/2)
    term3 = (1 + 0.25 * beta**2 / kappa**2)
    return term1 * term2 * term3

def kent_log_likelihood(params, data):
    kappa, beta = params[0], params[1]
    gamma = np.array([
        [params[2], params[3], params[4]],
        [params[5], params[6], params[7]],
        [params[8], params[9], params[10]]
    ])
    identity_approx = gamma @ gamma.T
    ortho_penalty = np.sum((identity_approx - np.eye(3))**2)
    gamma1 = gamma[0]
    gamma2 = gamma[1]
    gamma3 = gamma[2]
    term1 = kappa * np.dot(data, gamma1)
    term2 = beta * (np.dot(data, gamma2)**2 - np.dot(data, gamma3)**2)
    log_likelihood = np.sum(term1 + term2)
    penalty = 1000 * ortho_penalty
    log_c = kappa - 0.5 * np.log(kappa) if kappa > 1 else 0
    return -(log_likelihood - len(data) * log_c - penalty)

def fit_kent_distribution(data, max_iter=100):
    n = len(data)
    mean_direction = np.mean(data, axis=0)
    mean_direction = mean_direction / np.linalg.norm(mean_direction)
    cov_matrix = np.cov(data.T)
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    kappa_init = 3 * eigvals[0] / (eigvals[1] + eigvals[2]) if (eigvals[1] + eigvals[2]) > 0 else 5.0
    beta_init = 0.5 * kappa_init * (eigvals[1] - eigvals[2]) / (eigvals[1] + eigvals[2]) if (eigvals[1] + eigvals[2]) > 0 else 0.1
    gamma_init = eigvecs.flatten()
    initial_params = np.concatenate([[kappa_init, beta_init], gamma_init])
    bounds = [(0.1, 50), (0, 24.9)] + [(-1, 1)] * 9
    try:
        result = minimize(kent_log_likelihood, initial_params, args=(data,), method='L-BFGS-B', bounds=bounds, options={'maxiter': max_iter})
        if result.success:
            return result.x, result.fun
        else:
            return initial_params, np.inf
    except Exception as e:
        return initial_params, np.inf

print("=== ESTIMACIÓN CORREGIDA DISTRIBUCIÓN KENT ===")

unit_vectors = np.column_stack([x/r, y/r, z/r])
params, neg_log_lik = fit_kent_distribution(unit_vectors)

kappa_est, beta_est = params[0], params[1]
gamma_est = params[2:].reshape(3, 3)

print(f"Parámetros estimados:")
print(f"Kappa: {kappa_est:.4f}")
print(f"Beta: {beta_est:.4f}")
print(f"Gamma:")
print(gamma_est)

ortho_check = gamma_est @ gamma_est.T
print(f"Ortogonalidad:")
print(ortho_check)

def kent_goodness_of_fit(params, data):
    kappa, beta = params[0], params[1]
    gamma = params[2:].reshape(3, 3)
    gamma1, gamma2, gamma3 = gamma[0], gamma[1], gamma[2]
    proj_gamma1 = np.dot(data, gamma1)
    proj_gamma2 = np.dot(data, gamma2)
    proj_gamma3 = np.dot(data, gamma3)
    mean_gamma1_obs = np.mean(proj_gamma1)
    var_gamma2_obs = np.var(proj_gamma2)
    var_gamma3_obs = np.var(proj_gamma3)
    mean_gamma1_exp = 1 - 1/kappa if kappa > 1 else 0
    var_gamma2_exp = 1/kappa + beta/kappa**2
    var_gamma3_exp = 1/kappa - beta/kappa**2
    r2_mean = (mean_gamma1_obs - mean_gamma1_exp)**2
    r2_var2 = (var_gamma2_obs - var_gamma2_exp)**2
    r2_var3 = (var_gamma3_obs - var_gamma3_exp)**2
    total_r2 = r2_mean + r2_var2 + r2_var3
    return {
        'total_r2': total_r2,
        'mean_fit': r2_mean,
        'variance_fit': r2_var2 + r2_var3,
        'ellipticity_ratio': var_gamma2_obs / var_gamma3_obs if var_gamma3_obs > 0 else np.inf
    }

fit_metrics = kent_goodness_of_fit(params, unit_vectors)
print(f"\nBondad de ajuste:")
print(f"R² total: {fit_metrics['total_r2']:.6f}")
print(f"R² media: {fit_metrics['mean_fit']:.6f}")
print(f"R² varianza: {fit_metrics['variance_fit']:.6f}")
print(f"Ratio elipticidad: {fit_metrics['ellipticity_ratio']:.4f}")

theoretical_ellipticity = (1/kappa_est + beta_est/kappa_est**2) / (1/kappa_est - beta_est/kappa_est**2)
print(f"Ratio teórico: {theoretical_ellipticity:.4f}")

plots_dir = save_all_plots()
results_file = save_results_to_txt()

save_all_plots()
save_results_to_txt()

print(f"\n" + "=" * 50)
print("ANÁLISIS COMPLETADO")
print(f"Gráficos guardados en: {plots_dir}/")
print(f"Resultados guardados en: {results_file}")