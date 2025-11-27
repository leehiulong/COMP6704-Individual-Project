import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

plots_dir = "results"

class CompressedSensingOptimizers:
    """
    Implementation of first-order optimization methods for compressed sensing
    """
    
    def __init__(self, A, y, lambda_val):
        self.A = A
        self.y = y
        self.lambda_val = lambda_val
        self.m, self.n = A.shape
        # Fixed: Added error handling for Lipschitz constant calculation
        try:
            self.L = np.linalg.norm(A.T @ A, 2)  # Lipschitz constant
        except np.linalg.LinAlgError:
            self.L = 1.0  # Fallback value
    
    def soft_threshold(self, x, threshold):
        """Soft thresholding operator"""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def objective_function(self, x):
        """LASSO objective function"""
        data_fidelity = 0.5 * np.linalg.norm(self.A @ x - self.y)**2
        regularization = self.lambda_val * np.linalg.norm(x, 1)
        return data_fidelity + regularization
    
    def reconstruction_error(self, x, x_true):
        """Relative reconstruction error"""
        norm_true = np.linalg.norm(x_true)
        if norm_true < 1e-10:  # Avoid division by zero
            return np.linalg.norm(x)
        return np.linalg.norm(x - x_true) / norm_true
    
    def ista(self, max_iter, x_true=None, tol=1e-8):
        """ISTA implementation"""
        x = np.zeros(self.n)
        objectives = []
        errors = []
        times = []
        start_time = time.time()
        
        step_size = 1.0 / self.L
        
        for k in range(max_iter):
            # Gradient step
            residual = self.A @ x - self.y
            gradient = self.A.T @ residual
            z = x - step_size * gradient
            
            # Proximal operator (soft thresholding)
            x_new = self.soft_threshold(z, step_size * self.lambda_val)
            
            # Store metrics
            obj_val = self.objective_function(x_new)
            objectives.append(obj_val)
            
            if x_true is not None:
                error = self.reconstruction_error(x_new, x_true)
                errors.append(error)
            
            times.append(time.time() - start_time)
            
            # Check convergence - FIXED: more robust convergence check
            if k > 10 and len(objectives) >= 2:
                rel_change = np.abs(objectives[-1] - objectives[-2]) / (np.abs(objectives[-2]) + 1e-10)
                if rel_change < tol:
                    print(f"  ISTA: Converged at iteration {k}")
                    break
                    
            x = x_new
            
        return {
            'x_opt': x,
            'objectives': objectives,
            'errors': errors if x_true is not None else None,
            'times': times,
            'iterations': len(objectives)
        }
    
    def fista(self, max_iter, x_true=None, tol=1e-8):
        """FISTA implementation with Nesterov acceleration"""
        x = np.zeros(self.n)
        y = x.copy()
        t = 1.0  # FIXED: ensure float
        objectives = []
        errors = []
        times = []
        start_time = time.time()
        
        step_size = 1.0 / self.L
        
        for k in range(max_iter):
            # Gradient at y
            residual = self.A @ y - self.y
            gradient = self.A.T @ residual
            
            # ISTA update
            x_new = self.soft_threshold(y - step_size * gradient, 
                                      step_size * self.lambda_val)
            
            # Momentum update
            t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y_new = x_new + ((t - 1) / t_new) * (x_new - x)
            
            # Store metrics
            obj_val = self.objective_function(x_new)
            objectives.append(obj_val)
            
            if x_true is not None:
                error = self.reconstruction_error(x_new, x_true)
                errors.append(error)
            
            times.append(time.time() - start_time)
            
            # Check convergence - FIXED: more robust convergence check
            if k > 10 and len(objectives) >= 2:
                rel_change = np.abs(objectives[-1] - objectives[-2]) / (np.abs(objectives[-2]) + 1e-10)
                if rel_change < tol:
                    print(f"  FISTA: Converged at iteration {k}")
                    break
                    
            x, y, t = x_new, y_new, t_new
            
        return {
            'x_opt': x,
            'objectives': objectives,
            'errors': errors if x_true is not None else None,
            'times': times,
            'iterations': len(objectives)
        }

    def rho_selection(self):
        # Estimate spectral norm of A^T A
        try:
            # Use power iteration for efficient estimation
            max_eigval = self.estimate_spectral_norm()
            rho = self.lambda_val * np.sqrt(max_eigval)  # Better scaling
        except:
            rho = self.lambda_val  # Fallback
        return rho

    def admm_woodbury(self, max_iter, x_true=None, rho_init=None, tol=1e-8):
        """ADMM with residual balancing + Woodbury identity"""
        x = np.zeros(self.n)
        z = np.zeros(self.n)
        u = np.zeros(self.n)
        objectives = []
        errors = []
        times = []
        start_time = time.time()

        # Initial ρ
        if rho_init is None:
            rho = self.lambda_val * np.sqrt(self.L)
        else:
            rho = rho_init
        
        # Residual balancing parameters
        mu = 2.0
        tau = 1.2
        update_iter = 10
        rho_min, rho_max = 1e-6, 1e6
        
        print(f"  ADMM Residual Balancing (Woodbury): Initial rho = {rho:.6f}")
        
        # Woodbury setup
        m, n = self.A.shape
        AAT = self.A @ self.A.T
        woodbury_matrix = np.eye(m) + (1/rho) * AAT
        L_woodbury = np.linalg.cholesky(woodbury_matrix)
        ATy = self.A.T @ self.y
        
        rho_changed = False
        
        for k in range(max_iter):
            try:
                if rho_changed:
                    # Recompute Woodbury matrix if ρ changed
                    woodbury_matrix = np.eye(m) + (1/rho) * AAT
                    L_woodbury = np.linalg.cholesky(woodbury_matrix)
                    rho_changed = False
                
                # x-update using Woodbury
                b = ATy + rho * (z - u)
                Ab = self.A @ b
                temp = np.linalg.solve(L_woodbury, Ab)
                temp = np.linalg.solve(L_woodbury.T, temp)
                x = (1/rho) * b - (1/rho**2) * self.A.T @ temp
                
                # Store old z for residual computation
                z_old = z.copy()
                
                # z-update
                z = self.soft_threshold(x + u, self.lambda_val / rho)
                
                # Dual update
                u = u + x - z
                
                # Residual balancing (every N iterations after warm-up)
                if k > 20 and k % update_iter == 0:
                    primal_residual = np.linalg.norm(x - z)
                    dual_residual = rho * np.linalg.norm(z - z_old)
                    
                    if primal_residual > tau * dual_residual:
                        new_rho = min(rho * mu, rho_max)
                        if new_rho != rho:
                            rho = new_rho
                            u = u / mu
                            rho_changed = True
                    elif dual_residual > tau * primal_residual:
                        new_rho = max(rho / mu, rho_min)
                        if new_rho != rho:
                            rho = new_rho
                            u = u * mu
                            rho_changed = True
                
                # Store metrics and check convergence
                obj_val = self.objective_function(x)
                objectives.append(obj_val)
                
                if x_true is not None:
                    error = self.reconstruction_error(x, x_true)
                    errors.append(error)
                
                times.append(time.time() - start_time)
                
                # Convergence check
                if k > 10 and len(objectives) >= 2:
                    rel_change = np.abs(objectives[-1] - objectives[-2]) / (np.abs(objectives[-2]) + 1e-10)
                    if rel_change < tol:
                        print(f"  ADMM Woodbury: Converged at iteration {k}")
                        break
                
            except Exception as e:
                print(f"  ADMM Residual Balancing (Woodbury): Error at iteration {k}: {e}")
                break
        
        return {
            'x_opt': x,
            'objectives': objectives,
            'errors': errors if x_true is not None else None,
            'times': times,
            'iterations': len(objectives),
            'final_rho': rho
        }


    def admm_cg(self, max_iter, x_true=None, rho_init=None, tol=1e-8):
        """ADMM with residual balancing + Conjugate Gradient"""
        x = np.zeros(self.n)
        z = np.zeros(self.n)
        u = np.zeros(self.n)
        objectives = []
        errors = []
        times = []
        start_time = time.time()
        
        # Initial ρ
        if rho_init is None:
            rho = self.lambda_val * np.sqrt(self.L)
        else:
            rho = rho_init
        
        # Residual balancing parameters
        mu = 2.0
        tau = 1.2
        update_iter = 10
        rho_min, rho_max = 1e-6, 1e6
        
        print(f"  ADMM Residual Balancing (CG): Initial rho = {rho:.6f}")
        
        ATy = self.A.T @ self.y
        
        for k in range(max_iter):
            try:
                # Define linear operator for current ρ
                def A_operator(v):
                    return self.A.T @ (self.A @ v) + rho * v
                
                # x-update using CG
                b = ATy + rho * (z - u)
                x, cg_info = self.conjugate_gradient(A_operator, b, x0=x, tol=1e-6)
                
                # Store old z for residual computation
                z_old = z.copy()
                
                # z-update
                z = self.soft_threshold(x + u, self.lambda_val / rho)
                
                # Dual update
                u = u + x - z
                
                # Residual balancing
                if k > 20 and k % update_iter == 0:
                    primal_residual = np.linalg.norm(x - z)
                    dual_residual = rho * np.linalg.norm(z - z_old)
                    
                    if primal_residual > tau * dual_residual:
                        new_rho = min(rho * mu, rho_max)
                        if new_rho != rho:
                            rho = new_rho
                            u = u / mu
                            # No need to recompute anything for CG!
                    elif dual_residual > tau * primal_residual:
                        new_rho = max(rho / mu, rho_min)
                        if new_rho != rho:
                            rho = new_rho
                            u = u * mu
                
                # Store metrics and check convergence
                obj_val = self.objective_function(x)
                objectives.append(obj_val)
                
                if x_true is not None:
                    error = self.reconstruction_error(x, x_true)
                    errors.append(error)
                
                times.append(time.time() - start_time)
                
                if k > 10 and len(objectives) >= 2:
                    rel_change = np.abs(objectives[-1] - objectives[-2]) / (np.abs(objectives[-2]) + 1e-10)
                    if rel_change < tol:
                        print(f"  ADMM CG: Converged at iteration {k}, CG iterations: {cg_info['iterations']}")
                        break
                
            except Exception as e:
                print(f"  ADMM Residual Balancing (CG): Error at iteration {k}: {e}")
                break
        
        return {
            'x_opt': x,
            'objectives': objectives,
            'errors': errors if x_true is not None else None,
            'times': times,
            'iterations': len(objectives),
            'final_rho': rho
        }

    def conjugate_gradient(self, A_func, b, x0=None, tol=1e-6, max_iter=100):
        """Conjugate Gradient solver for symmetric positive definite systems"""
        if x0 is None:
            x = np.zeros_like(b)
        else:
            x = x0.copy()
        
        r = b - A_func(x)
        p = r.copy()
        rsold = np.dot(r, r)
        
        iterations = 0
        for i in range(max_iter):
            Ap = A_func(p)
            alpha = rsold / np.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.dot(r, r)
            
            if np.sqrt(rsnew) < tol:
                break
                
            p = r + (rsnew / rsold) * p
            rsold = rsnew
            iterations += 1
        
        return x, {'iterations': iterations, 'residual': np.sqrt(rsold)}

    def estimate_spectral_norm(self, num_iter=10):
        """Estimate spectral norm of A^T A using power iteration"""
        n = self.A.shape[1]
        x = np.random.randn(n)
        x = x / np.linalg.norm(x)
        
        for i in range(num_iter):
            Ax = self.A @ x
            x = self.A.T @ Ax
            norm_x = np.linalg.norm(x)
            x = x / norm_x
        
        # Final Rayleigh quotient
        Ax = self.A @ x
        return np.linalg.norm(Ax)**2


def generate_synthetic_data(n, m, k, noise_level, seed):
    """Generate synthetic compressed sensing problem"""
    np.random.seed(seed)
    
    # Generate sparse signal
    x_true = np.zeros(n)
    support = np.random.choice(n, k, replace=False)
    x_true[support] = np.random.randn(k)
    
    # Generate random sensing matrix with normalized columns
    A = np.random.randn(m, n) / np.sqrt(m)
    
    # Generate measurements with noise
    y = A @ x_true + noise_level * np.random.randn(m)
    
    return A, y, x_true

# With better strategies:
def improved_lambda_selection(compression_ratio,sparsity_ratio,noise_level):
    """Choose lambda based on scenario characteristics"""
    
    # Base lambda from theory
    lambda_base = noise_level * np.sqrt(2 * np.log(n))
    
    # Adjust for compression ratio (more measurements → smaller lambda)
    compression_factor = 1.0 / np.sqrt(compression_ratio)
    
    # Adjust for sparsity (sparser signals → smaller lambda)
    sparsity_factor = np.sqrt(sparsity_ratio)
    
    lambda_val = lambda_base * compression_factor * sparsity_factor
    
    # Ensure reasonable bounds
    return max(1e-6, min(1.0, lambda_val))

def run_comprehensive_experiments(n,trials,max_iter):
    """Run comprehensive experiments comparing optimization methods across different scenarios"""
    
    # Define scenarios
    scenarios = {
        'low_compression_high_sparsity_low_noise': {
            'compression_ratio': 0.25,  # m/n = 0.25
            'sparsity_ratio': 0.1,     # k/n = 0.1
            'noise_level': 0.01,
            'label':"LCHSLN"
        },
        'low_compression_high_sparsity_high_noise': {
            'compression_ratio': 0.25,
            'sparsity_ratio': 0.1,
            'noise_level': 0.1,
            'label':"LCHSHN"
        },
        'low_compression_low_sparsity_low_noise': {
            'compression_ratio': 0.25,
            'sparsity_ratio': 0.05,
            'noise_level': 0.01,
            'label':"LCLSLN"
        },
        'low_compression_low_sparsity_high_noise': {
            'compression_ratio': 0.25,
            'sparsity_ratio': 0.05,
            'noise_level': 0.1,
            'label':"LCLSHN"
        },
        'high_compression_high_sparsity_low_noise': {
            'compression_ratio': 0.7,  # m/n = 0.7
            'sparsity_ratio': 0.1,
            'noise_level': 0.01,
            'label':"HCHSLN"
        },
        'high_compression_high_sparsity_high_noise': {
            'compression_ratio': 0.7,
            'sparsity_ratio': 0.1,
            'noise_level': 0.1,
            'label':"HCHSHN"
        },
        'high_compression_low_sparsity_low_noise': {
            'compression_ratio': 0.7,
            'sparsity_ratio': 0.05,
            'noise_level': 0.01,
            'label':"HCLSLN"
        },
        'high_compression_low_sparsity_high_noise': {
            'compression_ratio': 0.7,
            'sparsity_ratio': 0.05,
            'noise_level': 0.1,
            'label':"HCLSHN"
        }
    }
    
    # Store results
    results = {}
    method_names = ['ISTA', 'FISTA', 'ADMM_Woodbury','ADMM_CG']
    
    # Run experiments for each scenario
    for scenario_name, params in scenarios.items():
        print(f"\n=== Running scenario: {scenario_name} ===")
        print(f"  Compression: {params['compression_ratio']}, "
              f"Sparsity: {params['sparsity_ratio']}, "
              f"Noise: {params['noise_level']}")
        
        m = int(n * params['compression_ratio'])
        k = int(n * params['sparsity_ratio'])
        noise_level = params['noise_level']
        
        scenario_results = {method: {'objectives': [], 'errors': [], 'times': [], 'iterations': []} 
                          for method in method_names}
        
        for trial in range(trials):
            print(f"  Trial {trial + 1}/{trials}")
            seed = 42 + trial
            
            # Generate synthetic data
            A, y, x_true = generate_synthetic_data(n, m, k, noise_level, seed)
            
            # Choose lambda adaptively
            lambda_val = improved_lambda_selection(params['compression_ratio'],params['sparsity_ratio'],noise_level)
            
            # Initialize optimizer
            optimizer = CompressedSensingOptimizers(A, y, lambda_val)
            
            # Run all methods
            methods = {
                'ISTA': optimizer.ista,
                'FISTA': optimizer.fista, 
                # 'ADMM': optimizer.admm,
                'ADMM_Woodbury': optimizer.admm_woodbury,
                'ADMM_CG': optimizer.admm_cg,
                # 'ADMM_Optimized': optimizer.admm_optimized
            }
            
            for method_name, method_func in methods.items():
                try:
                    result = method_func(max_iter, x_true)
                    
                    scenario_results[method_name]['objectives'].append(result['objectives'])
                    scenario_results[method_name]['errors'].append(result['errors'])
                    scenario_results[method_name]['times'].append(result['times'])
                    scenario_results[method_name]['iterations'].append(result['iterations'])
                    
                except Exception as e:
                    print(f"    {method_name} failed: {e}")
                    # Add dummy data for failed runs
                    dummy_data = list(range(max_iter))
                    scenario_results[method_name]['objectives'].append(dummy_data)
                    scenario_results[method_name]['errors'].append(dummy_data)
                    scenario_results[method_name]['times'].append(dummy_data)
                    scenario_results[method_name]['iterations'].append(max_iter)
        
        results[scenario_name] = scenario_results
    
    return results, scenarios

def analyze_and_visualize_results(results, scenarios):
    """Analyze results and create comprehensive visualizations"""
    
    method_names = ['ISTA', 'FISTA','ADMM_Woodbury','ADMM_CG']
    colors = {'ISTA': 'blue', 'FISTA': 'purple','ADMM_Woodbury':'green','ADMM_CG':'red'}
    # 1. Convergence plots for each scenario
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (scenario_name, scenario_results) in enumerate(results.items()):
        ax = axes[idx]
        
        for method in method_names:
            # Get the shortest length among trials for this method
            min_length = min(len(obj_list) for obj_list in scenario_results[method]['objectives'] 
                           if len(obj_list) > 0)
            
            # Average across trials
            avg_objectives = np.zeros(min_length)
            for trial_obj in scenario_results[method]['objectives']:
                if len(trial_obj) >= min_length:
                    avg_objectives += trial_obj[:min_length]
            avg_objectives /= len(scenario_results[method]['objectives'])
            
            iterations = range(min_length)

            # Use different line styles to distinguish methods
            linestyle = '-' 
            if method == 'ADMM_CG':
                linestyle = '--'   # Dotted for CG

            ax.semilogy(iterations, avg_objectives, 
                       label=method, color=colors[method], linewidth=2, linestyle=linestyle)
        
        ax.set_title(f'{scenario_name}', fontsize=10)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Objective Value (log)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Convergence Behavior Across Different Scenarios', fontsize=16, y=1.02)
    plt.savefig(f'{plots_dir}/convergence_behavior.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Final reconstruction error comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (scenario_name, scenario_results) in enumerate(results.items()):
        ax = axes[idx]
        
        final_errors = []
        method_labels = []
        
        for method in method_names:
            method_errors = []
            for trial_errors in scenario_results[method]['errors']:
                if len(trial_errors) > 0:
                    method_errors.append(trial_errors[-1])
            
            if method_errors:
                final_errors.append(method_errors)
                method_labels.append(method)
        
        if final_errors:
            box_plot = ax.boxplot(final_errors, labels=method_labels, patch_artist=True)
            
            # Add colors
            for patch, method in zip(box_plot['boxes'], method_labels):
                patch.set_facecolor(colors[method])
                patch.set_alpha(0.7)
        
        ax.set_title(f'{scenario_name}', fontsize=10)
        ax.set_ylabel('Final Reconstruction Error')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Final Reconstruction Error Distribution Across Scenarios', fontsize=16, y=1.02)
    plt.savefig(f'{plots_dir}/reconstruction_errors.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Computation time analysis
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (scenario_name, scenario_results) in enumerate(results.items()):
        ax = axes[idx]
        
        final_times = []
        method_labels = []
        
        for method in method_names:
            method_times = []
            for trial_times in scenario_results[method]['times']:
                if len(trial_times) > 0:
                    method_times.append(trial_times[-1])
            
            if method_times:
                final_times.append(method_times)
                method_labels.append(method)
        
        if final_times:
            box_plot = ax.boxplot(final_times, labels=method_labels, patch_artist=True)
            
            for patch, method in zip(box_plot['boxes'], method_labels):
                patch.set_facecolor(colors[method])
                patch.set_alpha(0.7)
        
        ax.set_title(f'{scenario_name}', fontsize=10)
        ax.set_ylabel('Computation Time (seconds)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Computation Time Distribution Across Scenarios', fontsize=16, y=1.02)
    plt.savefig(f'{plots_dir}/computation_times.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Performance summary heatmap
    performance_metrics = {}
    
    for scenario_name, scenario_results in results.items():
        performance_metrics[scenario_name] = {}
        
        for method in method_names:
            if scenario_results[method]['errors']:
                # Average final error
                final_errors = [err_list[-1] for err_list in scenario_results[method]['errors'] 
                              if len(err_list) > 0]
                avg_error = np.mean(final_errors) if final_errors else np.nan
                
                # Average computation time
                final_times = [time_list[-1] for time_list in scenario_results[method]['times'] 
                             if len(time_list) > 0]
                avg_time = np.mean(final_times) if final_times else np.nan
                
                # Success rate (non-diverged runs)
                success_rate = len([err for err in final_errors if err < 0.5]) / len(final_errors) if final_errors else 0
                
                performance_metrics[scenario_name][method] = {
                    'avg_error': avg_error,
                    'avg_time': avg_time,
                    'success_rate': success_rate
                }
    
    # Create heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['avg_error', 'avg_time', 'success_rate']
    titles = ['Average Reconstruction Error', 'Average Computation Time (s)', 'Success Rate']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        data = []
        scenario_names = list(results.keys())
        
        for scenario in scenario_names:
            row = []
            for method in method_names:
                if method in performance_metrics[scenario]:
                    row.append(performance_metrics[scenario][method][metric])
                else:
                    row.append(np.nan)
            data.append(row)
        
        data = np.array(data)
        
        # Create heatmap
        im = axes[i].imshow(data, cmap='viridis_r' if metric == 'avg_error' else 'viridis', 
                           aspect='auto')
        
        axes[i].set_xticks(range(len(method_names)))
        axes[i].set_xticklabels(method_names)
        axes[i].set_yticks(range(len(scenario_names)))
        axes[i].set_yticklabels([scenarios[name]['label'] for name in scenario_names])
        
        # Add text annotations
        for j in range(len(scenario_names)):
            for k in range(len(method_names)):
                if not np.isnan(data[j, k]):
                    text = f'{data[j, k]:.3f}' if metric != 'success_rate' else f'{data[j, k]:.2f}'
                    axes[i].text(k, j, text, ha='center', va='center', 
                               color='white' if data[j, k] > np.nanmax(data)/2 else 'black')
        
        axes[i].set_title(title)
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.suptitle('Performance Metrics Summary Across All Scenarios', fontsize=16, y=1.05)
    plt.savefig(f'{plots_dir}/performance_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()

    performance_metrics = {}
    
    for scenario_name, scenario_results in results.items():
        performance_metrics[scenario_name] = {}
        
        for method in method_names:
            if scenario_results[method]['errors']:
                # Average final error
                final_errors = [err_list[-1] for err_list in scenario_results[method]['errors'] 
                              if len(err_list) > 0]
                avg_error = np.mean(final_errors) if final_errors else np.nan
                std_error = np.std(final_errors) if final_errors else np.nan
                
                # Average computation time
                final_times = [time_list[-1] for time_list in scenario_results[method]['times'] 
                             if len(time_list) > 0]
                avg_time = np.mean(final_times) if final_times else np.nan
                std_time = np.std(final_times) if final_times else np.nan
                
                # Success rate and iterations
                success_rate = len([err for err in final_errors if err < 0.5]) / len(final_errors) if final_errors else 0
                avg_iterations = np.mean(scenario_results[method]['iterations'])
                
                performance_metrics[scenario_name][method] = {
                    'avg_error': avg_error,
                    'std_error': std_error,
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'success_rate': success_rate,
                    'avg_iterations': avg_iterations
                }

    # SAVE COMPREHENSIVE NUMERICAL RESULTS TO FILE
    output_filename = "compressed_sensing_results.txt"
    
    with open(plots_dir+'/'+output_filename, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE COMPRESSED SENSING OPTIMIZATION RESULTS\n")
        f.write("=" * 100 + "\n")
        f.write(f"Number of scenarios: {len(scenarios)}\n")
        f.write(f"Methods compared: {', '.join(method_names)}\n\n")
        
        # 1. Detailed results for each scenario
        f.write("=" * 100 + "\n")
        f.write("DETAILED RESULTS BY SCENARIO\n")
        f.write("=" * 100 + "\n")
        
        for scenario_name, scenario_results in results.items():
            f.write(f"\nSCENARIO: {scenario_name.upper()}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Method':<8} {'Avg Error':<12} {'Std Error':<12} {'Avg Time(s)':<12} {'Std Time':<12} {'Success Rate':<12} {'Avg Iters':<12}\n")
            f.write("-" * 100 + "\n")
            
            for method in method_names:
                if method in performance_metrics[scenario_name]:
                    metrics = performance_metrics[scenario_name][method]
                    f.write(f"{method:<8} {metrics['avg_error']:<12.6f} {metrics['std_error']:<12.6f} "
                          f"{metrics['avg_time']:<12.4f} {metrics['std_time']:<12.4f} "
                          f"{metrics['success_rate']:<12.2%} {metrics['avg_iterations']:<12.1f}\n")
                else:
                    f.write(f"{method:<8} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}\n")
        
        # 2. Overall performance ranking
        f.write("\n" + "=" * 100 + "\n")
        f.write("OVERALL PERFORMANCE RANKING BY SCENARIO\n")
        f.write("=" * 100 + "\n")
        
        for scenario_name in performance_metrics:
            f.write(f"\n{scenario_name}:\n")
            methods_data = []
            for method in method_names:
                if method in performance_metrics[scenario_name]:
                    metrics = performance_metrics[scenario_name][method]
                    methods_data.append((method, metrics['avg_error'], metrics['avg_time'], metrics['success_rate']))
            
            # Rank by error (lower better)
            methods_data.sort(key=lambda x: x[1])
            for rank, (method, error, time, success) in enumerate(methods_data, 1):
                f.write(f"  {rank}. {method}: Error={error:.6f}, Time={time:.4f}s, Success={success:.2%}\n")
        
        # 3. Stability analysis results
        f.write("\n" + "=" * 100 + "\n")
        f.write("STABILITY ANALYSIS RESULTS\n")
        f.write("=" * 100 + "\n")
        
        stability_data = []
        
        for scenario_name, scenario_results in results.items():
            for method in method_names:
                if scenario_results[method]['errors']:
                    final_errors = [err_list[-1] for err_list in scenario_results[method]['errors'] 
                                  if len(err_list) > 0]
                    final_times = [time_list[-1] for time_list in scenario_results[method]['times'] 
                                 if len(time_list) > 0]
                    
                    if final_errors and final_times:
                        error_mean = np.mean(final_errors)
                        error_std = np.std(final_errors)
                        time_mean = np.mean(final_times)
                        time_std = np.std(final_times)
                        
                        # Coefficient of variation
                        error_cv = error_std / error_mean if error_mean > 0 else float('inf')
                        time_cv = time_std / time_mean if time_mean > 0 else float('inf')
                        
                        # Reliability scores
                        high_reliability = len([err for err in final_errors if err < 0.1]) / len(final_errors)
                        medium_reliability = len([err for err in final_errors if err < 0.5]) / len(final_errors)
                        low_reliability = len([err for err in final_errors if err < 1.0]) / len(final_errors)
                        
                        stability_data.append({
                            'Scenario': scenario_name,
                            'Method': method,
                            'Error_Mean': error_mean,
                            'Error_Std': error_std,
                            'Error_CV': error_cv,
                            'Time_Mean': time_mean,
                            'Time_Std': time_std,
                            'Time_CV': time_cv,
                            'High_Reliability': high_reliability,
                            'Medium_Reliability': medium_reliability,
                            'Low_Reliability': low_reliability
                        })
        
        stability_df = pd.DataFrame(stability_data)
        
        # Write stability metrics by scenario
        f.write(f"\n{'Scenario':<40} {'Method':<8} {'Error CV':<10} {'Time CV':<10} {'High Rel':<10} {'Med Rel':<10} {'Low Rel':<10}\n")
        f.write("-" * 100 + "\n")
        
        for scenario_name in results.keys():
            scenario_stability = stability_df[stability_df['Scenario'] == scenario_name]
            for _, row in scenario_stability.iterrows():
                f.write(f"{scenario_name:<40} {row['Method']:<8} {row['Error_CV']:<10.4f} {row['Time_CV']:<10.4f} "
                      f"{row['High_Reliability']:<10.2%} {row['Medium_Reliability']:<10.2%} {row['Low_Reliability']:<10.2%}\n")
        
        # 4. Overall stability summary
        f.write("\n" + "=" * 100 + "\n")
        f.write("OVERALL STABILITY SUMMARY\n")
        f.write("=" * 100 + "\n")
        
        stability_summary = {}
        
        for method in method_names:
            method_data = stability_df[stability_df['Method'] == method]
            if len(method_data) > 0:
                error_stability = 1 / (1 + np.mean(method_data['Error_CV']))
                time_stability = 1 / (1 + np.mean(method_data['Time_CV']))
                overall_reliability = np.mean(method_data['Medium_Reliability'])
                
                overall_stability = 0.4 * error_stability + 0.3 * time_stability + 0.3 * overall_reliability
                
                stability_summary[method] = {
                    'overall_stability': overall_stability,
                    'error_stability': error_stability,
                    'time_stability': time_stability,
                    'avg_reliability': overall_reliability,
                    'avg_error_cv': np.mean(method_data['Error_CV']),
                    'avg_time_cv': np.mean(method_data['Time_CV']),
                    'scenarios_tested': len(method_data)
                }
        
        # Write overall stability ranking
        f.write(f"\n{'Rank':<6} {'Method':<8} {'Overall Stability':<18} {'Error CV':<12} {'Time CV':<12} {'Reliability':<12}\n")
        f.write("-" * 80 + "\n")
        
        sorted_methods = sorted(stability_summary.items(), key=lambda x: x[1]['overall_stability'], reverse=True)
        for rank, (method, metrics) in enumerate(sorted_methods, 1):
            f.write(f"{rank:<6} {method:<8} {metrics['overall_stability']:<18.4f} {metrics['avg_error_cv']:<12.4f} "
                  f"{metrics['avg_time_cv']:<12.4f} {metrics['avg_reliability']:<12.2%}\n")
        
        # 5. Best method recommendations
        f.write("\n" + "=" * 100 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("=" * 100 + "\n")
        
        # Best for accuracy
        accuracy_ranking = []
        for method in method_names:
            errors = []
            for scenario in performance_metrics:
                if method in performance_metrics[scenario]:
                    errors.append(performance_metrics[scenario][method]['avg_error'])
            if errors:
                accuracy_ranking.append((method, np.mean(errors)))
        
        accuracy_ranking.sort(key=lambda x: x[1])
        
        f.write(f"\nACCURACY RANKING (Lower error better):\n")
        for rank, (method, avg_error) in enumerate(accuracy_ranking, 1):
            f.write(f"  {rank}. {method}: Average Error = {avg_error:.6f}\n")
        
        # Best for speed
        speed_ranking = []
        for method in method_names:
            times = []
            for scenario in performance_metrics:
                if method in performance_metrics[scenario]:
                    times.append(performance_metrics[scenario][method]['avg_time'])
            if times:
                speed_ranking.append((method, np.mean(times)))
        
        speed_ranking.sort(key=lambda x: x[1])
        
        f.write(f"\nSPEED RANKING (Lower time better):\n")
        for rank, (method, avg_time) in enumerate(speed_ranking, 1):
            f.write(f"  {rank}. {method}: Average Time = {avg_time:.4f}s\n")
        
        # Best for reliability
        reliability_ranking = []
        for method in method_names:
            success_rates = []
            for scenario in performance_metrics:
                if method in performance_metrics[scenario]:
                    success_rates.append(performance_metrics[scenario][method]['success_rate'])
            if success_rates:
                reliability_ranking.append((method, np.mean(success_rates)))
        
        reliability_ranking.sort(key=lambda x: x[1], reverse=True)
        
        f.write(f"\nRELIABILITY RANKING (Higher success rate better):\n")
        for rank, (method, avg_success) in enumerate(reliability_ranking, 1):
            f.write(f"  {rank}. {method}: Success Rate = {avg_success:.2%}\n")
        
        # Best overall (balanced)
        f.write(f"\nOVERALL RECOMMENDATION:\n")
        if sorted_methods:
            best_overall = sorted_methods[0][0]
            f.write(f"  Based on balanced performance across accuracy, speed, and stability:\n")
            f.write(f"  -> RECOMMENDED: {best_overall}\n")
        
        # 6. Algorithm performance summary
        f.write("\n" + "=" * 100 + "\n")
        f.write("ALGORITHM PERFORMANCE SUMMARY\n")
        f.write("=" * 100 + "\n")
        
        for method in method_names:
            f.write(f"\n{method} PERFORMANCE SUMMARY:\n")
            f.write("-" * 50 + "\n")
            
            # Collect all metrics for this method
            all_errors = []
            all_times = []
            all_success = []
            
            for scenario in performance_metrics:
                if method in performance_metrics[scenario]:
                    all_errors.append(performance_metrics[scenario][method]['avg_error'])
                    all_times.append(performance_metrics[scenario][method]['avg_time'])
                    all_success.append(performance_metrics[scenario][method]['success_rate'])
            
            if all_errors:
                f.write(f"  Average Error: {np.mean(all_errors):.6f} ± {np.std(all_errors):.6f}\n")
                f.write(f"  Average Time: {np.mean(all_times):.4f}s ± {np.std(all_times):.4f}s\n")
                f.write(f"  Average Success Rate: {np.mean(all_success):.2%}\n")
                f.write(f"  Best Scenario: {min([(scenario, performance_metrics[scenario][method]['avg_error']) for scenario in performance_metrics if method in performance_metrics[scenario]], key=lambda x: x[1])[0]}\n")
                f.write(f"  Worst Scenario: {max([(scenario, performance_metrics[scenario][method]['avg_error']) for scenario in performance_metrics if method in performance_metrics[scenario]], key=lambda x: x[1])[0]}\n")

    print(f"\n📊 All results saved to: {output_filename}")
    
    
    # 5. OVERALL STABILITY ANALYSIS - SIMPLIFIED VERSION
    print("\n" + "="*60)
    print("OVERALL STABILITY ANALYSIS")
    print("="*60)
    
    stability_data = []
    
    for scenario_name, scenario_results in results.items():
        for method in method_names:
            if scenario_results[method]['errors']:
                final_errors = [err_list[-1] for err_list in scenario_results[method]['errors'] 
                              if len(err_list) > 0]
                final_times = [time_list[-1] for time_list in scenario_results[method]['times'] 
                             if len(time_list) > 0]
                
                if final_errors and final_times:
                    error_mean = np.mean(final_errors)
                    error_std = np.std(final_errors)
                    time_mean = np.mean(final_times)
                    time_std = np.std(final_times)
                    
                    # Coefficient of variation (lower = more stable)
                    error_cv = error_std / error_mean if error_mean > 0 else float('inf')
                    time_cv = time_std / time_mean if time_mean > 0 else float('inf')
                    
                    stability_data.append({
                        'Scenario': scenario_name,
                        'Method': method,
                        'Error_CV': error_cv,
                        'Time_CV': time_cv,
                        'Reliability_Score': len([err for err in final_errors if err < 0.5]) / len(final_errors)
                    })
    
    stability_df = pd.DataFrame(stability_data)
    
    # print(stability_df)
    # SINGLE OVERALL STABILITY COMPARISON GRAPH
    # plt.figure(figsize=(12, 8))
    
    # Calculate overall stability metrics per method
    stability_metrics = {}
    
    for method in method_names:
        method_data = stability_df[stability_df['Method'] == method]
        if len(method_data) > 0:
            # Calculate combined stability score
            error_stability = 1 / (1 + np.mean(method_data['Error_CV']))  # Inverse so higher = better
            time_stability = 1 / (1 + np.mean(method_data['Time_CV']))    # Inverse so higher = better
            reliability = np.mean(method_data['Reliability_Score'])
            
            # Overall stability score (weighted average)
            overall_stability = 0.4 * error_stability + 0.3 * time_stability + 0.3 * reliability
            
            stability_metrics[method] = {
                'overall_stability': overall_stability,
                'error_stability': error_stability,
                'time_stability': time_stability,
                'reliability': reliability,
                'error_cv_mean': np.mean(method_data['Error_CV']),
                'time_cv_mean': np.mean(method_data['Time_CV'])
            }
    
    # Create the stability comparison bar chart
    metrics_to_plot = ['overall_stability', 'error_stability', 'time_stability', 'reliability']
    metric_labels = ['Overall Stability', 'Error Stability', 'Time Stability', 'Reliability']
    
    x = np.arange(len(metric_labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, method in enumerate(method_names):
        if method in stability_metrics:
            values = [stability_metrics[method][metric] for metric in metrics_to_plot]
            bars = ax.bar(x + i * width, values, width, label=method, 
                         color=colors[method], alpha=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Stability Metrics')
    ax.set_ylabel('Stability Score (Higher = Better)')
    ax.set_title('Overall Algorithm Stability Comparison Across All Scenarios\n'
                'Based on Coefficient of Variation and Reliability', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/stability_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print brief stability summary
    print("\nSTABILITY RANKING (Higher scores = more stable):")
    print("-" * 50)
    
    sorted_methods = sorted(stability_metrics.items(), 
                          key=lambda x: x[1]['overall_stability'], reverse=True)
    
    for rank, (method, metrics) in enumerate(sorted_methods, 1):
        print(f"{rank}. {method}:")
        print(f"   Overall Stability: {metrics['overall_stability']:.3f}")
        print(f"   Error CV (avg): {metrics['error_cv_mean']:.3f} (lower better)")
        print(f"   Time CV (avg): {metrics['time_cv_mean']:.3f} (lower better)")
        print(f"   Reliability: {metrics['reliability']:.1%}")
        print()
    
    return performance_metrics, stability_df

def print_summary_statistics(performance_metrics, stability_df):
    """Print summary statistics of the experiments"""
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Best method per scenario
    print("\nBEST METHOD PER SCENARIO (Lowest Error):")
    print("-" * 50)
    
    for scenario in performance_metrics:
        best_method = None
        best_error = float('inf')
        
        for method, metrics in performance_metrics[scenario].items():
            if metrics['avg_error'] < best_error:
                best_error = metrics['avg_error']
                best_method = method
        
        if best_method:
            print(f"{scenario:40} -> {best_method:6} (Error: {best_error:.4f})")
    
    # Overall best method
    print("\nOVERALL PERFORMANCE RANKING:")
    print("-" * 30)
    
    method_scores = {method: 0 for method in ['ISTA', 'FISTA','ADMM_Woodbury','ADMM_CG']}
    
    for scenario in performance_metrics:
        for method, metrics in performance_metrics[scenario].items():
            if metrics['avg_error'] < 0.5:  # Only count reasonable solutions
                method_scores[method] += 1
    
    for method, score in sorted(method_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{method:6}: {score:2d} wins out of {len(performance_metrics)} scenarios")
    
    # Stability analysis - FIXED: Use 'Error_CV' instead of 'CV'
    print("\nSTABILITY ANALYSIS (Coefficient of Variation - Lower is Better):")
    print("-" * 60)
    
    if 'Error_CV' in stability_df.columns:
        stability_summary = stability_df.groupby('Method')['Error_CV'].mean().sort_values()
        for method, cv in stability_summary.items():
            print(f"{method:6}: Average Error CV = {cv:.4f}")
    else:
        print("Stability data not available in expected format")
        
    # Additional stability metrics if available
    if 'Time_CV' in stability_df.columns:
        print("\nTIME STABILITY ANALYSIS:")
        print("-" * 30)
        time_stability = stability_df.groupby('Method')['Time_CV'].mean().sort_values()
        for method, cv in time_stability.items():
            print(f"{method:6}: Average Time CV = {cv:.4f}")

# Main execution
if __name__ == "__main__":
    print("Starting Comprehensive Compressed Sensing Experiments...")
    
    # Experiment parameters
    n = 10000  # Signal dimension
    trials = 10  # Number of trials per scenario
    max_iter = 1000000

    # Run experiments
    results, scenarios = run_comprehensive_experiments(n,trials,max_iter)
    
    # Analyze and visualize results
    performance_metrics, stability_df = analyze_and_visualize_results(results, scenarios)
    
    # Print summary statistics
    print_summary_statistics(performance_metrics, stability_df)
    
    print("\nExperiments completed successfully!")
