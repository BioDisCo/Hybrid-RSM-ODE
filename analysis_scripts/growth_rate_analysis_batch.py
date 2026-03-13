"""
Script to analyze multiple microalgae datasets and generate YAML summary
for light dependency analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from typing import Dict, Tuple, List
import warnings
import os
import yaml

# Import functions from the original script
from growth_rate_analysis import (
    load_and_process_data, extract_growth_data, calculate_specific_growth_rate,
    find_exponential_phase, fit_monod_parameters, monod_model,
    calculate_theoretical_growth_rates, calculate_logistic_growth_rates
)

warnings.filterwarnings('ignore')

# Parameters from tuning_ODE.py
conv_OD_to_cell = 4.46e6
base_C0 = 6e7  # Updated from 5.5e7 as per our analysis
mu_max = 0.17
Gamma = 1e5

# Ensure results directory exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

def analyze_single_dataset(filepath: str, light_intensity: int, date: str) -> Dict:
    """Analyze a single dataset and return results dictionary."""
    print(f"\n{'='*80}")
    print(f"ANALYZING DATASET: {date} (L0 = {light_intensity} µmol/m²/s)")
    print(f"File: {filepath}")
    print(f"{'='*80}")
    
    results = {
        'filepath': filepath,
        'light_intensity': light_intensity,
        'date': date,
        'growth_rates': {},
        'fitted_parameters': {},
        'steady_states': {},
        'model_performance': {},
        'conditions': []
    }
    
    try:
        # Load and process data
        df = load_and_process_data(filepath)
        growth_data = extract_growth_data(df)
        
        if not growth_data:
            print("No growth data found. Skipping this dataset.")
            return None
        
        print("Found growth data for conditions:", list(growth_data.keys()))
        
        # Calculate specific growth rates
        growth_rates = {}
        c0_concentrations = []
        
        for condition, (time, od) in growth_data.items():
            print(f"\nAnalyzing condition: {condition}")
            print(f"Time range: {time[0]:.1f} - {time[-1]:.1f} hours")
            print(f"OD range: {od.min():.4f} - {od.max():.4f}")
            
            # Convert OD to cell concentration
            cells_per_ml = od * conv_OD_to_cell
            print(f"Cell concentration range: {cells_per_ml.min():.2e} - {cells_per_ml.max():.2e} cells/mL")
            
            # Find exponential phase
            exp_start, exp_end = find_exponential_phase(time, cells_per_ml)
            print(f"Exponential phase detected: {exp_start:.1f} - {exp_end:.1f} hours")
            
            # Calculate growth rate
            mu = calculate_specific_growth_rate(time, cells_per_ml, (exp_start, exp_end))
            
            if not np.isnan(mu):
                growth_rates[condition] = mu
                c0_factor = float(condition.split('*')[1])
                c0_concentrations.append(c0_factor)
                print(f"Specific growth rate (μ): {mu:.4f} h⁻¹")
                
                # Store condition results (convert numpy types to Python types)
                results['conditions'].append({
                    'condition': condition,
                    'c0_factor': float(c0_factor),
                    'growth_rate': float(mu),
                    'exp_start': float(exp_start),
                    'exp_end': float(exp_end),
                    'od_min': float(od.min()),
                    'od_max': float(od.max()),
                    'cells_min': float(cells_per_ml.min()),
                    'cells_max': float(cells_per_ml.max())
                })
            else:
                print("Could not calculate reliable growth rate")
        
        if len(growth_rates) < 2:
            print("Insufficient data points for analysis")
            return None
        
        # Store growth rates (convert to regular dict with float values)
        results['growth_rates'] = {k: float(v) for k, v in growth_rates.items()}
        
        # Print results summary
        print("\n" + "="*50)
        print("GROWTH RATE SUMMARY")
        print("="*50)
        
        sorted_conditions = sorted(growth_rates.keys(), key=lambda x: float(x.split('*')[1]), reverse=True)
        for condition in sorted_conditions:
            mu = growth_rates[condition]
            print(f"{condition:<12}: μ = {mu:.4f} h⁻¹")
        
        # Prepare data for fitting
        c0_values = []
        mu_values = []
        
        for condition in sorted_conditions:
            c0_factor = float(condition.split('*')[1])
            mu = growth_rates[condition]
            c0_values.append(c0_factor)
            mu_values.append(mu)
        
        # Convert to cell concentrations
        c0_cell_concentrations = [base_C0 * factor for factor in c0_values]
        
        # Fit Monod parameters
        print("\n" + "="*60)
        print("FITTING MONOD MODEL TO EXPERIMENTAL DATA")
        print("="*60)
        
        fitted_mu_max, fitted_gamma, r_squared_fit, param_errors = fit_monod_parameters(
            c0_cell_concentrations, mu_values
        )
        
        if fitted_mu_max is not None:
            print(f"Fitted Parameters:")
            print(f"  μ_max = {fitted_mu_max:.4f} ± {param_errors[0]:.4f} h⁻¹")
            print(f"  Γ = {fitted_gamma:.2e} ± {param_errors[1]:.2e} cells/mL")
            print(f"  R² = {r_squared_fit:.4f}")
            
            results['fitted_parameters'] = {
                'mu_max': float(fitted_mu_max),
                'mu_max_error': float(param_errors[0]),
                'gamma': float(fitted_gamma),
                'gamma_error': float(param_errors[1]),
                'r_squared': float(r_squared_fit)
            }
        else:
            print("Failed to fit Monod model parameters")
            results['fitted_parameters'] = {
                'mu_max': None,
                'gamma': None,
                'r_squared': 0.0
            }
        
        # Calculate steady states
        steady_states = {}
        for condition, (time, od) in growth_data.items():
            c0_factor = float(condition.split('*')[1])
            cells_per_ml = od * conv_OD_to_cell
            
            # Take steady state as average of last 10% of time points
            n_points = len(cells_per_ml)
            steady_start_idx = int(0.9 * n_points)
            steady_state = np.mean(cells_per_ml[steady_start_idx:])
            steady_states[condition] = float(steady_state)
        
        results['steady_states'] = steady_states
        
        # Calculate model performance metrics
        original_theoretical_mu = calculate_theoretical_growth_rates(c0_values, base_C0, mu_max, Gamma)
        logistic_theoretical_mu = calculate_logistic_growth_rates(c0_values, base_C0, 
                                                                fitted_mu_max if fitted_mu_max else mu_max)
        
        # Calculate fitted theoretical values if we have fitted parameters
        if fitted_mu_max is not None:
            fitted_theoretical_mu = monod_model(np.array(c0_cell_concentrations), fitted_mu_max, fitted_gamma)
        else:
            fitted_theoretical_mu = None
        
        ss_res_orig = sum((exp - theo)**2 for exp, theo in zip(mu_values, original_theoretical_mu))
        ss_res_logistic = sum((exp - theo)**2 for exp, theo in zip(mu_values, logistic_theoretical_mu))
        ss_tot = sum((exp - np.mean(mu_values))**2 for exp in mu_values)
        
        r_squared_orig = 1 - (ss_res_orig / ss_tot) if ss_tot > 0 else 0.0
        r_squared_logistic = 1 - (ss_res_logistic / ss_tot) if ss_tot > 0 else 0.0
        
        results['model_performance'] = {
            'original_monod_r2': float(r_squared_orig),
            'logistic_r2': float(r_squared_logistic),
            'fitted_monod_r2': float(r_squared_fit if fitted_mu_max else 0.0)
        }
        
        print(f"\nModel Performance:")
        print(f"  Original Monod R² = {r_squared_orig:.4f}")
        print(f"  Logistic Model R² = {r_squared_logistic:.4f}")
        if fitted_mu_max is not None:
            print(f"  Fitted Monod R²   = {r_squared_fit:.4f}")
        
        # Generate plots for this dataset
        generate_dataset_plots(filepath, date, growth_data, growth_rates, c0_values, mu_values, 
                             c0_cell_concentrations, fitted_mu_max, fitted_gamma, r_squared_fit,
                             original_theoretical_mu, logistic_theoretical_mu, fitted_theoretical_mu)
        
        return results
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {str(e)}")
        return None

def generate_dataset_plots(filepath, date, growth_data, growth_rates, c0_values, mu_values, 
                         c0_cell_concentrations, fitted_mu_max, fitted_gamma, r_squared_fit,
                         original_theoretical_mu, logistic_theoretical_mu, fitted_theoretical_mu):
    """Generate plots for a single dataset with unique filenames."""
    
    # Skip plotting if insufficient data
    if len(growth_rates) < 2:
        return
    
    # Parameters for calculations
    logistic_mu_param = fitted_mu_max if fitted_mu_max is not None else mu_max
    
    # Create smooth curve for plotting - limit to experimental data range
    C_min = min(c0_cell_concentrations) * 0.5
    C_max = max(c0_cell_concentrations) * 1.1
    C_smooth = np.linspace(C_min, C_max, 100)
    
    if fitted_mu_max is not None:
        mu_smooth_fitted = monod_model(C_smooth, fitted_mu_max, fitted_gamma)
    mu_smooth_original = monod_model(C_smooth, mu_max, Gamma)
    mu_smooth_logistic = logistic_mu_param * C_smooth / base_C0
    
    # Create two plots side by side for growth rates vs concentration
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Normalized concentrations
    ax1.plot(c0_values, mu_values, 'bo-', linewidth=2, markersize=8, label='Experimental data')
    ax1.plot(c0_values, original_theoretical_mu, 'r--', linewidth=2, markersize=6, 
            label=f'Original Monod: μ_max={mu_max:.2f}, Γ={Gamma:.0e}')
    if fitted_mu_max is not None and fitted_theoretical_mu is not None:
        ax1.plot(c0_values, fitted_theoretical_mu, 'g-', linewidth=2, markersize=6, 
                label=f'Fitted Monod: μ_max={fitted_mu_max:.3f}, Γ={fitted_gamma:.1e}')
    ax1.plot(c0_values, logistic_theoretical_mu, 'm:', linewidth=2, 
            label=f'Logistic: μ×C/C₀ (μ={logistic_mu_param:.3f})')
    ax1.set_xlabel('Normalized nutrient concentration (C₀)', fontsize=12)
    ax1.set_ylabel('Specific growth rate μ (h⁻¹)', fontsize=12)
    ax1.set_title(f'Growth Rate vs Normalized Concentration ({date})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Add data point labels
    for i, (c0, mu) in enumerate(zip(c0_values, mu_values)):
        ax1.annotate(f'{mu:.3f}', (c0, mu), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8, color='blue')
    
    # Plot 2: Cell concentrations with smooth curves
    ax2.plot(c0_cell_concentrations, mu_values, 'bo-', linewidth=2, markersize=8, label='Experimental data')
    ax2.plot(C_smooth, mu_smooth_original, 'r--', linewidth=2, 
            label=f'Original Monod: μ_max={mu_max:.2f}, Γ={Gamma:.0e}')
    if fitted_mu_max is not None and fitted_theoretical_mu is not None:
        ax2.plot(C_smooth, mu_smooth_fitted, 'g-', linewidth=2, 
                label=f'Fitted Monod: μ_max={fitted_mu_max:.3f}, Γ={fitted_gamma:.1e}')
    ax2.plot(C_smooth, mu_smooth_logistic, 'm:', linewidth=2, 
            label=f'Logistic: μ×C/C₀ (μ={logistic_mu_param:.3f})')
    ax2.set_xlabel('Nutrient concentration (cells/mL)', fontsize=12)
    ax2.set_ylabel('Specific growth rate μ (h⁻¹)', fontsize=12)
    ax2.set_title(f'Growth Rate vs Cell Concentration ({date})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # Add data point labels
    for i, (c0_cells, mu) in enumerate(zip(c0_cell_concentrations, mu_values)):
        ax2.annotate(f'{mu:.3f}', (c0_cells, mu), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8, color='blue')
    
    plt.tight_layout()
    growth_rate_plot_path = os.path.join(results_dir, f'growth_rate_vs_concentration_{date}.png')
    plt.savefig(growth_rate_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Growth rate plot saved: {growth_rate_plot_path}")
    
    # Individual growth curves plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(growth_data)))
    
    for i, (condition, (time, od)) in enumerate(sorted(growth_data.items(), 
                                                      key=lambda x: float(x[0].split('*')[1]), 
                                                      reverse=True)):
        plt.subplot(2, 3, i+1)
        
        # Convert OD to cell concentration
        cells_per_ml = od * conv_OD_to_cell
        
        plt.semilogy(time, cells_per_ml, 'o-', color=colors[i], label=condition)
        
        # Highlight exponential phase
        exp_start, exp_end = find_exponential_phase(time, cells_per_ml)
        exp_mask = (time >= exp_start) & (time <= exp_end)
        if np.any(exp_mask):
            cells_exp = cells_per_ml[exp_mask]
            plt.semilogy(time[exp_mask], cells_exp, 's-', color='red', alpha=0.7, 
                       markersize=4, label='Exponential phase')
        
        plt.xlabel('Time (hours)')
        plt.ylabel('Cell concentration (cells/mL)')
        plt.title(condition)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
    
    plt.suptitle(f'Individual Growth Curves - {date}', fontsize=16)
    plt.tight_layout()
    individual_curves_path = os.path.join(results_dir, f'individual_growth_curves_{date}.png')
    plt.savefig(individual_curves_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Individual curves plot saved: {individual_curves_path}")
    
    # Steady states plot
    plt.figure(figsize=(10, 6))
    
    # Extract steady state data
    dilution_factors = []
    steady_state_concentrations = []
    condition_labels = []
    
    for condition, (time, od) in sorted(growth_data.items(), 
                                      key=lambda x: float(x[0].split('*')[1]), 
                                      reverse=True):
        c0_factor = float(condition.split('*')[1])
        cells_per_ml = od * conv_OD_to_cell
        
        # Take steady state as average of last 10% of time points
        n_points = len(cells_per_ml)
        steady_start_idx = int(0.9 * n_points)
        steady_state = np.mean(cells_per_ml[steady_start_idx:])
        
        dilution_factors.append(c0_factor)
        steady_state_concentrations.append(steady_state)
        condition_labels.append(condition)
    
    # Plot steady states
    plt.plot(dilution_factors, steady_state_concentrations, 'ro-', 
             linewidth=2, markersize=8, label='Experimental steady states')
    
    # Add line with slope 1: starts at full normalized concentration 1 and has slope 1 (y = x)
    x_line = np.linspace(1.0, 0.0, 100)  # From 1 to 0 normalized concentration
    y_line = x_line * max(steady_state_concentrations)  # Scale to match data range
    plt.plot(x_line, y_line, 'k-', linewidth=2, alpha=0.7,
            label='y = x (slope=1, normalized)')
    
    # Add data point labels
    for i, (df, ss, label) in enumerate(zip(dilution_factors, steady_state_concentrations, condition_labels)):
        plt.annotate(f'{label}\n({ss:.1e} cells/mL)', (df, ss), 
                    textcoords="offset points", xytext=(0, 15), 
                    ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", alpha=0.7))
    
    plt.xlabel('Normalized nutrient concentration (C₀ factor)', fontsize=12)
    plt.ylabel('Steady-state cell concentration (cells/mL)', fontsize=12)
    plt.title(f'Steady-State Concentrations vs Dilution Factor - {date}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Set y-axis to scientific notation
    plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    steady_states_path = os.path.join(results_dir, f'steady_states_vs_dilution_{date}.png')
    plt.savefig(steady_states_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Steady states plot saved: {steady_states_path}")

def main():
    """Main function to analyze all datasets and generate YAML output."""
    # Define all datasets to analyze
    data_path = "../all_data/"
    files_to_analyze = [
        {"filepath": "data_exp_Chlamy_16-09-24.csv", "light_intensity": 22, "date": "16-09-24"},
        {"filepath": "data_exp_Chlamy_21-10-24.csv", "light_intensity": 45, "date": "21-10-24"},
        {"filepath": "data_exp_Chlamy_04-11-24.csv", "light_intensity": 45, "date": "04-11-24"},
        {"filepath": "data_exp_Chlamy_17-02-25.csv", "light_intensity": 90, "date": "17-02-25"},
        {"filepath": "data_exp_Chlamy_01-07-25.csv", "light_intensity": 180, "date": "01-07-25"},
        {"filepath": "data_exp_Chlamy_07-07-25.csv", "light_intensity": 300, "date": "07-07-25"},
    ]
    
    all_results = {"datasets": []}
    
    # Analyze each dataset
    for file_info in files_to_analyze:
        result = analyze_single_dataset(
            data_path + file_info["filepath"],
            file_info["light_intensity"], 
            file_info["date"]
        )
        
        if result:
            all_results["datasets"].append(result)
    
    # Save results to YAML
    yaml_filepath = os.path.join(results_dir, "microalgae_growth_analysis.yaml")
    with open(yaml_filepath, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False, indent=2)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {yaml_filepath}")
    print(f"Total datasets analyzed: {len(all_results['datasets'])}")
    
    # Print summary table
    print(f"\n{'Dataset':<12} {'L0':<5} {'μ_max':<8} {'Γ(×10⁶)':<10} {'R²':<6} {'Conditions':<10}")
    print("-"*60)
    
    for dataset in all_results["datasets"]:
        fitted = dataset['fitted_parameters']
        mu_max_val = fitted.get('mu_max', 'N/A')
        gamma_val = fitted.get('gamma', 'N/A')
        r2_val = fitted.get('r_squared', 'N/A')
        
        if isinstance(mu_max_val, (int, float)) and isinstance(gamma_val, (int, float)):
            print(f"{dataset['date']:<12} {dataset['light_intensity']:<5} {mu_max_val:<8.4f} {gamma_val/1e6:<10.2f} {r2_val:<6.4f} {len(dataset['conditions']):<10}")
        else:
            print(f"{dataset['date']:<12} {dataset['light_intensity']:<5} {'N/A':<8} {'N/A':<10} {'N/A':<6} {len(dataset['conditions']):<10}")

if __name__ == "__main__":
    main()