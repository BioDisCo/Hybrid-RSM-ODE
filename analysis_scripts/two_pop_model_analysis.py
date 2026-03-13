"""
Script to analyze TWO_POP_MODEL against experimental data and compare
growth rates and steady states with model predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp
from typing import Dict, Tuple, List
import warnings
import os
import sys
import yaml

# Add parent directory to path to import ode module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ode import AlgalSysParameters, Model, dxdt_algae_model

# Import functions from growth_rate_analysis
from growth_rate_analysis import (
    load_and_process_data, extract_growth_data, calculate_specific_growth_rate,
    find_exponential_phase
)

warnings.filterwarnings('ignore')

# Parameters
conv_OD_to_cell = 4.46e6
base_C0 = 6e7

# Ensure results directory exists
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)


def calculate_model_growth_rate(params, light_intensity, c0_factor, N_initial=None, time_max=50):
    """Calculate initial growth rate from TWO_POP_MODEL."""
    try:
        adjusted_params = params._replace(
            L0=light_intensity,
            C0=params.C0 * c0_factor
        )
        
        if N_initial is None:
            Nv0 = adjusted_params.Nv0
            Nnv0 = adjusted_params.Nnv0
        else:
            Nv0 = N_initial * 0.9
            Nnv0 = N_initial * 0.1
        
        y0 = [Nv0, Nnv0]
        
        sol = solve_ivp(
            fun=lambda t, y: dxdt_algae_model(t, y, adjusted_params, model=Model.TWO_POP_MODEL),
            t_span=(0, time_max),
            y0=y0,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10
        )
        
        if sol.success and len(sol.t) > 10:
            t_early = sol.t[:min(10, len(sol.t))]
            N_total_early = sol.y[0][:len(t_early)] + sol.y[1][:len(t_early)]
            
            valid_mask = N_total_early > 0
            if np.sum(valid_mask) > 5:
                t_valid = t_early[valid_mask]
                ln_N_total = np.log(N_total_early[valid_mask])
                mu_total = np.polyfit(t_valid, ln_N_total, 1)[0]
                return mu_total
        
        return np.nan
        
    except Exception as e:
        print(f"Error calculating model growth rate: {e}")
        return np.nan


def calculate_model_steady_state(params, light_intensity, c0_factor, time_max=1000):
    """Calculate steady state from TWO_POP_MODEL."""
    try:
        adjusted_params = params._replace(
            L0=light_intensity,
            C0=params.C0 * c0_factor
        )
        
        y0 = [adjusted_params.Nv0, adjusted_params.Nnv0]
        
        sol = solve_ivp(
            fun=lambda t, y: dxdt_algae_model(t, y, adjusted_params, model=Model.TWO_POP_MODEL),
            t_span=(0, time_max),
            y0=y0,
            method="LSODA",
            rtol=1e-8,
            atol=1e-10
        )
        
        if sol.success and len(sol.y[0]) > 0:
            N_total_ss = sol.y[0][-1] + sol.y[1][-1]
            return N_total_ss
        
        return np.nan
            
    except Exception as e:
        print(f"Error calculating model steady state: {e}")
        return np.nan


def analyze_single_dataset_with_model(filepath: str, light_intensity: int, date: str, params: AlgalSysParameters) -> Dict:
    """Analyze a single dataset and compare with TWO_POP_MODEL predictions."""
    print(f"\n{'='*80}")
    print(f"ANALYZING DATASET: {date} (L0 = {light_intensity} µmol/m²/s)")
    print(f"File: {filepath}")
    print(f"{'='*80}")
    
    results = {
        'filepath': filepath,
        'light_intensity': light_intensity,
        'date': date,
        'experimental_growth_rates': {},
        'model_growth_rates': {},
        'experimental_steady_states': {},
        'model_steady_states': {},
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
        
        # Calculate experimental growth rates and steady states
        for condition, (time, od) in growth_data.items():
            print(f"\nAnalyzing condition: {condition}")
            c0_factor = float(condition.split('*')[1])
            
            # Convert OD to cell concentration
            cells_per_ml = od * conv_OD_to_cell
            
            # Find exponential phase
            exp_start, exp_end = find_exponential_phase(time, cells_per_ml)
            
            # Calculate experimental growth rate
            mu_exp = calculate_specific_growth_rate(time, cells_per_ml, (exp_start, exp_end))
            
            # Calculate experimental steady state
            n_points = len(cells_per_ml)
            steady_start_idx = int(0.9 * n_points)
            steady_state_exp = np.mean(cells_per_ml[steady_start_idx:])
            
            # Calculate model predictions
            mu_model = calculate_model_growth_rate(params, light_intensity, c0_factor)
            steady_state_model = calculate_model_steady_state(params, light_intensity, c0_factor)
            
            if not np.isnan(mu_exp):
                results['experimental_growth_rates'][condition] = float(mu_exp)
                results['model_growth_rates'][condition] = float(mu_model) if not np.isnan(mu_model) else None
                results['experimental_steady_states'][condition] = float(steady_state_exp)
                results['model_steady_states'][condition] = float(steady_state_model) if not np.isnan(steady_state_model) else None
                
                print(f"  Experimental μ: {mu_exp:.4f} h⁻¹")
                print(f"  Model μ:        {mu_model:.4f} h⁻¹" if not np.isnan(mu_model) else "  Model μ:        N/A")
                print(f"  Exp. steady state:   {steady_state_exp:.2e} cells/mL")
                print(f"  Model steady state:  {steady_state_model:.2e} cells/mL" if not np.isnan(steady_state_model) else "  Model steady state:  N/A")
                
                results['conditions'].append({
                    'condition': condition,
                    'c0_factor': float(c0_factor),
                    'exp_growth_rate': float(mu_exp),
                    'model_growth_rate': float(mu_model) if not np.isnan(mu_model) else None,
                    'exp_steady_state': float(steady_state_exp),
                    'model_steady_state': float(steady_state_model) if not np.isnan(steady_state_model) else None,
                    'exp_start': float(exp_start),
                    'exp_end': float(exp_end)
                })
        
        if len(results['experimental_growth_rates']) < 2:
            print("Insufficient data points for analysis")
            return None
        
        # Generate plots in the same style as growth_rate_analysis.py
        generate_growth_rate_plots(filepath, date, light_intensity, growth_data, results, params)
        
        return results
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_growth_rate_plots(filepath, date, light_intensity, growth_data, results, params):
    """Generate plots matching the style of growth_rate_analysis.py."""
    
    # Extract and sort data
    conditions = sorted(results['experimental_growth_rates'].keys(),
                       key=lambda x: float(x.split('*')[1]), reverse=True)
    c0_factors = [float(c.split('*')[1]) for c in conditions]
    
    exp_growth_rates = [results['experimental_growth_rates'][c] for c in conditions]
    model_growth_rates = [results['model_growth_rates'][c] if results['model_growth_rates'][c] is not None else np.nan for c in conditions]
    
    exp_steady_states = [results['experimental_steady_states'][c] for c in conditions]
    model_steady_states = [results['model_steady_states'][c] if results['model_steady_states'][c] is not None else np.nan for c in conditions]
    
    # Create figure with 2 rows and 3 columns (same as growth_rate_analysis.py)
    fig = plt.figure(figsize=(18, 12))
    
    # Row 1: Individual growth curves (6 conditions, but we'll plot only available ones)
    colors = plt.cm.viridis(np.linspace(0, 1, len(growth_data)))
    
    for i, (condition, (time, od)) in enumerate(sorted(growth_data.items(),
                                                      key=lambda x: float(x[0].split('*')[1]),
                                                      reverse=True)):
        ax = plt.subplot(2, 3, i+1)
        c0_factor = float(condition.split('*')[1])
        
        # Experimental data
        cells_per_ml = od * conv_OD_to_cell
        ax.plot(time, cells_per_ml/1e6, 'o-', color=colors[i], label='Experimental',
                markersize=5, linewidth=2)
        
        # Model simulation
        adjusted_params = params._replace(L0=light_intensity, C0=params.C0 * c0_factor)
        y0 = [adjusted_params.Nv0, adjusted_params.Nnv0]
        
        try:
            sol = solve_ivp(
                fun=lambda t, y: dxdt_algae_model(t, y, adjusted_params, model=Model.TWO_POP_MODEL),
                t_span=(0, max(time)),
                y0=y0,
                method="LSODA",
                rtol=1e-8,
                atol=1e-10,
                dense_output=True
            )
            
            if sol.success:
                t_model = np.linspace(0, max(time), 200)
                N_model = sol.sol(t_model)[0] + sol.sol(t_model)[1]
                ax.plot(t_model, N_model/1e6, '--', color='red', linewidth=2.5,
                       label='TWO_POP_MODEL', alpha=0.8)
        except Exception as e:
            print(f"Could not plot model for {condition}: {e}")
        
        # Highlight exponential phase
        cond_result = next((c for c in results['conditions'] if c['condition'] == condition), None)
        if cond_result:
            exp_start = cond_result['exp_start']
            exp_end = cond_result['exp_end']
            ax.axvspan(exp_start, exp_end, alpha=0.2, color='green', label='Exponential phase')
        
        ax.set_xlabel('Time (hours)', fontsize=11)
        ax.set_ylabel('Cell concentration (×10⁶ cells/mL)', fontsize=11)
        ax.set_title(f'{condition}\nμ_exp={results["experimental_growth_rates"][condition]:.4f} h⁻¹',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
    
    plt.suptitle(f'Growth Analysis: {date} (L0={light_intensity} µmol/m²/s)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(results_dir, f'growth_analysis_{date}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Growth analysis plot saved: {output_path}")
    
    # Create separate plots for growth rates and steady states vs C0
    # (matching plot_light_dependency_from_yaml.py style)
    
    # Plot 1: Growth rate vs C0
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(c0_factors, exp_growth_rates, 'o-', color='#2E86AB', linewidth=2.5,
            markersize=10, label='Experimental data', markeredgecolor='white', markeredgewidth=1.5)
    
    valid_model = ~np.isnan(model_growth_rates)
    if np.any(valid_model):
        ax.plot(np.array(c0_factors)[valid_model], np.array(model_growth_rates)[valid_model],
                '^--', color='#A23B72', linewidth=2.5, markersize=10,
                label='TWO_POP_MODEL', markeredgecolor='white', markeredgewidth=1.5)
    
    # Add value labels
    for c0, mu in zip(c0_factors, exp_growth_rates):
        ax.annotate(f'{mu:.3f}', (c0, mu), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Normalized nutrient concentration (C₀)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Specific growth rate μ (h⁻¹)', fontsize=14, fontweight='bold')
    ax.set_title(f'Growth Rate vs Nutrient Concentration\n{date} (L0={light_intensity} µmol/m²/s)',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, framealpha=0.9, loc='best')
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'growth_rate_vs_C0_{date}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Growth rate vs C0 plot saved: {output_path}")
    
    # Plot 2: Steady state vs C0
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(c0_factors, np.array(exp_steady_states)/1e6, 'o-', color='#2E86AB',
            linewidth=2.5, markersize=10, label='Experimental data',
            markeredgecolor='white', markeredgewidth=1.5)
    
    valid_model = ~np.isnan(model_steady_states)
    if np.any(valid_model):
        ax.plot(np.array(c0_factors)[valid_model], np.array(model_steady_states)[valid_model]/1e6,
                '^--', color='#A23B72', linewidth=2.5, markersize=10,
                label='TWO_POP_MODEL', markeredgecolor='white', markeredgewidth=1.5)
    
    # Add value labels
    for c0, ss in zip(c0_factors, exp_steady_states):
        ax.annotate(f'{ss/1e6:.2f}', (c0, ss/1e6), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Normalized nutrient concentration (C₀)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Steady state (×10⁶ cells/mL)', fontsize=14, fontweight='bold')
    ax.set_title(f'Steady State vs Nutrient Concentration\n{date} (L0={light_intensity} µmol/m²/s)',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, framealpha=0.9, loc='best')
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, f'steady_state_vs_C0_{date}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Steady state vs C0 plot saved: {output_path}")


def plot_light_dependency_comparison(all_results, params):
    """
    Plot growth rates and steady states vs light intensity.
    Matches the style of plot_light_dependency_from_yaml.py
    """
    
    # Extract data for C0*1.0 condition
    light_intensities = []
    c0_1_exp_growth = []
    c0_1_model_growth = []
    c0_1_exp_steady = []
    c0_1_model_steady = []
    dates = []
    
    for result in all_results:
        if 'C0*1.0' in result['experimental_growth_rates']:
            light_intensities.append(result['light_intensity'])
            dates.append(result['date'])
            c0_1_exp_growth.append(result['experimental_growth_rates']['C0*1.0'])
            c0_1_model_growth.append(result['model_growth_rates'].get('C0*1.0', np.nan))
            c0_1_exp_steady.append(result['experimental_steady_states']['C0*1.0'])
            c0_1_model_steady.append(result['model_steady_states'].get('C0*1.0', np.nan))
    
    # Sort by light intensity
    sorted_indices = np.argsort(light_intensities)
    light_intensities = np.array(light_intensities)[sorted_indices]
    c0_1_exp_growth = np.array(c0_1_exp_growth)[sorted_indices]
    c0_1_model_growth = np.array(c0_1_model_growth)[sorted_indices]
    c0_1_exp_steady = np.array(c0_1_exp_steady)[sorted_indices]
    c0_1_model_steady = np.array(c0_1_model_steady)[sorted_indices]
    dates = np.array(dates)[sorted_indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Growth rate vs light intensity
    valid_exp = ~np.isnan(c0_1_exp_growth)
    valid_model = ~np.isnan(c0_1_model_growth)
    
    ax1.plot(light_intensities[valid_exp], c0_1_exp_growth[valid_exp],
             'o-', color='#2E86AB', linewidth=3, markersize=12,
             label='Experimental (C0*1.0)', markeredgecolor='white', markeredgewidth=2)
    
    if np.any(valid_model):
        ax1.plot(light_intensities[valid_model], c0_1_model_growth[valid_model],
                 '^--', color='#A23B72', linewidth=3, markersize=12,
                 label='TWO_POP_MODEL (C0*1.0)', markeredgecolor='white', markeredgewidth=2)
    
    # Add date labels
    for x, y, date in zip(light_intensities[valid_exp], c0_1_exp_growth[valid_exp], dates[valid_exp]):
        ax1.annotate(date, (x, y), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Light Intensity L₀ (µmol/m²/s)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Growth Rate μ (h⁻¹)', fontsize=14, fontweight='bold')
    ax1.set_title('Growth Rate vs Light Intensity (C0*1.0)', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=12, framealpha=0.9, loc='best')
    
    # Plot 2: Steady state vs light intensity
    ax2.plot(light_intensities[valid_exp], c0_1_exp_steady[valid_exp]/1e6,
             'o-', color='#2E86AB', linewidth=3, markersize=12,
             label='Experimental (C0*1.0)', markeredgecolor='white', markeredgewidth=2)
    
    if np.any(valid_model):
        ax2.plot(light_intensities[valid_model], c0_1_model_steady[valid_model]/1e6,
                 '^--', color='#A23B72', linewidth=3, markersize=12,
                 label='TWO_POP_MODEL (C0*1.0)', markeredgecolor='white', markeredgewidth=2)
    
    # Add date labels
    for x, y, date in zip(light_intensities[valid_exp], c0_1_exp_steady[valid_exp]/1e6, dates[valid_exp]):
        ax2.annotate(date, (x, y), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Light Intensity L₀ (µmol/m²/s)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Steady State (×10⁶ cells/mL)', fontsize=14, fontweight='bold')
    ax2.set_title('Steady State vs Light Intensity (C0*1.0)', fontsize=16, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=12, framealpha=0.9, loc='best')
    
    plt.suptitle('Light Dependency Analysis - TWO_POP_MODEL',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(results_dir, 'light_dependency_two_pop_model.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nLight dependency plot saved: {output_path}")
    plt.close()


def main():
    """Main function to analyze all datasets with TWO_POP_MODEL."""
    
    # Define TWO_POP_MODEL parameters
    params = AlgalSysParameters(
        C0= 55965472.635662824,
        Gamma= 10000000.0,
        K= 4.671881841476606e-08,
        Ki= 1996.150989776576,
        L0= 300.0,
        L_meters= 0.01,
        N0= 100000.0,
        Nnv0= 4877.73075749416,
        Nv0= 55831.37025797249,
        V_mL= 50.0,
        alpha= 1.0,
        dnv0= 0.005502020916403655,
        dnvs= 0.011371587711016977,
        dv0= 0.009293488364607923,
        dvs= 0.001,
        k0= 0.05966814254057332,
        lamb_L= 12.422611374972211,
        m= 0.01,
        mu= 0.17,
        mu_max= 1.0,
        mu_nv_max= 0.006589625867586623,
        mu_v_max= 0.1514887843752428,
        q0= 1.0,
        q_init= 1.0,
        ql= 1.0,
        rho_max= 1.0,
        s0= 1.0,
        wL= 0.4245558571412867,
        wc= 0.4197983723692126,
        xi_c= 3764203.582669872
    )
    
    print("="*80)
    print("TWO POPULATION MODEL - EXPERIMENTAL DATA ANALYSIS")
    print("="*80)
    print(f"\nModel parameters:")
    print(f"  mu_v_max = {params.mu_v_max} h⁻¹")
    print(f"  mu_nv_max = {params.mu_nv_max} h⁻¹")
    print(f"  C0 = {params.C0:.2e} cells/mL")
    print(f"  xi_c = {params.xi_c:.2e} cells/mL")
    
    # Define datasets to analyze
    data_path = "../all_data/"
    files_to_analyze = [
        {"filepath": "data_exp_Chlamy_16-09-24.csv", "light_intensity": 22, "date": "16-09-24"},
        {"filepath": "data_exp_Chlamy_21-10-24.csv", "light_intensity": 45, "date": "21-10-24"},
        {"filepath": "data_exp_Chlamy_04-11-24.csv", "light_intensity": 45, "date": "04-11-24"},
        {"filepath": "data_exp_Chlamy_17-02-25.csv", "light_intensity": 90, "date": "17-02-25"},
        {"filepath": "data_exp_Chlamy_01-07-25.csv", "light_intensity": 180, "date": "01-07-25"},
        {"filepath": "data_exp_Chlamy_07-07-25.csv", "light_intensity": 300, "date": "07-07-25"},
    ]
    
    all_results = []
    
    # Analyze each dataset
    for file_info in files_to_analyze:
        result = analyze_single_dataset_with_model(
            data_path + file_info["filepath"],
            file_info["light_intensity"],
            file_info["date"],
            params
        )
        
        if result:
            all_results.append(result)
    
    # Save results to YAML
    yaml_filepath = os.path.join(results_dir, "two_pop_model_analysis.yaml")
    with open(yaml_filepath, 'w') as f:
        yaml.dump({"datasets": all_results}, f, default_flow_style=False, indent=2)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {yaml_filepath}")
    print(f"Total datasets analyzed: {len(all_results)}")
    
    # Print summary table
    print(f"\n{'Dataset':<12} {'L0':<5} {'Exp μ (C0*1.0)':<15} {'Model μ':<12} {'Diff':<10}")
    print("-"*65)
    
    for result in all_results:
        exp_mu = result['experimental_growth_rates'].get('C0*1.0', np.nan)
        model_mu = result['model_growth_rates'].get('C0*1.0', np.nan)
        diff = exp_mu - model_mu if not np.isnan(exp_mu) and not np.isnan(model_mu) else np.nan
        
        if not np.isnan(exp_mu):
            print(f"{result['date']:<12} {result['light_intensity']:<5} "
                  f"{exp_mu:<15.4f} {model_mu:<12.4f} {diff:<10.4f}")
        else:
            print(f"{result['date']:<12} {result['light_intensity']:<5} {'N/A':<15} {'N/A':<12} {'N/A':<10}")
    
    # Generate light dependency comparison
    if len(all_results) > 0:
        plot_light_dependency_comparison(all_results, params)
    
    print(f"\nAll plots have been saved to the '{results_dir}' directory.")


if __name__ == "__main__":
    main()
