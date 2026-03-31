# Repository content

This repository contains all the Python scripts to reproduce results obtained in the paper 'Embedding response surfaces in kinetic models predicts microalgae growth across light and nutrient gradients'. It also contains the open dataset generated for this study: 1,315 growth curves of microalga C. reinhardtii in batch cultures in two volumes (200 µL in 96 wells microplate and 50 mL in flask) with varying light intensity and cell medium (TAP) dilution conditions. All the cultures were performed in a shaking incubator at 100 rpm with temperature and CO2 rate regulated at 25°C and 1.5%, respectively.

## Datasets
### C. reinhardtii (mixotrophic and photoautotrophic cultures)
Time series of C. reinhardtii cultures and their pre-processing are in the folder `all_data`. 

### Monoraphidium sp. (photoautotrophic cultures)
Time series of Monoraphidium sp. cultures can be obtained in [Kambe et al. 2022] repository available at the following link: https://github.com/hori-group/logistic_eq_for_cultivation_planning
The dataset is contained in the folder `_OD_excel_data` in the file `OD_data.xlsx`. By running the script `format_Kambe_2022_data.py` from this repository it will create a folder `data_Kambe` containing 4 csv files with OD time series of Monoraphidium at four light intensities (0.274, 0.521, 1.09, 2.92 $\mu mol_{h\nu} \cdot s^{-1}$) containing each 4 dilutions of the BG-11 medium conditions $C_{0} = 1, 0.5, 0.25, 0.125, 0.0625$ in the column 'mean 0', 'mean 1', 'mean 2', 'mean 3' and 'mean 4', respectively.

All details concerning the collecting of C.reinhardtii data and their pre-processing are available in the 'Methods' section of 'Embedding response surfaces in kinetic models predicts microalgae growth across light and nutrient gradients' paper.

## Models
The models used to benchmark RSM-ODE model are : 
- LMMB developed in [Kambe et al. 2022]
- DMB developed in [Martínez et al. 2020]
- MMmB developed in this study

All details concerning these model are available in the 'Methods' section of 'Embedding response surfaces in kinetic models predicts microalgae growth across light and nutrient gradients' paper. 

## Scripts
This is a description of the purpose of each scripts: 

- `comparison_TAP_BG11.py` allows the comparison of experimental growth data of C. reinhardtii in mixotrophic conditions (TAP medium) and photoautotrophic conditions (BG-11 medium) for two light intensity. 
- `data_import.py` allows to import the different dataset (Monoraphidium sp. and C. reinhardtii growth curves)
- `DMB_sliders_erlen.py` generates the visual adjustment of Droop model from [Martínez et al. 2020] on C. reinhardtii flask cultures data
- `DMB_sliders_plate.py` generates the visual adjustment of Droop model from [Martínez et al. 2020] on C. reinhardtii microplate cultures data
- `final_dataset_composition.py` is applying linearity correction steps to the C. reinhardtii growth data in microplate 
- `Monoraphidium_data_RSM_ODE.py` is fitting the RSM-ODE model on Monoraphidium sp. growth data from [Kambe et al. 2022]
- `Monoraphidium_data_LMMB.py` is fitting the LMMB model developed in [Kambe et al. 2022] on Monoraphidium sp. growth data to reproduce the results authors obtains and compute a global $R^2$. 
- `Chlamydomonas_data_RSM_ODE_erlen.py` is fitting the RSM-ODE model on C. reinhardtii flask cultures data
- `Chlamydomonas_data_RSM_ODE_plate.py` is fitting the RSM-ODE model on C. reinhardtii plate cultures data
- `growth_models.py` contains growth models 
- `growth_rate_analysis.py` is a script that can compute the growth rate on growth time series 
- `Chlamydomonas_data_erlen_LMMB.py` is applying the same fitting strategy used in [Kambe et al. 2022] for LMMB model on C. reinhardtii flask culture dataset.
- `LMMB_sliders_erlen.py` generates the visual adjustment of LMMB model from [Kambe et al. 2022] on C. reinhardtii flask cultures data
- `LMMB_sliders_plate.py` generates the visual adjustment of LMMB model from [Kambe et al. 2022] on C. reinhardtii microplate cultures data
- `MMmB_sliders_erlen.py` generates the visual adjustment of MMmB model on C. reinhardtii flask cultures data. This model was introduced for this specific study and takes into account a death rate depending on light intensity. 
- `MMmB_sliders_plate.py` generates the visual adjustment of MMmB model on C. reinhardtii microplate cultures data.
- `requirements.txt` contains all the version of Python libraries used for this work. 
- `plot_microalgae_data.py` is used to plot a comparative figure between growth curves dataset on Monoraphidium sp. and C. reinhardtii
- `rsm_surface_comparison_erlen.py` generates a comparative study between surface functions used for $\mu_{max}(C_{0}, L_{0})$ and $N_{max}(C_{0}, L_{0})$ for C. reinhardtii flask culture dataset.
- `rsm_surface_comparison_plate.py` generates a comparative study between surface functions used for $\mu_{max}(C_{0}, L_{0})$ and $N_{max}(C_{0}, L_{0})$ for C. reinhardtii plate culture dataset.
- `format_Kambe_2022_data.py` allows to format the data from [Kambe et al. 2022] contains in a `.xlsx` file into `.csv` files
- `cyto_processing.py` allows to process flow cytometry data flow_cytometry_data.fcs in `all_data` folder

## Setup
The code is running with Python 3.12.8. All libraries needed to run the code can be installed with the command:
```bash
pip3 install -r requirements.txt
```

Warning: The following scripts needs to be run after running `Chlamydomonas_data_RSM_ODE_erlen.py`, `Chlamydomonas_data_RSM_ODE_plate.py` otherwise t_lag adjustments files will not be created:
- `DMB_sliders_erlen.py` and `DMB_sliders_plate.py`
- `LMMB_sliders_erlen.py` and `LMMB_sliders_plate.py`
- `MMmB_sliders_erlen.py` and `MMmB_sliders_plate.py`
- `scaling_surfaces.py`

## Supplementary information
This repository contains supplementary data and materials associated with the paper "Embedding response surfaces in kinetic models predicts microalgae growth across light and nutrient gradients." These resources are provided to support transparency, reproducibility, and reuse.

### Supplementary data
The following datasets, not shown in the main manuscript, are available for visualization and reuse:

- Time-series chlorophyll fluorescence measurements (excitation 430 nm, emission 645 nm) for both flask and microplate cultures at $L_{0} = 102$ and $170 \mu mol_{h\nu} \cdot s^{-1}$ across all five medium dilutions $C_{0} = 1, 1/2, 1/4, 1/8$ and $1/16$. For microplate cultures, intermediate dilutions ($C_{0} = 1 − 0.05·i$, for $i = 0$ to $17$) were additionally measured.
- Confocal microscopy images (Leica TCS SP8) of cells cultivated in flasks at $L_{0} = 25.5 \mu mol_{h\nu} \cdot s^{-1}$  across all five medium dilutions.
- Flow cytometry analysis (Guava easyCyte, EMD Millipore) of microalgae population structure at $L_{0} = 11.9 \mu mol_{h\nu} \cdot s^{-1}$ for five TAP medium dilutions.
- RSM cross-sections and sensitivity analysis derived from microplate data.

These data can be visualized in the `supplementary_figures.pdf` document in `all_data` folder.

### Reproducibility

To facilitate reproducibility, the document `supplementary_tables.pdf` in `all_data` folder provides fitted parameter values and performance metrics for all models tested on the dataset.

## References
[Kambe et al. 2022] Kazuki Kambe, Yasutaka Hirokawa, Asuka Koshi, and Yutaka Hori. A parametric logistic equation with light
flux and medium concentration for cultivation planning of microalgae. Journal of The Royal Society Interface,
19(191):20220166, 2022. doi: 10.1098/rsif.2022.0166. URL https://royalsocietypublishing.org/doi/abs/10.1098/rsif.2022.0166.

[Martínez et al. 2020] Carlos Martínez, Francis Mairet, and Olivier Bernard. Dynamics of the periodically forced light-
limited droop model. Journal of Differential Equations, 269(4):3890–3913, 2020. ISSN 0022-0396. doi: https://doi.org/10.1016/j.jde.2020.03.020. URL https://www.sciencedirect.com/science/article/pii/S0022039620301273.
