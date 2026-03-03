# Multimodal MRI-based neuromarkers trace longitudinal changes in cognitive functioning in ADHD

**Code and ReadMe Author,  
**Affiliation:** University of Otago  

---

## Overview

This repository contains the code base used for our manuscript Multimodal MRI-based neuromarkers trace longitudinal changes in cognitive functioning in ADHD . The study predicts general cognitive ability (g-factor) from multi-modal brain imaging data (functional connectivity, ALFF, ReHo, and structural MRI) in the Oregon ADHD-1000 dataset using multimodal stacking machine learning, and is analysed using Linear Mixed Effects modelling.
The analysis pipeline proceeds from raw fMRI preprocessing through feature extraction, g-factor estimation, first-level predictive modelling, stacked ensemble modelling, and final statistical analysis and visualisation.

---

## Repository Structure

```
.
|-- hpc_scripts/                   # SLURM batch scripts for HPC cluster jobs
|-- feature_extraction/            # Brain imaging feature and cognitive data extraction
|-- data_preparation/              # Data alignment, fold assignment, and censoring QC
|-- g_factor/                      # G-factor CFA modelling (R)
|-- models/                        # First-level and stacked prediction models
|-- statistics/                    # LME analyses, bootstrap comparisons, and R2 decomposition
|-- visualisation/                 # All figure generation scripts
|-- notebooks/                     # Jupyter notebooks (structural descriptives)
```

### `hpc_scripts/`

SLURM array job scripts for running fMRIPrep and XCP-D used on the University of Otago's Aoraki cluster.

| File | Description |
| `fmriprep_apptainer_one_single.sl` | fMRIPrep preprocessing, subject-timepoints 0-999 |
| `fmriprep_apptainer_one_single_2.sl` | fMRIPrep preprocessing, subject-timepoints 1000-1217 |
| `xcp_d_alff_acompcor_censorMT_12_5_2025.sl` | XCP-D post-processing with motion censoring (threshold), part 1 |
| `xcp_d_alff_acompor_censorMT_12_5_2025_p2.sl` | XCP-D post-processing with motion censoring, part 2 |
| `xcp_d_alff_acompcor_nocensor_REDO_9_5_2025.sl` | XCP-D post-processing without censoring (for ALFF), part 1 |
| `xcp_d_alff_acompcor_nocensor_REDO_9_5_2025_2.sl` | XCP-D post-processing without censoring, part 2 |

### `feature_extraction/`

Scripts that extract and compute brain imaging features and cognitive test scores from raw outputs.

| File | Description |
| `Oregon_rest_connect_v18_rework.py` | Computes functional connectivity matrices from XCP-D parcellated time series; organises connectivity by Cole network ordering |
| `Oregon_ALFF_ReHo_v7.py` | Extracts ALFF (uncensored) and ReHo (censored) parcel-level values from XCP-D outputs; maps to Glasser/HCP and Cole networks |
| `Extract_cog_data_4_Gscore_v9.py` | Extracts cognitive test scores (N-Back, Stop Signal, Digit Span, etc.) from NDAR CSVs; creates 5-fold independent cross-validation splits |
| `Extract_ADHD_RS_Data_4_NCV_v7_robust_pheno.py` | Extracts ADHD Rating Scale (inattention/hyperactivity) scores matched to the final neuroimaging sample using stable longitudinal phenotypes |

### `notebooks/`

| File | Description |
| `01_OREGON_structural_tables_from_FSoutput.ipynb` | Extracts cortical area, cortical thickness, subcortical volume, and total brain volume from FreeSurfer outputs |

### `data_preparation/`

Scripts for data quality control, modality alignment, and fold structure preparation.

| File | Description |
| `Pre_processing_script_v7.py` | Organises all neuroimaging modality data into nested CV fold structures matching g-factor fold assignments |
| `Modality_alignment_fix.py` | Aligns all neuroimaging modalities (structural and functional) to the functional connectivity subject-timepoint baseline |
| `Cog_Inattention_alignment.py` | Aligns the stacked cognition predictions to the inattention dataset, retaining only matching observations |
| `Cog_Hyperactivity_alignment.py` | Aligns the stacked cognition predictions to the hyperactivity dataset, retaining only matching observations |
| `Lightweight_censoring_longitudinal_analysis_v2.py` | Assesses the impact of fMRI volume censoring (FD > 0.5mm) on longitudinal data retention, stratified by phenotype |

### `g_factor/`

| File | Description |
| `G_Factor_Oregon_Regularized_simple_v2.Rmd` | Regularized confirmatory factor analysis (CFA) to estimate a general cognitive ability (g-factor) score for each subject-timepoint using lavaan; applied independently within each cross-validation fold |

### `models/`

| File | Description |
| `First_level_model_v14.py` | Nested cross-validated first-level models (XGBoost, Kernel Ridge Regression, PLS Regression) for all neuroimaging modalities; designed for SLURM array job deployment |
| `First_level_model_v14_FC_ONLY_1.py` | Variant of the first-level model optimised for the functional connectivity modality (reduced parallelism to manage large feature space memory requirements) |
| `stacked_model_v11_ridge.py` | Stacked ridge regression ensemble that combines first-level model predictions across all modalities and model types into a final g-factor prediction |
| `Recover_stacked_individual_fold_preds.py` | Utility script to recompute per-fold performance metrics (Pearson r, R2, MAE) from the stacked model predictions CSV |

### `statistics/`

| File | Description |
| `LME_Cognition_Biomarkers_symptoms_AIO_refactored.Rmd` | Linear mixed-effects models (lme4/lmerTest) relating stacked g-factor predictions to ADHD symptoms (inattention, hyperactivity) and neuroimaging biomarkers; includes R2mlm decomposition and hierarchical partitioning |
| `LME_Cognition_Biomarkers_symptoms_AIO_refactored.html` | An HTML knit of the notebook 
| `stacked_bootstrap_comparison_v9.py` | Bootstrap resampling analysis (n=5000) comparing stacked model performance against individual model-modality combinations using Fisher's z-transformation |
| `Oregon_final_descriptives_debug_v2.py` | Generates sample demographics and descriptive statistics for the final analysis sample |

### `visualisation/`

| File | Description |
| `Stacked_Visualisations_v5.py` | Figure suite for the stacked model: predicted vs observed scatterplots, density heatmaps, longitudinal spaghetti plots stratified by phenotype |
| `Stacked_vis_spaghetti_redo.py` | Revised spaghetti plot style for the stacked model; shows trajectory lines only for subjects with multiple timepoints, trend lines for Control and ADHD groups |
| `first_level_model_visualizations_for_Manu.py` | Predicted vs observed scatterplots and spaghetti plots for all first-level model/modality combinations |
| `g-factor_spaghett_v4_phenotype_rework.py` | Longitudinal g-factor spaghetti plot with stable phenotype assignment; shows within-person trajectories and group-level trends |
| `Feat_importance_v16_correlations.py` | Correlates raw brain imaging features (ALFF, ReHo, FC) with stacked model g-factor predictions; generates brain surface and network heatmap visualisations |
| `Feat_importance_network_correlations_g.py` | Variant of the feature importance correlation script with updated output paths; uses HCP MMP parcellation and Cole network organisation used to derive network importance |
| `Haufe_feat_importance.py` | Plots Haufe-transformed feature importance scores averaged across folds, categorised by imaging modality |
| `Correlation_plots_inattention_hyp_v2.py` | Spaghetti plots of observed g-factor against inattention and hyperactivity symptom scores |
| `cor_plots_within_and_between_v4_no_box.py` | Within-person and between-person correlation plots with confidence intervals uses R notebook output as input, final visualisations for corr plots in MS |
| `m5_r2_decomposition_plot_corrected.py` | Stacked bar plot of R2mlm decomposition components (fixed within, fixed between, mean variation, residuals), used in manuscript |
| `commonality_euler_final.py` | Euler/Venn diagram visualising commonality analysis across within-person trajectories, between-person means, and age components used in manuscript |

---

## Pipeline Overview

```
Raw fMRI (BIDS)
      |
      v
fMRIPrep (hpc_scripts/)
      |
      v
XCP-D post-processing: ALFF (uncensored), ReHo + FC (motion censored) (hpc_scripts/)
      |
      v
Feature Extraction (feature_extraction/)
  - Functional connectivity matrices
  - ALFF / ReHo parcellations
  - Structural MRI (FreeSurfer via notebook)
  - Cognitive test scores
  - ADHD-RS symptom scores
      |
      v
Preprocessing & Alignment (preprocessing/)
  - Fold assignment
  - Modality alignment
  - Censoring QC
      |
      v
G-Factor Estimation per fold (g_factor/)
      |
      v
First-Level Predictive Models (models/)
  - XGBoost / KRR / PLS per modality per fold
      |
      v
Stacked Ensemble Model (models/)
      |
      v
Statistics (statistics/)
  - LME models (cognition ~ symptoms + biomarkers)
  - Bootstrap comparison vs single models
  - R2 decomposition
      |
      v
Visualisation (visualisation/)
```

---

## Dependencies

### Python

Core packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `scipy`, `statsmodels`, `matplotlib`, `seaborn`, `nilearn`, `nibabel`, `surfplot`, `tqdm`, `joblib`

Install via conda or pip.

### R

Core packages: `tidyverse`, `lme4`, `lmerTest`, `lavaan`, `psych`, `r2mlm`, `glmm.hp`, `sjPlot`, `broom.mixed`, `corrplot`

### HPC

Scripts target the University of Otago's Aoraki cluster (SLURM) using Apptainer containers for fMRIPrep 24.0.0 and XCP-D 0.10.6.

---

## Data Availability

The Oregon ADHD-1000 dataset is publicly available through the NIMH Data Archive (NDA):

> https://doi.org/10.15154/1528485

Derived feature matrices used as model inputs may be made available upon reasonable request.

---

## Citation

If you use this code, please cite:
(pre-print doi forthcoming)

---
