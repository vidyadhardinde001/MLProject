# Unsupervised Anomaly Detection in Alzheimer's Disease Cognitive Trajectories Using Machine Learning

## Abstract
Alzheimer's disease (AD) exhibits heterogeneous progression patterns across patients, with some showing rapid cognitive decline while others remain stable for extended periods. Identifying atypical disease trajectories is crucial for personalized treatment strategies and clinical trial design. This paper presents a comprehensive unsupervised machine learning pipeline for detecting anomalous cognitive progression patterns in longitudinal AD patient data. We analyze 1,943 patients from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset with 11,017 longitudinal visits spanning multiple years. Our approach combines trajectory-based feature engineering, dimensionality reduction via Principal Component Analysis (PCA), and dual anomaly detection using Isolation Forest and DBSCAN clustering algorithms. The pipeline extracts 46 temporal features capturing cognitive decline slopes, acceleration, variability, and change-from-baseline patterns, subsequently reduced to 11 principal components. Results demonstrate the detection of 60 anomalous patients (8.1% of the cohort) with 59.8% inter-method agreement and 0.900 cross-validation correlation across temporal splits. Temporal consistency validation confirms robust early detection capability at 12-month follow-up. The identified atypical patients exhibit clinically significant deviations in MMSE and FAQ assessment trajectories, providing valuable insights for AD progression research and patient stratification.

**Keywords**: Alzheimer's disease, anomaly detection, longitudinal analysis, machine learning, DBSCAN, Isolation Forest, cognitive trajectories

## I. INTRODUCTION

### A. Background and Motivation
Alzheimer's disease (AD) is a progressive neurodegenerative disorder affecting over 50 million people worldwide, characterized by cognitive decline, memory loss, and functional impairment [1]. While the general trajectory of AD progression is well-documented, substantial heterogeneity exists across individual patients. Some patients experience rapid cognitive deterioration within months, while others maintain cognitive stability for years despite biomarker evidence of disease [2].

Understanding this heterogeneity is critical for multiple clinical applications:
* Personalized Treatment: Identifying patients with atypical progression enables tailored therapeutic interventions
* Clinical Trial Design: Stratifying patients by progression patterns improves trial power and reduces sample size requirements
* Prognostic Modeling: Early detection of atypical trajectories informs care planning and resource allocation
* Disease Mechanism Research: Studying outlier cases reveals novel pathophysiological pathways

### B. Research Objectives
This work addresses the following research questions:

1. Can unsupervised machine learning identify clinically meaningful atypical AD progression patterns in longitudinal data?
2. What proportion of AD patients exhibit trajectories significantly deviating from population norms?
3. Do anomaly detection methods (Isolation Forest vs. DBSCAN) converge on consistent patient subsets?
4. Are early-detected anomalies (12-month data) predictive of long-term atypical progression?

### C. Contributions
Our primary contributions include:

1. A comprehensive trajectory feature engineering framework extracting temporal patterns from longitudinal cognitive assessments
2. A dual anomaly detection pipeline combining tree-based and density-based methods with consensus scoring
3. Temporal cross-validation methodology ensuring early detection and longitudinal consistency
4. Validation on real-world ADNI dataset with 1,943 patients and 11,017 visits
5. Open-source implementation enabling reproducibility and extension

## II. METHODOLOGY

### A. Dataset
The Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset, specifically the ADNIMERGE merged table (June 2024 release), contains:

* Participants: 1,943 individuals (CN, MCI, and AD patients)
* Longitudinal Visits: 11,017 total assessments over 10+ years
* Visit Schedule: Baseline, 6-month, 12-month, and annual follow-ups
* Cognitive Assessments: MMSE, FAQ, ADAS-Cog, CDR-SB
* Demographics: Age, gender, education, APOE4 genotype
* Clinical Diagnosis: Updated at each visit

### B. Feature Engineering
We extract 46 temporal features across 6 categories:

1. **Slope Features** (12 features):
   * Per assessment (MMSE, FAQ) and time interval
   * Formula: $\text{slope}_{t_1 \to t_2} = \frac{\text{score}_{t_2} - \text{score}_{t_1}}{t_2 - t_1}$

2. **Acceleration Features** (6 features):
   * Second derivative of cognitive decline
   * Formula: $\text{acceleration} = \frac{\text{slope}_{t_2} - \text{slope}_{t_1}}{t_2 - t_1}$

3. **Variability Features** (4 features):
   * Standard deviation and coefficient of variation
   * Formula: $CV = \frac{\sigma(\text{trajectory})}{\mu(\text{trajectory})}$

4. **Change-from-Baseline** (8 features):
   * Absolute and percentage changes
   * Formula: $\Delta_t = \text{score}_t - \text{score}_0$

5. **Ratio Features** (8 features):
   * Relative changes between visits
   * Formula: $\text{ratio}_t = \frac{\text{score}_t}{\text{score}_{t-1}}$

6. **Timing Features** (8 features):
   * Time to reach specific decline thresholds

[Sections continue with similar formatting...]

## References

[1] Alzheimer's Association, "2024 Alzheimer's disease facts and figures," _Alzheimer's & Dementia_, vol. 20, no. 5, pp. 3708-3821, 2024.

[2] J. L. Whitwell et al., "Neuroimaging correlates of pathologically defined subtypes of Alzheimer's disease: a case-series study," _Lancet Neurology_, vol. 11, no. 10, pp. 868-877, 2012.

[3] V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," _ACM Computing Surveys_, vol. 41, no. 3, pp. 1-58, 2009.

[Continue with remaining references...]

## Author Information

This work was conducted as part of advanced research in machine learning applications for healthcare at [Institution Name].

**Authors**: [List authors with their PRNs and contributions]

**Contact**: [Corresponding author's email]

**GitHub Repository**: https://github.com/sagarM1729/Anomaly-Detection-in-Cognitive-Trajectories