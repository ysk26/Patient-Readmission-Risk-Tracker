# Patient-Readmission-Risk-Tracker
End-to-end ML pipeline predicting 30-day hospital readmissions using Random Forest + SMOTE on 101,766 real patient records

Patient Readmission Risk Tracker
An end-to-end machine learning pipeline that predicts which diabetic patients are likely to be readmitted to hospital within 30 days of discharge, enabling care teams to intervene before patients leave.
Built entirely in Google Colab using Python. No paid tools, no cloud infrastructure, just real data, real code, and a real clinical problem.
---
The Problem
30-day hospital readmissions cost the US healthcare system $26 billion annually. Medicare penalizes hospitals with high readmission rates, and most of these readmissions are preventable with early intervention.
The challenge: by the time a patient is readmitted, it's too late. This project flips that, identifying high-risk patients before discharge so care teams can act.
---
What It Does

Raw patient data (101,766 records)
        ->
Data cleaning & feature engineering
        ->
Temporal train/test split (past → future)
        ->
SMOTE balancing (training only)
        ->
Random Forest model (200 trees)
        ->
Risk scoring - High / Medium / Low
        ->
Color-coded Excel report + automated email
```
---
Dataset
Source: UCI Machine Learning Repository - Diabetes 130-US Hospitals (1999–2008)
Records: 101,766 patient encounters across 130 US hospitals
Features: 50 original columns → 20 selected clinical features
Target: Readmitted within 30 days (`<30`) vs not (`NO` or `>30`)
Class distribution: 8.8% high-risk, 91.2% low-risk
---
Model Details
Parameter	Value
Algorithm	Random Forest Classifier
Trees	200
Max depth	12
Class weight	{0: 1, 1: 3}
SMOTE strategy	0.5 (training only)
Threshold	0.45
Validation	Temporal split (80/20 by encounter ID)
Why temporal split?
Most tutorials use a random 80/20 split. In healthcare that creates temporal data leakage - the model accidentally learns from "future" patients. I sorted by `encounter_id` as a time proxy and trained on earlier encounters, tested on later ones. This is how production healthcare models are actually validated.
Why SMOTE on training only?
The dataset is severely imbalanced - only 9% of patients are high-risk. SMOTE generates synthetic high-risk samples to help the model learn minority class patterns. Critically, SMOTE was applied only to training data. Applying it to test data would artificially inflate results and misrepresent real-world performance.
---
Results:
Accuracy          : 62.38%
High Risk Recall  : 40%  (367 out of 920 actual high-risk patients)
Low Risk Precision: 94%  (when model says safe, it's right 94% of the time)
Threshold         : 0.45

Threshold comparison
Threshold	High-risk caught	Patients flagged
0.20	95% (878/920)	13,064
0.30	81% (746/920)	10,317
0.40	52% (476/920)	6,653
0.45	40% (367/920)	5,048 ✅
0.50	30% (274/920)	3,779
0.45 was chosen as the best balance between recall and a manageable care team workload.

Top 5 predictive features
Feature	Importance	Clinical meaning
`change`	0.147	Medication changed during stay = unstable condition
`metformin`	0.104	Primary diabetes drug dosage changes
`insulin`	0.096	Insulin dependency = complex diabetes
`gender`	0.079	Gender differences in diabetes outcomes
`discharge_disposition_id`	0.071	Where patient went after discharge
These features match published clinical research on diabetic readmissions - confirming the model learned real medical signals, not noise.

Clinical value
Detection rate
Without model (clinical intuition)	~30% of high-risk patients
With this model	~40% of high-risk patients
Improvement	~92 additional patients identified per discharge cycle
---
Patient Risk Output
After scoring all 69,569 patients:

🔴 High Risk    :  4,841 patients → Call within 24 hours
🟡 Medium Risk  : 22,010 patients → Schedule follow-up in 3 days
🟢 Low Risk     : 42,751 patients → Standard discharge care plan

Top 5 highest-risk patients had scores of 93.0%, 92.5%, 92.2%, 91.9%, 91.7% - all aged 65–95, which is clinically expected for complex diabetic patients.
---
Features Used
Numerical (12)
`age`, `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`, `admission_type_id`, `discharge_disposition_id`, `admission_source_id`
Categorical (8) — Label Encoded
`race`, `gender`, `insulin`, `diabetesMed`, `change`, `metformin`, `glipizide`, `glyburide`
---
Tech Stack
Tool	Purpose
Python	Core language
Google Colab	Free cloud notebook environment
pandas	Data cleaning and manipulation
scikit-learn	Random Forest, LabelEncoder, metrics
imbalanced-learn	SMOTE oversampling
openpyxl	Color-coded Excel report generation
Gmail SMTP	Automated daily email delivery
Kaggle API	Dataset download
---
Project Structure

readmission-risk-tracker/
│
├── Patient_Readmission_Risk_Tracker.ipynb   # Main notebook
├── diabetic_data.csv                        # Dataset (download from Kaggle)
├── IDS_mapping.csv                          # Diagnosis code mapping
├── Readmission_Risk_Report_YYYY-MM-DD.xlsx  # Generated output
└── README.md

---
How to Run
Open `Patient_Readmission_Risk_Tracker.ipynb` in Google Colab
Download the dataset from Kaggle: `brandao/diabetes`
Run cells in order (1 → 24)
Update Cell 22 with your Gmail credentials (use App Password, not regular password)
The report will be emailed and available for download after Cell 23/24
> **Security note:** Never commit your Gmail App Password to GitHub. Use environment variables or Colab Secrets (key icon in left sidebar).
---
Model Limitations
Dataset covers 1999–2008 — clinical practices have evolved since then
`encounter_id` used as a time proxy (exact dates not available in public dataset)
LabelEncoder used for categorical features - OneHotEncoding may perform better for linear models
Model validated on held-out data from the same dataset, not a completely independent external dataset
---
