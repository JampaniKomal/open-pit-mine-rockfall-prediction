# Open-Pit Mine Rockfall Risk Assessment via Data Analytics & Visualization

_A Project for Data Analytics & Visualization (DAV) Course_

This project addresses problem statement **SIH25071** from the Smart India Hackathon, adapted for the Data Analytics & Visualization (DAV) course (G5AD21DAV) at Rashtriya Raksha University. It focuses specifically on **rockfall prediction in open-pit mining operations** using advanced data analytics and machine learning.

---

## 1. Project Overview

The objective is to develop a **robust risk assessment system** for predicting rockfall events. This system uses a meticulously **engineered high-fidelity synthetic dataset** where critical environmental factors are **statistically modeled on real-world Kaggle distributions** to ensure realism.

We classify rockfall risk into four distinct categories: **Low, Medium, High, and Critical**. The final deployed model is the **XGBClassifier**, selected for its high robustness and safety performance against the challenge of imbalanced data.

### Our Data Strategy: Statistical Realism (The DAV Method)

We utilize **High-Fidelity Synthetic Modeling informed by Real Data Statistics**. This corrects the methodological flaw of merging unrelated industrial datasets.

#### Method Explanation:
1. **Real-World Sourcing:** We analyzed the statistical distribution (the "recipe") of **real-world rainfall data** (for zero-inflation) and **real-world seismic data** (for long-tail magnitude frequency) from public Kaggle files.
2. **High-Fidelity Generation:** We used those exact statistics (e.g., 20.17% zero-rainfall days, mean 1.77 seismic magnitude) to programmatically generate a clean, unified 20,000-sample dataset where the environmental factors behave realistically.
3. **Logical Correlation:** We engineered strong, logical relationships and ensured **displacement** is the primary, strongest driver of risk, which the model must learn.

---

## 2. Core Objectives

- **Data Generation & Sourcing:** Create a high-fidelity synthetic dataset by **statistically modeling** external factors using two specific Kaggle datasets.
- **Exploratory Data Analysis:** Perform deep EDA, visualizing feature distributions, correlations (heatmap), and feature-to-target relationships (box plots) to prove our engineered logic.
- **Preprocessing & Modeling:** Apply correct preprocessing (StandardScaler) and use the **class_weight='balanced'** strategy to train models that excel on our **imbalanced dataset**.
- **Model Evaluation & Interpretation:** Select the **XGBClassifier** as the final model based on its **99% Critical Recall and high robustness** using the **Confusion Matrix** and **Built-in Feature Importance**.

---

## 3. Dataset Composition

Our final dataset (`rockfall_synthetic_data.csv`, 20,000 samples) is defined by its realistic structure:

### Geotechnical Features
| Feature | Role in Project | Modeling Strategy |
| :--- | :--- | :--- |
| **Displacement (mm)** | **Primary Predictor.** Engineered as the main indicator of risk. | Custom logic derived from combined factors. |
| **Joint Water Pressure (kPa)** | Correlated factor, directly influenced by rainfall events. | Random distribution informed by rainfall statistics. |
| **Seismic Activity (Magnitude)** | Contributor to high risk events. | Modeled on **All the Earthquakes Dataset** (long-tail distribution). |
| **Rainfall (mm/24h)** | Weakest predictor alone, but key driver of water pressure. | Modeled on **Rainfall Dataset** (20.17% zero-inflation). |

### Risk Distribution (Realistic Imbalance)
- **Low Risk:** ~50.5%
- **Medium Risk:** ~40.5%
- **High Risk:** ~7.3%
- **Critical Risk:** **~1.7%** (The model's ability to find these rare events is the primary safety objective).

---

## 4. Technology Stack

- **Language:** Python 3.13+
- **Data Science & ML:** pandas, numpy, scikit-learn, **xgboost**
- **Data Visualization:** matplotlib, seaborn, plotly
- **Model Interpretability:** Feature Importance (Built-in)
- **Deployment:** Streamlit

---

## 5. Setup Instructions

### IMPORTANT: Kaggle API Setup (Required!)

This project downloads real **Rainfall and Seismic driver data** from Kaggle. You **must** set up Kaggle API credentials before running the notebooks.

#### Step 1: Get Your Kaggle API Token
1. Go to [Kaggle.com](https://www.kaggle.com/) and sign in (create account if needed)
2. Click on your profile picture (top right) → **Account**
3. Scroll down to **API** section
4. Click **Create New API Token**
5. This downloads a file called `kaggle.json` to your computer

#### Step 2: Place kaggle.json in the Correct Location
**Windows Users:**
```powershell
# Final location: C:\Users\<YourUsername>\.kaggle\kaggle.json
mkdir C:\Users\<YourUsername>\.kaggle
Move-Item -Path ~\Downloads\kaggle.json -Destination C:\Users\<YourUsername>\.kaggle\kaggle.json
````

**Mac/Linux Users:**

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### Step 3: Accept Dataset Terms on Kaggle

**IMPORTANT:** Before running the notebooks, you must accept the dataset's terms for the two driver datasets used in **Notebook 1**:

1.  Visit: [Rainfall Dataset for Simple Time Series Analysis](https://www.kaggle.com/datasets/sujithmandala/rainfall-dataset-for-simple-time-series-analysis)
2.  Visit: [All the Earthquakes Dataset : from 1990-2023](https://www.kaggle.com/datasets/alessandrolobello/the-ultimate-earthquake-dataset-from-1990-2023)
3.  Log in to your Kaggle account and click the **"Download"** button on both pages (this accepts the terms).

-----

### How to Run the Project

1.  **Clone Repository:**

    ```powershell
    git clone [repository-link-here]
    cd open-pit-mine-rockfall-prediction
    ```

2.  **Create Virtual Environment and Install Dependencies:**

    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1    # (Windows)
    # source .venv/bin/activate   # (Mac/Linux alternative)
    pip install -r requirements.txt
    ```

3.  **Run Notebooks Sequentially:**
    Open the notebook files (e.g., using **VS Code** or **Jupyter**) and execute them in this **new, clean order**:

    1.  **`01_data_sourcing_and_generation.ipynb`**
    2.  **`02_exploratory_data_analysis.ipynb`**
    3.  **`03_preprocessing_and_modelling.ipynb`**
    4.  **`04_model_interpretation_and_results.ipynb`**

4.  **Run the Final Deployment App:**
    After running all four notebooks successfully, execute the final interactive web application:

    ```powershell
    streamlit run app.py
    ```

### Streamlit App Highlights

- **Overview tab:** Summarises dataset scale, class imbalance, and feature ranges derived directly from Notebook 2.
- **Predict tab:**
    - Manual sliders flag when values stray from the 10-90% training range.
    - Scenario presets replicate common slope conditions (Low -> Critical).
    - Batch uploader auto-aligns column headings (`displacement`, `Slope Displacement (mm)`, etc.) before scoring.
- **Diagnostics tab:** Rebuilds the stratified test split to render the confusion matrix, classification report, and XGBoost feature importance.
- **About tab:** Recaps the four-notebook workflow and artefacts stored under `models/` and `data/`.

-----

## 6\. Acknowledgement

This project was developed with assistance from Gemini, a large language model by Google, which helped re-engineer the data strategy, validate the model interpretation, and create the robust final workflow.
## Future Roadmap
- [ ] **Real-time Integration:** Connect to live sensor feeds via IoT MQTT brokers.
- [ ] **Advanced Model Architecture:** Experiment with LSTM/Transformers for temporal sequence prediction.
- [ ] **Cloud Deployment:** Containerize with Docker and deploy to AWS/Azure for scalability.
