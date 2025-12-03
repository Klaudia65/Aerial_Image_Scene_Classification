# 🚀 Aerial Image Scene Classification: Land Use Analysis Project

This repository documents the preliminary steps of a Master's level Machine Learning project focused on classifying aerial imagery. We cover problem definition, feature engineering, exploratory data analysis (EDA), and establishing a performance baseline.

## 🎯 1. Project Goal and Problem Definition (Step 1)

### Business Objective
The primary objective is to **develop an automated tool for classifying land use** (urban, agricultural, natural, etc.) from aerial images. This is a critical task in remote sensing and urban planning.

### Data Source
The project utilizes the **UC Merced Land Use Dataset**, featuring 21 distinct land use scene categories.

### ML Problem Type
This is a **Supervised Image Classification** problem.

**Research Question:** "Which Deep Learning model (CNN) achieves the highest accuracy in classifying the 21 land use scenes of the UC Merced dataset, leveraging techniques like Transfer Learning and Data Augmentation?"

### Variables
* **Target Variables (y):** The **21 land use classes** (e.g., `baseball diamond`, `freeway`, `forest`, etc.).
* **Descriptive Variables (X):** The image data (256x256x3 pixels).

---

## 2. 📝 Data Preparation and Feature Engineering (Step 2 & 3)

### Data Integrity and Missing Values
The dataset organization ensures data integrity:
* **Missing Labels:** No missing labels were found, as the class label is determined by the parent folder name (e.g., the folder name `airplane` is the label).
* **Missing Values (NoData):** The images exhibit black borders due to orthorectification. These were treated as "NoData" zones and their influence was accepted/minimized through necessary image resizing during the preprocessing phase.

### Baseline Feature Extraction (Statistical Features)
To establish a quick benchmark, we extracted simple statistical features from the RGB channels for each image, resulting in a DataFrame (`image_features.csv`).

| Feature Name | Description |
| :--- | :--- |
| $\text{mean\_R}, \text{mean\_G}, \text{mean\_B}$ | Mean intensity of the RGB channels. |
| $\text{std\_R}, \text{std\_G}, \text{std\_B}$ | Standard deviation (variance) of the RGB channels. |
| $\text{mean\_ExG}$ | Mean of the **Excess Green Index** ($\text{ExG} = 2G - R - B$), used to highlight vegetation. |

---

## 3. 📈 Exploratory Data Analysis (EDA) and Normalization (Step 3)

### Univariate and Bivariate Analysis

#### Univariate Analysis (Distribution)
We analyzed the distribution of each feature across all images (e.g., histogram for $\text{mean\_ExG}$).
* **Visualization:** 

[Image of Histogram illustrating univariate analysis]

* **Purpose:** To check for symmetry, skewness, and potential outliers before normalization.

#### Bivariate Analysis (Boxplots)
Boxplots were generated to study the distribution of a single feature ($\text{mean\_ExG}$) across the 21 target classes. This checks the **discriminative power** of the feature.
* **Visualization:** 

[Image of Boxplot illustrating bivariate analysis]

* **Key Observations:**
    * **Separability:** Classes with distinct quartile boxes (e.g., **"forest" vs "runway"**) are easily distinguishable by this feature.
    * **Overlap:** Classes with heavily overlapping boxes (e.g., **"dense residential" vs "medium residential"**) are harder to separate.
    * **Relevance:** Vegetation classes (forest, agricultural) exhibit higher $\text{mean\_ExG}$ values than classes composed of inert materials (buildings, freeways).

### Statistical Correlation (ANOVA Test)

The ANOVA test was used to statistically confirm that the differences observed in the Boxplots are not due to chance, testing the **Null Hypothesis ($\mathbf{H_0}$):** that the means of the feature are the same across all 21 classes.

The results strongly indicate that the basic features are highly useful discriminators:

| Feature Analyzed | F-Statistic | P-value ($p < 0.05$) | Conclusion |
| :--- | :--- | :--- | :--- |
| $\text{mean\_R}$ | $172.63$ | $0.000\text{e}+00$ | **Reject $H_0$** (Highly Significant) |
| $\text{mean\_G}$ | $170.43$ | $0.000\text{e}+00$ | **Reject $H_0$** (Highly Significant) |
| $\text{mean\_ExG}$ | $79.10$ | $1.113\text{e}-300$ | **Reject $H_0$** (Highly Significant) |
| $\text{std\_G}$ | $76.15$ | $1.187\text{e}-289$ | **Reject $H_0$** (Highly Significant) |
| $\text{std\_B}$ | $97.99$ | $0.000\text{e}+00$ | **Reject $H_0$** (Highly Significant) |

**Interpretation:** The P-values confirm that the observed differences are highly improbable to be due to random chance. **Rejecting $\mathbf{H_0}$** means the feature is statistically useful for classification.

### Normalization
The dataset was prepared for scale-sensitive models (like Neural Networks) by applying **Standardization (Z-score)** to all numerical features. This is performed to accelerate model convergence, although it does not affect correlation.

---

## 4. 📉 Baseline Model Performance (Step 3: Prediction Task)

A **Random Forest Classifier** was used on the extracted statistical features to establish a performance benchmark.

### Model Setup
* **Model:** Random Forest Classifier (Insensitive to scale/normalization)
* **Features:** Statistical features (e.g., $\text{mean\_R}, \text{mean\_ExG}$)
* **Data Split:** $80\%$ Train, $20\%$ Test (stratified).

### Code Snippet: Baseline Evaluation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ... data loading and splitting steps ...

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) 
y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the Test Set: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

```
### Results

precision    recall  f1-score   support

     agricultural       0.37      0.32      0.34       100
         airplane       0.21      0.18      0.19       100
  baseballdiamond       0.46      0.44      0.45       100
            beach       0.33      0.37      0.35       100
        buildings       0.15      0.15      0.15       100
        chaparral       0.39      0.51      0.44       100
 denseresidential       0.24      0.19      0.21       100
           forest       0.45      0.54      0.49       100
          freeway       0.13      0.17      0.15       100
       golfcourse       0.32      0.37      0.34       100
           harbor       0.60      0.64      0.62       100
     intersection       0.11      0.13      0.12       100
mediumresidential       0.15      0.15      0.15       100
   mobilehomepark       0.26      0.31      0.28       100
         overpass       0.22      0.20      0.21       100
       parkinglot       0.19      0.19      0.19       100
            river       0.34      0.32      0.33       100
           runway       0.19      0.14      0.16       100
sparseresidential       0.19      0.17      0.18       100
     storagetanks       0.23      0.17      0.20       100
      tenniscourt       0.15      0.12      0.13       100

         accuracy                           0.28      2100
        macro avg       0.27      0.28      0.27      2100
     weighted avg       0.27      0.28      0.27      2100


**Accuracy (Overall)	0.28 (28%)	2100**

### Conclusion and Next Steps

Performance is Poor: An accuracy of 28% is significantly better than random guessing (≈4.8%), but it means the model is incorrect more than two-thirds of the time.

Justification for Step 4: The basic features (means, standard deviations, ExG) are insufficient to correctly distinguish the 21 land use classes as they only capture simple color and brightness information.

**Need for CNN**: To significantly improve accuracy, the next step must focus on implementing the core Deep Learning pipeline to extract deep features (CNN Embeddings). These features will capture the essential textures, shapes, and spatial patterns needed for high-fidelity classification.
