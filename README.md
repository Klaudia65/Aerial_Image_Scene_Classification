# 🚀 Aerial Image Scene Classification: Land Use Analysis Project

This repository documents the preliminary steps of a Master's level Machine Learning project focused on classifying aerial imagery. We cover problem definition, feature engineering, exploratory data analysis (EDA), and establishing a performance baseline.

## 1. Project Goal and Problem Definition (Step 1)

### Business Objective
The primary objective is to **develop an automated tool for classifying land use** (urban, agricultural, natural, etc.) from aerial images. This is a critical task in remote sensing and urban planning.

### Data Source
The project utilizes the[**UC Merced Land Use Dataset**](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset), featuring 21 distinct land use scene categories, from  USGS National Map Urban Area Imagery.

### ML Problem Type
This is a **Supervised Image Multi-class Classification** problem.

**Research Question:** "Which learning model achieves the highest accuracy in classifying the 21 land use scenes of the UC Merced dataset, leveraging techniques like Transfer Learning and Data Augmentation?"

### Variables
* **Target Variables (y):** The **21 land use classes** (e.g., `baseball diamond`, `freeway`, `forest`, etc.).
* **Descriptive Variables (X):** The image data (256x256x3 pixels).

---

## 2. Data Preparation and Feature Engineering

### Data Integrity and Missing Values
The dataset organization ensures data integrity:
* **Missing Labels:** No missing labels were found, as the class label is determined by the parent folder name (e.g., the folder name `airplane` is the label).
* **Missing Values (NoData):** The images exhibit black borders due to orthorectification. These were can be consideres as "NoData" zones but their influence is accepted due to the regularity of the border proportion in every class.

### Baseline Feature Extraction (Statistical Features)
To establish a quick benchmark, we extracted simple statistical features from the RGB channels for each image, resulting in a DataFrame (`image_features.csv`).

| Feature Name | Description |
| :--- | :--- |
| mean_R, mean_G, mean_B | Mean intensity of the RGB channels. |
| std_R, std_G, std_B | Standard deviation (variance) of the RGB channels. |
| mean_ExG | Mean of the **Excess Green Index** (ExG = 2G - R - B), used to highlight vegetation. |

---

## 3. Exploratory Data Analysis (EDA) and Normalization

### Univariate and Bivariate Analysis

#### Univariate Analysis (Distribution)

[visualization/ditributions.png]

The graphs generally show shapes close to Gaussian distributions,
suggesting a certain overall homogeneity in the dataset while maintaining sufficient variability between images.
The distribution of mean_G is relatively centered and not very extensive, indicating that the
majority of images share similar levels of green, regardless of their category. Conversely, mean_ExG reflects strong heterogeneity, particularly between scenes rich in vegetation and urban, maritime, or sandy areas. Finally, std_G highlights differences in texture between images. Some are very uniform with low variance (e.g., agricultural fields or forests), while others are visually complex (different types of composition) with high variance.

#### Bivariate Analysis (Boxplots)
Boxplots were generated to study the distribution of a single feature (mean_ExG) across the 21 target classes. This checks the **discriminative power** of the feature.


[visualization/boxplots.png]

* **Key Observations:**
    * **Separability:** Classes with distinct quartile boxes (e.g., **"forest" vs "runway"**) are easily distinguishable by this feature.
    * **Overlap:** Classes with heavily overlapping boxes (e.g., **"dense residential" vs "medium residential"**) are harder to separate.
    * **Relevance:** Vegetation classes (forest, agricultural) exhibit higher mean_ExG values than classes composed of inert materials (buildings, freeways).

* **Conclusion**
    No single feature is sufficient to distinguish all classes. Mean_ExG is the most discriminating for vegetation classes, while mean_G and std_G help to separate vegetation classes from urban classes. These results highlight the need to combine several features, or even use advanced models such as CNNs, to improve classification accuracy.


### Statistical Correlation (ANOVA Test)

The ANOVA test was used to statistically confirm that the differences observed in the Boxplots are not due to chance, testing the **Null Hypothesis (H₀):** that the means of the feature are the same across all 21 classes.

The results strongly indicate that the basic features are highly useful discriminators:

| Feature Analyzed | F-Statistic | P-value (p < 0.05) | Conclusion |
| :--- | :--- | :--- | :--- |
| mean_G | 170.43 | 0.000e+00 | **Reject H₀** (Highly Significant) |
| mean_ExG | 79.10 | 1.113e-300 | **Reject H₀** (Highly Significant) |
| std_G | 76.15 | 1.187e-289 | **Reject H₀** (Highly Significant) |

**Interpretation:** The P-values confirm that the observed differences are highly improbable to be due to random chance. **Rejecting H₀** means the feature is statistically useful for classification.

### Pearson Correlation

#### Redundancy of characteristics
The strong correlations between mean_R, mean_G, mean_B, and between std_R, std_G, std_B suggest redundancy in the information provided by these characteristics. This means that using all of these characteristics may not provide any significant additional information for a classification model.

---

## 4. Baseline Model Performance (Prediction Task)

A **Random Forest Classifier** was used on the extracted statistical features to establish a performance benchmark.

### Model Setup
* **Model:** Random Forest Classifier (Insensitive to scale/normalization)
* **Features:** Statistical features (e.g., mean_R, mean_ExG)
* **Data Split:** 80% Train, 20% Test.

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

| Class | Precision | Recall | F1-Score | Support |
| :--- | ---: | ---: | ---: | ---: |
| agricultural | 0.37 | 0.32 | 0.34 | 100 |
| airplane | 0.21 | 0.18 | 0.19 | 100 |
| baseballdiamond | 0.46 | 0.44 | 0.45 | 100 |
| beach | 0.33 | 0.37 | 0.35 | 100 |
| buildings | 0.15 | 0.15 | 0.15 | 100 |
| chaparral | 0.39 | 0.51 | 0.44 | 100 |
| denseresidential | 0.24 | 0.19 | 0.21 | 100 |
| forest | 0.45 | 0.54 | 0.49 | 100 |
| freeway | 0.13 | 0.17 | 0.15 | 100 |
| golfcourse | 0.32 | 0.37 | 0.34 | 100 |
| harbor | 0.60 | 0.64 | 0.62 | 100 |
| intersection | 0.11 | 0.13 | 0.12 | 100 |
| mediumresidential | 0.15 | 0.15 | 0.15 | 100 |
| mobilehomepark | 0.26 | 0.31 | 0.28 | 100 |
| overpass | 0.22 | 0.20 | 0.21 | 100 |
| parkinglot | 0.19 | 0.19 | 0.19 | 100 |
| river | 0.34 | 0.32 | 0.33 | 100 |
| runway | 0.19 | 0.14 | 0.16 | 100 |
| sparseresidential | 0.19 | 0.17 | 0.18 | 100 |
| storagetanks | 0.23 | 0.17 | 0.20 | 100 |
| tenniscourt | 0.15 | 0.12 | 0.13 | 100 |
| **accuracy** | | | **0.28** | **2100** |
| **macro avg** | **0.27** | **0.28** | **0.27** | **2100** |
| **weighted avg** | **0.27** | **0.28** | **0.27** | **2100** |



**Accuracy (Overall)	0.28 (28%)	2100**

### Conclusion and Next Steps

- An accuracy of 28% is significantly better than random guessing (≈4.8%), but it means the model is incorrect more than two-thirds of the time.

- Justification: The basic features (means, standard deviations, ExG) are insufficient to correctly distinguish the 21 land use classes as they only capture simple color and brightness information.

- **Need for CNN**: To significantly improve accuracy, the next step must focus on implementing the core Deep Learning pipeline to extract deep features (CNN Embeddings). These features will capture the essential textures, shapes, and spatial patterns needed for high-fidelity classification.


## CNN Embeddings

### Model's Architecture

```python

model = tf.keras.Sequential([
layers.Rescaling(1./255), #change of the numerical scale from 0-255 to 0-1
layers.Conv2D(32, 3, activation= 'relu', padding = 'same'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, activation= 'relu', padding = 'same'),
layers.MaxPooling2D(),
layers.Conv2D(128, 3, activation= 'relu', padding = 'same'),
layers.MaxPooling2D(),
layers.GlobalAveragePooling2D(), #risk of overfitting with Flatten, this reduces the dimensions and keeps the important info with the
average of each feature map
layers.Dense(64, activation = 'relu'),
layers.Dropout(0.3), #30% of the neurons will be ignored during
training to prevent overfitting
layers.Dense(num_classes, activation = 'softmax')
])
```

- **Convolution and Pooling Layers**
    Conv2D 32 filters of size 3x3 to extract local features (edges, textures).
    MaxPooling2D() reduces the spatial dimension to capture the dominant features.
    Repeat with 64 and 128 filters to extract higher-level features.
- **Dense Layers**
    The first dense layer is fully connected to combine the extracted features.
    Dropout(0.3) randomly deactivates 30% of neurons during training to improve  generalization.
    Final dense output layer for multi-class classification (21 classes).

Choice of the Adam optimizer for its effectiveness with computer vision problems, as well as the loss function: SparseCategoricalCrossentropy because it is suitable for whole labels.

### Training

[visualization/first_epochs.jpg]
[visualization/last_epochs.jpg]

We see a fairly synchronized increase in the accuracy of the train and validation, as well as a consistent decrease in the loss of train and validation.
These trends indicate that the model is learning effectively without falling into **underfitting** or **overfitting**.

The model showed a clear improvement in performance between epoch 1 and epoch 50.
Accuracy and macro recall, initially close to random levels, increased by a factor of approximately 6, achieving robust performance.
We also observe stability in the metrics from epoch 49 onwards, where they stabilized around a value of 0.8, indicating the convergence of the model.

### Results

[metrics/confusion_matrix_epoch_50.png]

The dominant diagonal indicates that the model is capable of correctly classifying the majority of samples. The following classes show particularly high prediction rates (out of 100 samples):

- Agricultural : 97
- Parkinglot : 97
- Runway : 94
- Harbor : 93
- Beach : 86

## Conclusion

 - A Random Forest model was used as a benchmark, achieving 28% accuracy, confirming the need to use more advanced techniques.
 Switching to a CNN model, with convolution layers, pooling layers, and a Dropout mechanism, achieved a macro **accuracy of 80%** and an **F1-score of 79%** after 50 epochs, without overfitting or underfitting. 
 - Although classes such as agricultural or parking lot are well predicted, some, such as intersection or storage tanks, remain difficult to distinguish due to their visual similarity. To further improve results, avenues such as transfer learning or hyperparameter optimization could be explored. The potential of CNNs for automated aerial image analysis is well demonstrated, with possible applications in urban planning, environmental monitoring, and natural resource management.


## Sources

Yi Yang and Shawn Newsam, "Bag-Of-Visual-Words and Spatial Extensions for Land-Use Classification," ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems (ACM GIS), 2010