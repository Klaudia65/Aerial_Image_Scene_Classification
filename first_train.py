import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


df_features = pd.read_csv('dataset/image_features.csv') 

#feature columns mean, std, ExG
feature_cols = df_features.columns.drop(['label', 'image_name'])

X = df_features[feature_cols]
y = df_features['label']       #target classes

#split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, # 20% of the data for testing
    random_state=42, 
    stratify=y     # Ensure the 21 classes are evenly distributed
)

#standardization but not necessary for Random Forest
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Model Training and Evaluation (Step 4 - Baseline Model) ---

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train) 

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n--- Random Forest Baseline Model Results ---")
print(f"Accuracy on the Test Set: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""
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

Conclusion: Performance is poor.
For a 21-class classification problem, random accuracy (if the model guessed at random) would be approximately 1/21≈4.8%.
The model performs much better than random (28%), confirming that the basic features have discriminative power, as suggested by the ANOVA.
However, an accuracy of 28% means that the model is wrong more than two-thirds of the time.
Result justification (Step 4):
Accuracy ≈28%
The basic features (means, standard deviations, ExG) are insufficient to correctly distinguish the 21 land use classes, as they only capture simple information (color/brightness).
Need for CNN: 
To significantly improve accuracy, it is essential to extract deep features (CNN embeddings) that capture textures, shapes,
and spatial patterns (e.g., the lines of a runway vs. the patterns of a forest).



"""