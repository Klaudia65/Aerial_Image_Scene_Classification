import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df_features = pd.read_csv('dataset/image_features.csv')

#drop non-numeric columns for correlation analysis
feature_cols = df_features.columns.drop(['label', 'image_name'])
df_numeric = df_features[feature_cols]

scaler = StandardScaler()

# fit_transform calcule la moyenne (μ) et l'écart-type (σ) sur les données d'entraînement
df_features[feature_cols] = scaler.fit_transform(df_features[feature_cols])

print("Mean of the features after standardization:")
print(df_features[feature_cols].mean())


#pearson correlation
correlation_matrix = df_numeric.corr(method='pearson')

plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,      
    fmt=".2f", 
    cmap='coolwarm',
    linewidths=.5,
    cbar_kws={'label': 'Pearson Correlation Coefficient'}
)
plt.title('Heatmap of Pearson Correlation between Basic Image Features')
plt.tight_layout()
plt.savefig('visualization/heatmap_correlation_basic_features_after_standard.png')
