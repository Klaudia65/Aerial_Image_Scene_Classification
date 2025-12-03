import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_features = pd.read_csv('dataset/image_features.csv')

# the feature to analyze
feature_name = 'std_R'

#univariate with histograms

plt.figure(figsize=(10, 6))

#histogram
sns.histplot(
    data=df_features, 
    x=feature_name, 
    kde=True,  #add the Density Curve
    bins=30,   #number of bins
    color='skyblue'
)

plt.title(f'Feature : {feature_name} distribution (All categories)')
plt.xlabel(feature_name)
plt.ylabel('Frequency (Count)')
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig(f'visualization/histogram_{feature_name}.png') 


#bivariate analysis with boxplots

plt.figure(figsize=(18, 8))
sns.boxplot(
    x='label', 
    y=feature_name, 
    data=df_features, 
    palette='Set2'
)

plt.title(f'Distribution of {feature_name} per land use class')
plt.xlabel('Land use class (label)')
plt.ylabel(feature_name)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--')
plt.tight_layout()
plt.savefig(f'visualization/boxplot_{feature_name}.png')

"""
Séparabilité : Les classes qui ont des boîtes à moustaches (quartiles) complètement séparées (ex : "forest" vs "runway") sont facilement distinguables par cette seule feature.

Chevauchement : Les classes dont les boîtes se chevauchent fortement (ex : "dense residential" vs "medium residential") seront plus difficiles à séparer.

Pertinence : Vous devriez observer que les classes de végétation (forêt, champs, etc.) ont des valeurs de mean_ExG plus élevées que les classes de matériaux inertes (autoroute, bâtiment, etc.)."""


from scipy import stats

# Récupérer les valeurs de la feature 'mean_ExG' regroupées par classe
groups = [df_features[feature_name][df_features['label'] == cls].values for cls in df_features['label'].unique()]

# Test ANOVA à un facteur
f_statistic, p_value = stats.f_oneway(*groups)

print("\n--- Résultat de l'ANOVA ---")
print(f"Feature analysée : {feature_name}")
print(f"Statistique F : {f_statistic:.2f}")
print(f"P-valeur : {p_value:.3e}")


"""
H0​ énonce qu'il n'y a aucune différence significative entre les moyennes de la feature que vous analysez (par exemple, mean_ExG) pour les 21 classes d'utilisation du sol.
Signification pratique : Il est très improbable que les différences que vous observez dans les Boxplots
 (par exemple, la moyenne de mean_ExG d'une forêt est beaucoup plus élevée que celle d'un aéroport) soient dues au simple hasard
 rejeter H0​ est un succès : cela signifie que votre feature est utile et qu'il y a des différences réelles dans vos données.
 Si vous ne pouviez pas rejeter H0​, cela signifierait que la feature analysée n'a aucune valeur pour la classification.

--- Résultat de l'ANOVA ---
Feature analysée : std_B
Statistique F : 97.99
P-valeur : 0.000e+00

--- Résultat de l'ANOVA ---
Feature analysée : mean_ExG
Statistique F : 79.10
P-valeur : 1.113e-300

--- Résultat de l'ANOVA ---
Feature analysée : std_G
Statistique F : 76.15
P-valeur : 1.187e-289

--- Résultat de l'ANOVA ---
Feature analysée : mean_G
Statistique F : 170.43
P-valeur : 0.000e+00

--- Résultat de l'ANOVA ---
Feature analysée : mean_R
Statistique F : 172.63
P-valeur : 0.000e+00

"""