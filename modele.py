import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('dataset.csv')

# Supprimer la colonne 'Student_ID' si elle existe
if 'Student_ID' in df.columns:
    df = df.drop(columns=['Student_ID'])

# Colonnes catégorielles
cat_cols = ['Gender', 'Academic_Level', 'Country', 'Most_Used_Platform', 'Relationship_Status']

# Cible à transformer
if 'Affects_Academic_Performance' in df.columns:
    df['Affects_Academic_Performance'] = df['Affects_Academic_Performance'].map({'Yes': 1, 'No': 0})

# Gestion des variables catégorielles avec get_dummies
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Features et target
X = df.drop(columns=['Affects_Academic_Performance'])
y = df['Affects_Academic_Performance']

# Remplacer les valeurs manquantes
for col in X.columns:
    if X[col].dtype in [np.float64, np.int64]:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(X[col].mode()[0])

# Séparer train et test pour le contexte (ici, on entraîne sur tout mais c'est bon usage)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#clf = RandomForestClassifier(random_state=42)
#clf.fit(X_train, y_train)

# Mais ici, on entraîne sur tout le jeu
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Importance des variables
importances = clf.feature_importances_
feat_names = X.columns

# Top 10
indices_top = np.argsort(importances)[-10:][::-1]
top_feat_names = feat_names[indices_top]
top_importances = importances[indices_top]

# Graphique esthétique
sns.set(style='whitegrid', palette='muted', font_scale=1.1)
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=top_importances, y=top_feat_names, orient='h', palette='Blues_d')
plt.xlabel('Importance')
plt.ylabel('Variables')
plt.title('Top 10 des variables influençant la baisse des performances académiques')
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight')
print("Graphique généré avec succès !")