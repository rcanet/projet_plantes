import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Lecture du tableau de données
data_df = pd.read_csv('02_data/25-07-30_data_preliminary.csv')
data_df['is_diseased'] = data_df['disease'].notna().astype(int)


# Graphic representation
## Distribution des images
order_distribution = data_df['sp'].value_counts().index
counts = data_df['sp'].value_counts().values
sns.catplot(data = data_df, y = 'sp', kind = 'count', order = order_distribution, hue = 'is_diseased')
for count, sp in zip(counts, order_distribution): # Annotation du nombre d'image pour chaque espèce
    plt.annotate(str(count), 
                 xy = ((count + 1), sp),
                 va = 'center')
plt.xlabel("Nombre d'images")
plt.ylabel("Espèces")
plt.show();


## Distribution des tailles
plt.hist(data = data_df, x = 'nb_pixel')
plt.show();
data_df.nb_pixel.value_counts(normalize = True)

## Et-ce qu'il y a des doublons ? 
data_df["id"] = data_df.sp.astype(str) + data_df.name.astype(str)
print(f"Il y a {data_df.id.duplicated().sum()} doublon(s).")

## Moyenne des canaux RGB
sns.catplot(data = data_df, )

sns.catplot(y = "nb_pixel", data = data_df, kind = 'boxen')
sns.catplot(x = 'sp', y = 'nb_pixel', data = data_df, kind = 'box')
sns.catplot(y = 'sp', data = data_df, kind = 'count')
plt.show();
