import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

#data = sc.read_h5ad('./data/mesmer/.h5ad')
#print(data.obs)
#print(data.var)

#df = pd.DataFrame(data.X, columns=data.var.index)
df = pd.read_csv('./data/mesmer/preprocessed/9_14_1.tsv', delimiter='\t', header=0)
print(df)

print(df['CK19'].describe())

# df = pd.read_csv('./data/Tumor/preprocessed/9_3_2_preprocessed_dataset.tsv', delimiter='\t', header=0)


sns.histplot(df, x='CK19', stat='proportion', bins=10, label='CK19')
plt.ylabel("Proportion of cells")
plt.xlabel("Expression of CK19")
plt.title("Distribution of CK19 expression in biopsy 9 14 1")
plt.legend(loc='upper right')
plt.show()
