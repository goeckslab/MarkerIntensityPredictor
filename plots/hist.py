import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./data/Tumor/preprocessed/9_3_2_preprocessed_dataset.tsv', delimiter='\t', header=0)



sns.histplot(df, x='CK19', stat='proportion', bins=10, label='CK19')
plt.ylabel("Proportion of cells")
plt.xlabel("Expression of CK19")
plt.title("Distribution of CK19 expression in biopsy 9 3 2")
plt.legend(loc='upper right')
plt.show()

