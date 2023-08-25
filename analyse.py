import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Processing...")

train = pd.read_csv("DataSet/train.csv")
card = pd.read_csv("DataSet/card.csv")
user = pd.read_csv("DataSet/user.csv")
print("Read file complite.")

# Delete $ simbol from amount column
print("Delete $ simbol from amount column...")
train['amount'] = train['amount'].str.replace('$', '').astype(float)
print("Complite.")

'''
import pandas_profiling

pr_t = train.profile_report()
pr_c = card.profile_report()
pr_u = user.profile_report()

pr_t.to_file("DataSet/Analyse/train.html")
pr_c.to_file("DataSet/Analyse/card.html")
pr_u.to_file("DataSet/Analyse/user.html")
'''

#sampling
print("Sampling...")
sample_data = train.sample(frac=0.05, random_state=1225)
sample_data = sample_data.drop(columns=["index"])
print("Complite.")

print("Drawing Graph...")
sns.set(style="ticks", color_codes=True)
sns.pairplot(sample_data, hue="is_fraud?", plot_kws={'s': 10})
plt.savefig("DataSet/Analyse/Scatter_plot.png")
print("Saved figuare.")
