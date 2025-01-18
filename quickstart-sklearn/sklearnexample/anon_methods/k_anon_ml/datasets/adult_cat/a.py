import pandas as pd

df = pd.read_csv('./adult_cat.csv', delimiter=',')
df.to_csv('./adult_cat.csv', index=False, sep=';')
