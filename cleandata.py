import pandas as pd
import numpy as np

data = pd.read_csv('data.csv', header = None)

# transform columns into discrete columns
for i in range(4):
    data[i] = data[i].apply(lambda x: int(x))

data.to_csv('cleandata.csv', index=False)