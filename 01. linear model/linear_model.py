import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 

data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

X  = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

lifesat.plot(kind = "scatter", grid = True, x = "GDP per capita (USD)", y = "Life satisfaction")
print(plt)
plt.axis([23_500, 62_500, 4, 9])
plt.show()

model = LinearRegression()
model.fit(X, y)

X_new = np.array([[37_655.2]], dtype=float)

print("Model output: ", model.predict(X_new))