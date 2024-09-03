from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

data_path = './DatosF.csv'
df = pd.read_csv(data_path)

df['Demanda'] = df['Generación Nacional'] + \
    df["Importación"] - df["Exportación"]
df.drop(['Generación Nacional', 'Importación', 'Exportación'],
        axis='columns', inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True)

y = df['Demanda']
X = df[['Temperatura']]

reg = linear_model.LinearRegression().fit(X, y)
print(reg.coef_)

fig, ax = plt.subplots()
fig.set_facecolor("#f4f4f4")
ax.set_facecolor("#f4f4f4")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

fig.suptitle("Demanda vs. Temperatura 19 de diciembre 2017")
df[df['Fecha'] == datetime(2017, 12, 19)].plot(
    x="Temperatura", y="Demanda", kind="scatter", xlabel="Temperatura °C", ylabel="Demanda MW", ax=ax, color="#727272")

reg_x = np.arange(18, 30)
reg_y = reg.coef_[0]*reg_x

ax.plot(reg_x, reg_y, color="#004651")
plt.show()

print(f'r^2 = {reg.score(X, y)}')
