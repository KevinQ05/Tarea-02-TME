import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

data = "DatosF.csv"
df = pd.read_csv(data)
df['Demanda'] = df['Generación Nacional'] + \
    df["Importación"] - df["Exportación"]
df.drop(['Generación Nacional', 'Importación', 'Exportación'],
        axis='columns', inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True)

cutoff_date = datetime(2017, 12, 23)

df_train = df[df["Fecha"] < cutoff_date]
df_test = df[df["Fecha"] >= cutoff_date]

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(10, 6)
ax1.set_ylim(300, 1100)
ax2.set_ylim(300, 1100)
ax1.set_ylabel("MW")
ax2.set_ylabel("MW")
ax1.set_title("Demanda 1-ene-17 a 22-dic-17")
ax2.set_title("Demanda 22-dic-17 a 31-dic-17")

df_train.boxplot("Demanda", ax=ax1)
df_test.boxplot("Demanda", ax=ax2)

plt.show()
