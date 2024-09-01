from sklearn import linear_model
import pandas as pd

data_path = './DatosF.csv'
df = pd.read_csv(data_path)

df['Demanda'] = df['Generación Nacional'] + df["Importación"] - df["Exportación"]
df.drop(['Generación Nacional', 'Importación', 'Exportación'], axis='columns', inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True)

y = df['Demanda']
X = df[['Temperatura']]

reg = linear_model.LinearRegression().fit(X, y)

print(f'r^2 = {reg.score(X, y)}')
