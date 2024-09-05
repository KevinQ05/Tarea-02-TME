import pandas as pd
from sklearn import linear_model
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score

data_path = './DatosF.csv'
df = pd.read_csv(data_path)

df['Demanda'] = df['Generación Nacional'] + \
    df["Importación"] - df["Exportación"]
df.drop(['Generación Nacional', 'Importación', 'Exportación'],
        axis='columns', inplace=True)
df["Fecha"] = pd.to_datetime(df["Fecha"], dayfirst=True)
df["Día"] = df["Fecha"].dt.dayofweek
df["Demanda Previa"] = df["Demanda"].shift(1)

# Son un montón de variables, según Valdez, 2016 (Grande UT)
dias = ['Lun', 'Mar', 'Mie', 'Jue', 'Vie', 'Sab']
horas = range(0, 23)
for idx, dia in enumerate(dias):
    df[dia] = df['Día'].apply(lambda d: 1 if d == idx else 0)
for hora in horas:
    df[f'Hora {hora}'] = df['Hora'].apply(lambda h: 1 if h == hora else 0)
# Subset para la regresión y evaluación del modelo
cutoff_date = datetime.datetime(2017, 12, 17)
training_df = pd.DataFrame(
    df[(df["Fecha"] < cutoff_date) & (df["Demanda Previa"].notna())])
testing_df = pd.DataFrame(df[df['Fecha'] >= cutoff_date])

horas_cols = [f'Hora {h}' for h in horas]
columns_train = {
    'M1': dias + horas_cols,
    'M2': dias + horas_cols + ['Demanda Previa'],
    'M3': dias + horas_cols + ["Temperatura", "Demanda Previa"]
}
columns_test = {
    'M1': dias + horas_cols,
    'M2': dias + horas_cols,
    'M3': dias + horas_cols + ['Temperatura']
}

y_train = training_df['Demanda']
y_test = testing_df["Demanda"]

X_train = {key: training_df[cols] for key, cols in columns_train.items()}
X_test = {key: testing_df[cols] for key, cols in columns_test.items()}

models = {key: linear_model.LinearRegression().fit(
    X_train[key], y_train) for key in columns_test.keys()}

predictions = {
    "M1": models["M1"].predict(X_test["M1"])
}

# Para predecir iterativamente
demanda_previa = {}

for model in models.keys():
    if model == "M1":
        continue

    demanda_previa[model] = [X_train[model]["Demanda Previa"].iloc[-1]]
    predictions[model] = []
    for t in range(testing_df.shape[0]):
        X = X_test[model].iloc[t].to_frame().transpose()
        X["Demanda Previa"] = demanda_previa[model][t]

        y = models[model].predict(X)

        predictions[model].append(y[0])
        demanda_previa[model].append(y[0])

for model, pred in predictions.items():
    testing_df[f"Pred. {model}"] = pred

r_squared = {}
rmse = {}
mape = {}
coefficients = {}
y_real = testing_df["Demanda"]

for key, model in models.items():
    r_squared[key] = r2_score(y_real, predictions[key])
    rmse[key] = root_mean_squared_error(y_real, predictions[key])
    mape[key] = mean_absolute_percentage_error(y_real, predictions[key])
    coefficients[key] = pd.DataFrame(zip(X_train[key], model.coef_))
    # print(model.intercept_)
print(coefficients)

# For plotting


def extract_date(date): return testing_df[testing_df["Fecha"] == date]


date = datetime.datetime(2017, 12, 18)
df_pred = extract_date(date)
x_values = df_pred['Hora']
indices = df_pred['Hora'].index

testing_df[['Fecha', 'Hora', 'Demanda', 'Pred. M1',
            'Pred. M2', 'Pred. M3']].to_csv('output.csv', index=False)

fig, ax = plt.subplots()
ax.set_ylabel("Demanda (MW)")
fig.suptitle(f"Predicción de demanda {date.day}-{date.month}-{date.year}")
colors = {
    'M1': '#003f5c',
    'M2': '#7a5195',
    'M3': '#ef5675',
}
for key, val in predictions.items():
    y_values = df_pred[f'Pred. {key}']
    ax.plot(x_values, y_values,
            label=f'Pred. {key}', color=colors[key], alpha=0.5)

ax.legend()
df_pred.plot(x="Hora", y="Demanda", ax=ax, color='#ffa600', linewidth=1.5)
plt.show()
