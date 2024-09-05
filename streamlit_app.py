import streamlit as st
import pandas as pd
import datetime
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, r2_score


def metrics(y_pred, y_true):
    return {
        'R2': r2_score(y_true=y_true, y_pred=y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAPE': 100*mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
    }


colors = [
    '#ffbd59',
    '#4079a1',
    '#6bb266',
    '#00a181',
]

source_choice = st.selectbox(
    "Origen de los datos",
    ("Actuales", "Sesgados"),
)

data_source = "output.csv" if source_choice == "Actuales" else "output_biased.csv"
df = pd.read_csv(data_source, parse_dates=['Fecha'])

model_names = ['M1', 'M2', 'M3']
for model in model_names:
    df[f'Error {model}'] = (df[f'Pred. {model}'] -
                            df['Demanda'])*100/df['Demanda']

day = st.slider('Día', min_value=17, max_value=31, step=1)

date = datetime.datetime(2017, 12, day)
df_specific_day = df[df["Fecha"] == date]

metrics_dict = {name: metrics(
    df_specific_day[f'Pred. {name}'], df_specific_day['Demanda']) for name in model_names}
metrics_df = pd.DataFrame.from_dict(metrics_dict).transpose()

st.write(f'Predicción de Demanda {day} diciembre 2017')
st.line_chart(df_specific_day, x='Hora', y=[
              'Demanda', 'Pred. M1', 'Pred. M2', 'Pred. M3'], y_label='MW', color=colors)

left_column, right_column = st.columns([0.65, 0.35])

with right_column:
    st.write('Métricas')
    st.write(metrics_df)

with left_column:
    st.write('Error Porcentual de Predicción')
    st.bar_chart(df_specific_day, x='Hora', y=[
                 'Error M1', 'Error M2', 'Error M3'], stack="layered", color=colors[1:4], y_label='%')

st.write("Métricas Globales")
global_metrics_dict = {name: metrics(
    df[f'Pred. {name}'], df['Demanda']) for name in model_names}
global_metrics_df = pd.DataFrame.from_dict(global_metrics_dict).transpose()
st.write(global_metrics_df)
