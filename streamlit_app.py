import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

from sklearn.model_selection import train_test_split
import SessionState


'''
# Consumo de Cerveja de Estudantes

Este conjunto de dados foi disponibilizado por estudantes e pode ser acessado neste [link](https://www.kaggle.com/dongeorge/beer-consumption-sao-paulo) do Kaggle.

Os dados são reais.
'''


'''
## Conjunto de Dados

'''

df_beer = pd.read_csv("data/Consumo_cerveja.csv", decimal=",", thousands=".")
df_beer


'''
## Análise Exploratória Rápida
'''
st.set_option('deprecation.showPyplotGlobalUse', False)

plt.figure(figsize=(19, 15))
sns.lmplot("Consumo de cerveja (litros)", "Temperatura Maxima (C)", df_beer,
           scatter_kws={"marker": "x", "color": "blue"},
           line_kws={"linewidth": 1, "color": "orange"})
st.pyplot()

plt.figure(figsize=(19, 15))
sns.lmplot("Precipitacao (mm)", "Consumo de cerveja (litros)", df_beer,
           scatter_kws={"marker": "x", "color": "blue"},
           line_kws={"linewidth": 1, "color": "orange"})
st.pyplot()

corrMatrix = df_beer.corr()

'''
## Matriz de Correlação


Suporta a decisão de escolhas de atributos para serem utilizados no treinamento. Quanto mais correlação houver, melhor será para o modelo, o inverso também ocorre, pois ao abrir mão de atributos que não contribuem para o aprendizado para o modelo, ele ficará mais preciso e mais leve.

'''

sns.heatmap(corrMatrix)
st.pyplot()

'## Atributos Utilizados no Treinamento '

numerical = ['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)',
             'Precipitacao (mm)', 'Final de Semana']

atributes_numerical = st.multiselect(
    'Atributos Numéricos Selecionados', numerical, ['Temperatura Maxima (C)', 'Precipitacao (mm)', 'Final de Semana'])


# Store the numerical features to a dataframe attrition_num
beer_final = df_beer[atributes_numerical]

target = df_beer['Consumo de cerveja (litros)']

session_state = SessionState.get(trained=False, train=None)

lr_model = linear_model.LinearRegression()

# Split data into train and test sets as well as for validation and testing
train, test, target_train, target_val = train_test_split(beer_final,
                                                         target,
                                                         train_size=0.80,
                                                         random_state=1)

if st.button('Treinar Modelo') or session_state.trained:
    'Iniciando o treinamento...'

    lr_model.fit(train, target_train.ravel())

    session_state.trained = True

    'Treinamento terminado.'
    'Verificando predições...'

    lr_model_predictions = lr_model.predict(test)

    st.success('Modelo treinado e valido com sucesso!')
    score = "Pontuação de Precisão (R2): {}".format(
        r2_score(target_val, lr_model_predictions))

    st.info(score)

    test_value = [30, 0, 0]

    '## Teste de Inferência Unitária'
    'Supondo um determinado dia com certas condições, qual seria o consumo de cerveja?'

    s_final_semana = st.selectbox("Final de Semana", ['Sim', 'Não'])

    if s_final_semana == 'Sim':
        test_value[2] = 1
    else:
        test_value[2] = 0

    n_temp = st.slider("Temperatura Maxima (C)", int(df_beer["Temperatura Maxima (C)"].min()), int(
        2*df_beer["Temperatura Maxima (C)"].max()), test_value[0])
    test_value[0] = n_temp

    n_precip = st.slider("Precipitacao (mm)", int(df_beer["Precipitacao (mm)"].min()), int(
        2*df_beer["Precipitacao (mm)"].max()), test_value[1])
    test_value[1] = n_precip

    test_value = [test_value]

    try:
        prediction = lr_model.predict(test_value)[0]
        st.success(str(int(prediction/1000)) + " mil litros de cerveja.")
        st.balloons()

    except Exception as e:
        st.warning(e)
