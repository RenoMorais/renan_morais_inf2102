#Junção das bases de dados metereológicos. Em anos diferentes e localizações diferentes.

#possible resource: https://github.com/topics/inmet

#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

from pmdarima import ARIMA
from sklearn.base import BaseEstimator, RegressorMixin

import csv
import datetime as dt
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

from geopy.distance import geodesic

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, LinearRegression, Lasso
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


class ARIMAModel(BaseEstimator, RegressorMixin):
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
    
    def fit(self, X, y):
        self.model = ARIMA(order=self.order)
        self.model.fit(y)
        return self
    
    def predict(self, X):
        n_periods = len(X)
        return self.model.predict(n_periods=n_periods)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return -mean_squared_error(y, y_pred)




# TRATAMENTO E JUNÇÃO DAS BASES
def columns_renamer(name: str) -> str:
    name = name.lower()
    if re.match(r"data", name):
        return "data"
    if re.match(r"hora", name):
        return "hora"
    if re.match(r"precipita(ç|c)(ã|a)o", name):
        return "precipitacao"
    if re.match(r"press(ã|a)o atmosf(é|e)rica ao n(í|i)vel", name):
        return "pressao_atmosferica"
    if re.match(r"press(ã|a)o atmosf(é|e)rica m(á|a)x", name):
        return "pressao_atmosferica_maxima"
    if re.match(r"press(ã|a)o atmosf(é|e)rica m(í|i)n", name):
        return "pressao_atmosferica_minima"
    if re.match(r"radia(ç|c)(ã|a)o", name):
        return "radiacao"
    if re.match(r"temperatura do ar", name):
        return "temperatura_ar"
    if re.match(r"temperatura do ponto de orvalho", name):
        return "temperatura_orvalho"
    if re.match(r"temperatura m(á|a)x", name):
        return "temperatura_maxima"
    if re.match(r"temperatura m(í|i)n", name):
        return "temperatura_minima"
    if re.match(r"temperatura orvalho m(á|a)x", name):
        return "temperatura_orvalho_maxima"
    if re.match(r"temperatura orvalho m(í|i)n", name):
        return "temperatura_orvalho_minima"
    if re.match(r"umidade rel\. m(á|a)x", name):
        return "umidade_relativa_maxima"
    if re.match(r"umidade rel\. m(í|i)n", name):
        return "umidade_relativa_minima"
    if re.match(r"umidade relativa do ar", name):
        return "umidade_relativa"
    if re.match(r"vento, dire(ç|c)(ã|a)o", name):
        return "vento_direcao"
    if re.match(r"vento, rajada", name):
        return "vento_rajada"
    if re.match(r"vento, velocidade", name):
        return "vento_velocidade"


def read_metadata(filepath: Path | zipfile.ZipExtFile) -> dict[str, str]:
    if isinstance(filepath, zipfile.ZipExtFile):
        f = io.TextIOWrapper(filepath, encoding="latin-1")
    else:
        f = open(filepath, "r", encoding="latin-1")
    reader = csv.reader(f, delimiter=";")
    _, regiao = next(reader)
    _, uf = next(reader)
    _, estacao = next(reader)
    _, codigo_wmo = next(reader)
    _, latitude = next(reader)
    try:
        latitude = float(latitude.replace(",", "."))
    except:
        latitude = np.nan
    _, longitude = next(reader)
    try:
        longitude = float(longitude.replace(",", "."))
    except:
        longitude = np.nan
    _, altitude = next(reader)
    try:
        altitude = float(altitude.replace(",", "."))
    except:
        altitude = np.nan
    _, data_fundacao = next(reader)
    if re.match("[0-9]{4}-[0-9]{2}-[0-9]{2}", data_fundacao):
        data_fundacao = dt.datetime.strptime(
            data_fundacao,
            "%Y-%m-%d",
        )
    elif re.match("[0-9]{2}/[0-9]{2}/[0-9]{2}", data_fundacao):
        data_fundacao = dt.datetime.strptime(
            data_fundacao,
            "%d/%m/%y",
        )
    f.close()
    return {
        "regiao": regiao,
        "uf": uf,
        "estacao": estacao,
        "codigo_wmo": codigo_wmo,
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "data_fundacao": data_fundacao,
    }


def convert_dates(s: pd.Series) -> pd.DataFrame:
    datas = s.str.replace("/", "-").str.split("-", expand=True)
    datas = datas.rename(columns={0: "ano", 1: "mes", 2: "dia"})
    datas = datas.apply(lambda x: x.astype(int))
    return datas


def convert_hours(s: pd.Series) -> pd.DataFrame:
    s = s.apply(
        lambda x: x if re.match(r"\d{2}\:\d{2}", x) else x[:2] + ":" + x[2:]
    )
    horas = s.str.split(":", expand=True)[[0]]
    horas = horas.rename(columns={0: "hora"})
    horas = horas.apply(lambda x: x.astype(int))
    return horas


def read_data(filepath: Path) -> pd.DataFrame:
    d = pd.read_csv(filepath, sep=";", decimal=",", na_values="-9999",
                    encoding="latin-1", skiprows=8, usecols=range(19))
    d = d.rename(columns=columns_renamer)
    datas = convert_dates(d["data"])
    horas = convert_hours(d["hora"])
    d = d.assign(
        ano=datas["ano"],
        mes=datas["mes"],
        dia=datas["dia"],
        hora=horas["hora"],
    )
    return d


def read_zipfile(filepath: Path) -> pd.DataFrame:
    data = pd.DataFrame()
    with zipfile.ZipFile(filepath) as z:
        files = [zf for zf in z.infolist() if not zf.is_dir()]
        for zf in tqdm(files):
            d = read_data(z.open(zf.filename))
            meta = read_metadata(z.open(zf.filename))
            d = d.assign(**meta)
            empty_columns = [
                "precipitacao",
                "pressao_atmosferica",
                "pressao_atmosferica_maxima",
                "pressao_atmosferica_minima",
                "radiacao",
                "temperatura_ar",
                "temperatura_orvalho",
                "temperatura_maxima",
                "temperatura_minima",
                "temperatura_orvalho_maxima",
                "temperatura_orvalho_minima",
                "umidade_relativa_maxima",
                "umidade_relativa_minima",
                "umidade_relativa",
                "vento_direcao",
                "vento_rajada",
                "vento_velocidade",
            ]
            empty_rows = d[empty_columns].isnull().all(axis=1)
            d = d.loc[~empty_rows]
            data = pd.concat((data, d), ignore_index=True)
    return data


def plot_estacoes_com_raios(df, estado=None):
    if estado:
        df = df[df['uf'] == estado]
        df = df[['estacao', 'codigo_wmo', 'latitude', 'longitude']].drop_duplicates()


    # Configuração do plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Plot das estações
    for index, row in df.iterrows():
        estacao, lat, lon = row['estacao'], row['latitude'], row['longitude']
        ax.plot(lon, lat, 'bo', markersize=7, label=estacao)
        ax.text(lon, lat, estacao, fontsize=8, color='blue', ha='right')

    # Plot dos raios de 10km
    for index, row in df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        theta = np.linspace(0, 2 * np.pi, 100)
        r = 10 / (111.32 * np.cos(np.deg2rad(lat)))  # Correção para latitude
        x = lon + r * np.cos(theta)
        y = lat + r * np.sin(theta)
        ax.plot(x, y, 'r--', transform=ccrs.PlateCarree())

    # Adiciona contornos de países e oceanos
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Configurações adicionais
    plt.title('Raios de 10km ao redor das estações' + (f' no estado {estado}' if estado else ''))
    plt.show()


def join_databases(data_sensor, data_stations):
    """
    Processa os dados dos sensores e das estações, juntando-os e encontrando a estação mais próxima para cada entrada.

    Parâmetros:
        data_sensor (DataFrame): DataFrame contendo os dados dos sensores.
        data_stations (DataFrame): DataFrame contendo os dados das estações.

    Retorna:
        DataFrame: DataFrame resultante com a estação mais próxima para cada entrada.
    """
    
    # Converter as colunas de data e hora para datetime
    data_stations['data'] = pd.to_datetime(data_stations['data'])
    data_stations['hora'] = pd.to_timedelta(data_stations['hora'], unit='h')

    # Adicionar a coluna de data_hora
    data_stations['data_hora'] = data_stations['data'] + data_stations['hora']

    # Fazer a junção dos dados dos sensores com os dados das estações
    data_merge = pd.merge(data_sensor, data_stations, on=['data_hora', 'estacao'], how='inner')

    # Encontrar a estação mais próxima
    def estacao_mais_proxima(latitude, longitude, estacoes):
        menor_distancia = float('inf')
        estacao_mais_proxima = None

        for index, estacao in estacoes.iterrows():
            # Calcular a distância entre a coordenada fornecida e a coordenada da estação
            coord_estacao = (estacao['latitude'], estacao['longitude'])
            distancia = geodesic((latitude, longitude), coord_estacao).kilometers

            # Atualizar a estação mais próxima se a distância for menor
            if distancia < menor_distancia:
                menor_distancia = distancia
                estacao_mais_proxima = estacao['estacao']

        return estacao_mais_proxima

    # Criar coluna para a estação mais próxima
    data_merge['estacao_mais_proxima'] = data_merge.apply(lambda row: estacao_mais_proxima(row['latitude'], row['longitude'], data_stations), axis=1)

    return data_merge

### função models
def testar_modelos(data):
    X = data[[
        "pressao_atmosferica", "pressao_atmosferica_maxima", "pressao_atmosferica_minima",
        "radiacao", "temperatura_ar", "temperatura_orvalho",
        "temperatura_maxima", "temperatura_minima", "temperatura_orvalho_maxima",
        "temperatura_orvalho_minima", "umidade_relativa_maxima", "umidade_relativa_minima",
        "umidade_relativa", "vento_direcao", "vento_rajada", "vento_velocidade"
    ]]
    y = data["precipitacao"]

    # Dividir os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Lista de modelos a serem testados
    modelos = {
        "Regressao Linear": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor()
    }

    resultados = {}

    for nome, modelo in modelos.items():
        # Treinar o modelo
        modelo.fit(X_train, y_train)
        # Fazer previsões
        y_pred = modelo.predict(X_test)
        # Avaliar o modelo
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        resultados[nome] = {"MSE": mse, "R2": r2}

    return resultados

import pandas as pd

def tratando_missing_values(df):
    """
    Preenche os valores ausentes em colunas especificadas de um DataFrame com a média de cada coluna.

    Parâmetros:
    df (pd.DataFrame): DataFrame onde os valores ausentes serão preenchidos.

    Retorna:
    pd.DataFrame: DataFrame com os valores ausentes preenchidos nas colunas especificadas.
    """
    columns_to_fill = [
        "pressao_atmosferica", "pressao_atmosferica_maxima", "pressao_atmosferica_minima",
        "radiacao", "temperatura_ar", "temperatura_orvalho",
        "temperatura_maxima", "temperatura_minima", "temperatura_orvalho_maxima",
        "temperatura_orvalho_minima", "umidade_relativa_maxima", "umidade_relativa_minima",
        "umidade_relativa", "vento_direcao", "vento_rajada", "vento_velocidade","precipitacao"
    ]

    for column in columns_to_fill:
        if column in df.columns:
            if df[column].isnull().any():  # Verifica se há valores ausentes na coluna
                mean_value = df[column].mean()  # Calcula a média da coluna
                df[column].fillna(mean_value, inplace=True)  # Preenche os valores ausentes com a média
    return df


def pipeline_generator(model, numeric_features=None, categorical_features=None):
    if isinstance(model, ARIMA):
        # Se o modelo for ARIMA, criar um pipeline simples para manipulação de séries temporais
        pipeline = Pipeline(steps=[
            ('model', model)  # Aqui você pode adicionar transformações de dados específicas, se necessário
        ])
    else:
        # Para outros modelos do scikit-learn
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    return pipeline




def cross_val_metrics(model, X, y, is_arima=False):
    if is_arima:
        tscv = TimeSeriesSplit(n_splits=5)
        
        mae_scores = []
        mse_scores = []
        mape_scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            try:
                # Ajustar o modelo ARIMA
                model_fit = model.fit(y_train)
                
                # Prever os valores
                y_pred = model_fit.forecast(steps=len(y_test))
                
                mae_scores.append(mean_absolute_error(y_test, y_pred))
                mse_scores.append(mean_squared_error(y_test, y_pred))
                mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))
            except Exception as e:
                print(f"Erro ao ajustar o modelo ARIMA: {e}")
                mae_scores.append(np.nan)
                mse_scores.append(np.nan)
                mape_scores.append(np.nan)
        
        mae = np.nanmean(mae_scores)
        mse = np.nanmean(mse_scores)
        rmse = np.sqrt(mse)
        mape = np.nanmean(mape_scores)
        
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        mse_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
        mape_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_percentage_error')

        mae = -mae_scores.mean()
        mse = -mse_scores.mean()
        rmse = np.sqrt(mse)
        mape = -mape_scores.mean()

    return mae, mse, rmse, mape



import pandas as pd
from sklearn.preprocessing import LabelEncoder

def train_and_evaluate_models(models, data, numeric_features, categorical_features, target, break_variable=None):
    metrics_df = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'MAPE'])

    for key, model_class in models.items():
        model = model_class()

        pipeline = pipeline_generator(model, numeric_features, categorical_features)
        encoded_break_variable = LabelEncoder().fit_transform(data[break_variable]) if break_variable else None
        
        mae, mse, rmse, mape = cross_val_metrics(pipeline, data[numeric_features + categorical_features], data[target], encoded_break_variable)
        
        # Cria um DataFrame de uma linha com os resultados atuais
        new_row = pd.DataFrame([{
            'Model': key,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }])
        
        # Concatena o novo DataFrame de uma linha com o DataFrame de resultados
        metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
    
    
    return metrics_df


def plot_mape_comparison(metrics_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='MAPE', data=metrics_df, palette='viridis')
    plt.title('Comparação do MAPE entre Modelos')
    plt.xlabel('Modelo')
    plt.ylabel('MAPE')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def choose_best_model(metrics_df, metric='MAE'):
    # Escolhe o modelo com o menor valor na métrica especificada
    best_model_row = metrics_df.loc[metrics_df[metric].idxmin()]
    best_model_name = best_model_row['Model']
    return best_model_name, best_model_row  


def plot_model_predictions(models, data, numeric_features, categorical_features, target):
    plt.figure(figsize=(14, 7))
    
    for key, model_info in models.items():
        model_class = model_info['model']
        is_arima = model_info.get('is_arima', False)
        
        if is_arima:
            # Para ARIMA, os dados de entrada são univariados (apenas a série temporal)
            X = data[target].values
            y = data[target].values
            
            # Ajustar e prever usando ARIMA
            model = model_class()
            model_fit = model.fit(y)
            y_pred = model_fit.forecast(steps=len(y))
        else:
            # Para modelos não ARIMA, usamos as features e o target
            model = model_class()
            pipeline = pipeline_generator(model, numeric_features, categorical_features)
            X = data[numeric_features + categorical_features]
            y = data[target]
            
            # Ajustar e prever usando o pipeline
            pipeline.fit(X, y)
            y_pred = pipeline.predict(X)
        
        # Plotar as previsões
        plt.plot(data.index, y_pred, label=f'{key} Predictions')
    
    # Plotar os valores reais
    plt.plot(data.index, data[target], label='Actual Values', color='black', linestyle='dashed')
    
    plt.title('Model Predictions vs Actual Values')
    plt.xlabel('Index')
    plt.ylabel(target)
    plt.legend()
    plt.show()


