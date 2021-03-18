"---------------Visualização prévia do Dataset-------------------"
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

nasa = pd.read_csv("nasa.csv")
nasa.head()
column = nasa.columns

" Analizar o tipo dos dados"

types = nasa.dtypes

# Verificando registros nulos:
dados_null = nasa.isnull().sum()

'''Retirar o atributosque não serão utilizados, incluindo o target, 
pois como vou treinar um algoritmo não supervisionado
 ele não precisará  destaq variável (Hazardous)'''

nasa.drop(columns=['Hazardous',
                   'Orbiting Body',
                   'Equinox',
                   'Orbit Determination Date',
                   'Name',
                   'Neo Reference ID',
                   ], inplace=True)
"Tratamento em campos de data "
nasa['Close Approach Date'] = pd.to_datetime(nasa['Close Approach Date'])
nasa['Close Approach Date'] = nasa['Close Approach Date'].dt.strftime('%Y')

" Aplicação do método Elbow"
'''O método Elbow é uma das formas usadas para descobrir 
a quantidade ideal de clusters no conjunto de dados.'''

# Selecionando o número de clusters através do método Elbow
valores_dataset = nasa.iloc[:, :].values
inertia = []
for n in range(1, 11):
    aplicacao_do_algoritmo = (KMeans(n_clusters=n))  # aplicando o método

    aplicacao_do_algoritmo.fit(valores_dataset)  # treinando o modelo
    print(n, aplicacao_do_algoritmo.inertia_)
    print(aplicacao_do_algoritmo.inertia_)  # Soma das distâncias quadráticas intra cluster.
    print(aplicacao_do_algoritmo.labels_)  # Rótulos dos Clusters atribuídos.
    print(aplicacao_do_algoritmo.cluster_centers_)  # Valores dos Centroides.
    inertia.append(aplicacao_do_algoritmo.inertia_)  # somatório dos erros quadráticos das instâncias de cada cluster.

plt.figure(1, figsize=(20, 10))
plt.title('Metodo Elbow')
plt.plot(np.arange(1, 11), inertia, '*')  # clusters
plt.plot(np.arange(1, 11), inertia, '-', alpha=0.5)
plt.xlabel('N}Clusters'), plt.ylabel('Soma das Distâncias Q intra Clusters')
plt.show()
