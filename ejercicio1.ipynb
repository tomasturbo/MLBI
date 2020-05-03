#%%

import tensorflow as tf
import pandas as pd
import sklearn as sk
import numpy as np
df = pd.read_excel('dataset1.xlsx')
df.head()
#%%

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
crr_LabelEncoder = preprocessing.LabelEncoder()
crr_LabelEncoder.fit(df.crr_Nom)
crr_Nom_numeros=crr_LabelEncoder.transform(df.crr_Nom)

# %%

crr_OneHotEncoder = OneHotEncoder(handle_unknown='error')
crr_OneHotEncoder.fit(np.array([crr_Nom_numeros]).T)
crr_Nom_binarios=crr_OneHotEncoder.transform(np.array([crr_Nom_numeros]).T).toarray()

#%%
columnas_nuevas = ["Carrera " + carrera for carrera in df.crr_Nom.unique()]
for i, columna in enumerate(columnas_nuevas):
    df[columna] = crr_Nom_binarios[:,i]
df["posicion"] = [str(df["Latitud"][i]) + "," + str(df["Longitud"][i]) for i in range(len(df))]

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
data_scaled = scaler.fit_transform(np.array([df.psu_Nem,df.psu_Leng,df.psu_Mate,df.psu_Cie,df.psu_Pond,df.distancia,df.cred_apr]).T)
#%%
PSU = ["psu_Nem","psu_Leng","psu_Mate","psu_Cie","psu_Pond","distancia","cred_apr"]
columnas_mas_nuevas = ["std " + psu for psu in PSU]
for i, columna in enumerate(columnas_mas_nuevas):
    df[columna] = data_scaled[:,i]
borrar = ["posicion","Longitud","Latitud","psu_Nem","psu_Leng","psu_Mate","psu_Cie","psu_Pond","distancia","cred_apr"]
for i in borrar:
    df=df.drop(i,axis=1)

#%%
cluster_LabelEncoder = preprocessing.LabelEncoder()
cluster_LabelEncoder.fit(df.cluster)
cluster_numeros=cluster_LabelEncoder.transform(df.cluster)
cluster_OneHotEncoder = OneHotEncoder(handle_unknown='error')
cluster_OneHotEncoder.fit(np.array([cluster_numeros]).T)
cluster_numeros_binarios=cluster_OneHotEncoder.transform(np.array([cluster_numeros]).T).toarray()
df=df.drop("cluster",axis=1)
df=df.drop("crr_Nom",axis=1)

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.values, cluster_numeros_binarios, test_size=0.20, random_state=42)
#%%
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model

input_layer = Input(shape=(df.shape[1],))
dense_layer_1 = Dense(15, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
output = Dense(cluster_numeros_binarios.shape[1], activation='softmax')(dense_layer_2)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

#%%
print(model.summary())

#%%
history = model.fit(X_train.astype('float64'), y_train, batch_size=8, epochs=50, verbose=1, validation_split=0.2)

#%%
score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])
