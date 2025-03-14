import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from copy import deepcopy
import os
from PIL import Image

# Rutas correctas a los archivos en OneDrive
train_path = "/Users/linaavilamoreno/Library/CloudStorage/OneDrive-Personal/Documentos/Alzheimer/Data/train-00000-of-00001-c08a401c53fe5312.parquet"
test_path = "/Users/linaavilamoreno/Library/CloudStorage/OneDrive-Personal/Documentos/Alzheimer/Data/test-00000-of-00001-44110b9df98c5585.parquet"

# Cargar los datasets
train_df = pd.read_parquet(train_path, engine="pyarrow")
test_df = pd.read_parquet(test_path, engine="pyarrow")

# Mostrar información básica
print("Train Dataset:")
print(train_df.info())
#print(train_df.head())

print("\nTest Dataset:")
print(test_df.info())
#print(test_df.head())

#Ambos conjuntos de datos están balanceados.

#Función para obtener dimensiones

def obtener_dimensiones(directorio):
    dimensiones = []
    etiquetas = []

    for clase in os.listdir(directorio):
        clase_path = os.path.join(directorio,clase)

        if os.path.isdir(clase_path):
            for filename in os.listdir(clase_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    filepath = os.path.join(clase_path, filename)
                    try:
                        with Image.open(filepath) as img:
                            dimensiones.append(img.size)  
                            etiquetas.append(clase)  
                    except Exception as e:
                        print(f"Error al procesar {filename}: {e}")

    return dimensiones, etiquetas

dimensiones_train, etiquetas_train = obtener_dimensiones(train_path)
dimensiones_test, etiquetas_test = obtener_dimensiones(test_path)

print(f"Se procesaron {len(dimensiones_train)} imágenes en el conjunto de entrenamiento.")
print(f"Se procesaron {len(dimensiones_test)} imágenes en el conjunto de prueba.")