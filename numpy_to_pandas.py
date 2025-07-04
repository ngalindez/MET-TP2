import numpy as np
import pandas as pd
from process_images import load_processed_dataset

def numpy_to_pandas_dataset(npz_file="processed_dataset.npz"):
    """
    Convierte el dataset procesado (.npz) a DataFrame de Pandas.
    
    Args:
        npz_file: Ruta al archivo .npz
        
    Returns:
        DataFrame con imágenes aplanadas y etiquetas
    """
    # Cargar dataset
    images, labels = load_processed_dataset(npz_file)
    
    # Aplanar imágenes: (N, 128, 128, 1) -> (N, 16384)
    images_flat = images.reshape(images.shape[0], -1)
    
    # Crear DataFrame
    df = pd.DataFrame(images_flat)
    
    # Agregar etiquetas
    df['label'] = labels
    
    # Renombrar columnas de píxeles
    pixel_columns = [f'pixel_{i:05d}' for i in range(images_flat.shape[1])]
    df.columns = pixel_columns + ['label']
    
    return df

def pandas_to_numpy_images(df, image_shape=(128, 128, 1)):
    """
    Convierte DataFrame de vuelta a formato de imágenes NumPy.
    
    Args:
        df: DataFrame con imágenes aplanadas
        image_shape: Forma original de las imágenes
        
    Returns:
        Array de imágenes en formato (N, height, width, channels)
    """
    # Obtener columnas de píxeles (excluir 'label')
    pixel_columns = [col for col in df.columns if col.startswith('pixel_')]
    
    # Extraer datos de píxeles
    images_flat = df[pixel_columns].values
    
    # Reshape a formato de imagen
    images = images_flat.reshape(-1, *image_shape)
    
    return images