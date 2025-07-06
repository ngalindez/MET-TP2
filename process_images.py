import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

def process_image(image_path, target_size=(64, 64)):
    """
    Procesa una imagen:
    1. Lee la imagen
    2. Convierte a escala de grises
    3. Escala a 64x64
    4. Normaliza los valores a [0, 1]
    
    Args:
        image_path: Ruta a la imagen
        target_size: Tamaño objetivo (ancho, alto)
    
    Returns:
        Imagen procesada como array numpy de forma (64, 64, 1)
    """
    # Leer imagen
    img = cv2.imread(str(image_path))
    
    if img is None:
        print(f"Error: No se pudo leer la imagen {image_path}")
        return None
    
    # Convertir BGR a RGB (cv2 lee en BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convertir a escala de grises
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Escalar a 64x64
    img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalizar a [0, 1]
    #img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Agregar dimensión de canal para que sea (64, 64, 1)
    img_final = np.expand_dims(img_resized, axis=-1)
    
    return img_final

def load_and_process_dataset(dataset_path="DatasetTP"):
    """
    Carga y procesa todo el dataset de imágenes.
    
    Args:
        dataset_path: Ruta al directorio del dataset
    
    Returns:
        images: Lista de imágenes procesadas
        labels: Lista de etiquetas (0 para Healthy, 1 para Parkinson)
    """
    images = []
    labels = []
    
    dataset_dir = Path(dataset_path)
    
    # Procesar imágenes Healthy (etiqueta 0)
    healthy_dir = dataset_dir / "Healthy"
    if healthy_dir.exists():
        print(f"Procesando imágenes Healthy...")
        for img_file in healthy_dir.glob("*.png"):
            processed_img = process_image(img_file)
            if processed_img is not None:
                images.append(processed_img)
                labels.append(0)  # Healthy
    
    # Procesar imágenes Parkinson (etiqueta 1)
    parkinson_dir = dataset_dir / "Parkinson"
    if parkinson_dir.exists():
        print(f"Procesando imágenes Parkinson...")
        for img_file in parkinson_dir.glob("*.png"):
            processed_img = process_image(img_file)
            if processed_img is not None:
                images.append(processed_img)
                labels.append(1)  # Parkinson
    
    return np.array(images), np.array(labels)

def visualize_samples(images, labels, num_samples=8):
    """
    Visualiza algunas muestras del dataset procesado.
    """
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    # Seleccionar muestras aleatorias
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img = images[idx]
        label = labels[idx]
        
        # Mostrar imagen (quitar la dimensión del canal para visualización)
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f"Sample {idx}\n({'Healthy' if label == 0 else 'Parkinson'})")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_processed_dataset(images, labels, output_path="processed_dataset.npz"):
    """
    Guarda el dataset procesado en formato .npz para uso posterior.
    """
    np.savez(output_path, 
             images=images, 
             labels=labels)
    print(f"Dataset guardado en {output_path}")

def load_processed_dataset(file_path="processed_dataset.npz"):
    """
    Carga el dataset procesado desde un archivo .npz
    """
    data = np.load(file_path)
    return data['images'], data['labels']

if __name__ == "__main__":
    # Cargar el dataset
    print("Cargando y procesando el dataset...")
    images, labels = load_and_process_dataset()
    
    print(f"Dataset cargado exitosamente!")
    print(f"Total de imágenes: {len(images)}")
    print(f"Forma de las imágenes: {images.shape}")
    print(f"Distribución de etiquetas:")
    print(f"  - Healthy (0): {np.sum(labels == 0)}")
    print(f"  - Parkinson (1): {np.sum(labels == 1)}")
    
    # Visualizar algunas muestras
    print("\nVisualizando muestras del dataset procesado...")
    visualize_samples(images, labels)
    
    # Guardar el dataset procesado
    print("\nGuardando dataset procesado...")
    save_processed_dataset(images, labels)
    
    print("\n¡Procesamiento completado!") 