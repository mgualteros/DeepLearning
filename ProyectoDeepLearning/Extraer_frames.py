import cv2
import os

def extraer_frames(video_path, output_folder, frame_rate):
    """
    Extrae frames de un video y los guarda en una carpeta.

    Parámetros:
    - video_path: Ruta del archivo de video.
    - output_folder: Carpeta donde se guardarán los frames.
    - frame_rate: Extrae un frame cada 'frame_rate' segundos.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtener los FPS del video
    frame_interval = fps * frame_rate  # Calcular cada cuántos frames extraer
    
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break  # Termina cuando el video se acabe
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"framep2_{saved_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Guardando {frame_filename}")
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Frames extraídos: {saved_count}")

# Uso del código
video_path = "ProyectoDeepLearning\\Fecha4Millonarios.mp4"  # Ruta del video
output_folder = "frames_prueba"  # Carpeta de salida
extraer_frames(video_path, output_folder, frame_rate=20) 
