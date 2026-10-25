from huggingface_hub import snapshot_download
import os

# ==========================================
# CONFIGURACIÓN DE DESTINO
# Pon la ruta de tu disco secundario aquí.
# El conversor NF4 espera la carpeta con este nombre junto al estudio.
# ==========================================
DIRECTORIO_DESTINO = "./Fibo2_Dev-Raw"

os.makedirs(DIRECTORIO_DESTINO, exist_ok=True)

print("====================================================")
print(" Iniciando descarga de Fibo 2.0 Dev (BRIA)")
print(f" Destino: {DIRECTORIO_DESTINO}")
print("====================================================\n")
print("Aviso: el modelo tiene 8.100 millones de parámetros y pide acceso en Hugging Face.")
print("Son unos 16 GB. Ve a tomar un café...\n")

# Hacemos la descarga directa a tu carpeta elegida
snapshot_download(
    repo_id="briaai/Fibo-2.0-dev",
    local_dir=DIRECTORIO_DESTINO,
    resume_download=True,         # Si se corta el internet, continuará por donde iba
    ignore_patterns=["*.msgpack", "*.h5", "*.bin"]  # Descarga solo los Safetensors modernos para ahorrar espacio
)

print("\n====================================================")
print(" ¡Descarga completada con éxito!")
print(" Ahora ejecuta tools/fibo2/5_conversor_Fibo2_NF4.py")
print(" para crear la carpeta Fibo2_Dev-NF4.")
print("====================================================")
