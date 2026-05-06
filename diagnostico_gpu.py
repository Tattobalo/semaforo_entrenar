"""
=============================================================================
DIAGNÓSTICO GPU — PyTorch + RTX 3060
=============================================================================
Ejecuta este script ANTES de entrenar para verificar que PyTorch
detecta correctamente la GPU y que CUDA está bien instalado.

Uso:
    python diagnostico_gpu.py
=============================================================================
"""

import sys
import subprocess

print("=" * 60)
print("  DIAGNÓSTICO GPU / CUDA / PyTorch")
print("=" * 60)

# =============================================================================
# 1. Versión de Python
# =============================================================================
print(f"\n[Python] {sys.version}")

# =============================================================================
# 2. Versión de PyTorch y soporte CUDA
# =============================================================================
try:
    import torch
    print(f"\n[PyTorch]")
    print(f"  Versión        : {torch.__version__}")
    print(f"  CUDA disponible: {torch.cuda.is_available()}")
    print(f"  CUDA compilado : {torch.version.cuda}")
    print(f"  cuDNN versión  : {torch.backends.cudnn.version()}")
except ImportError:
    print("[ERROR] PyTorch no está instalado.")
    sys.exit(1)

# =============================================================================
# 3. Información de GPU si CUDA está disponible
# =============================================================================
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"\n[GPU] {n_gpus} dispositivo(s) encontrado(s):")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / 1e9
        print(f"  GPU {i}: {props.name}")
        print(f"    VRAM total    : {vram:.2f} GB")
        print(f"    CUDA Compute  : {props.major}.{props.minor}")
        print(f"    Multiproc.    : {props.multi_processor_count}")

    # Probar operación real en GPU
    print("\n[Test] Ejecutando operación de prueba en GPU...")
    try:
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = torch.mm(a, b)
        print(f"  ✓ Multiplicación de matrices en GPU: OK")
        print(f"  ✓ Tensor resultado en: {c.device}")
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ✗ Error en operación GPU: {e}")

    # VRAM libre actual
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram_reserv = torch.cuda.memory_reserved(0)  / 1e9
    vram_alloc  = torch.cuda.memory_allocated(0) / 1e9
    vram_libre  = vram_total - vram_reserv
    print(f"\n[VRAM]")
    print(f"  Total      : {vram_total:.2f} GB")
    print(f"  Libre aprox: {vram_libre:.2f} GB")

else:
    # ==========================================================================
    # CUDA NO DISPONIBLE — diagnóstico de causas comunes
    # ==========================================================================
    print("\n[⚠] torch.cuda.is_available() = FALSE")
    print("    La GPU NO será usada. Causas posibles:\n")

    # Causa 1: PyTorch CPU-only
    if "+cpu" in torch.__version__ or torch.version.cuda is None:
        print("  ✗ CAUSA DETECTADA: PyTorch instalado en versión CPU-only")
        print("    torch.__version__ =", torch.__version__)
        print("    torch.version.cuda =", torch.version.cuda)
        print()
        print("  ── SOLUCIÓN ─────────────────────────────────────────────")
        print("  Desinstala PyTorch actual y reinstala con soporte CUDA:")
        print()
        print("  # Para CUDA 12.1 (recomendado para RTX 3060 con drivers recientes):")
        print("  pip uninstall torch torchvision torchaudio -y")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print()
        print("  # Para CUDA 11.8 (si tu driver es más antiguo):")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("  ─────────────────────────────────────────────────────────")
    else:
        print(f"  PyTorch versión CUDA: {torch.version.cuda} (parece correcto)")
        print("  El problema puede ser el driver de NVIDIA o la versión de CUDA toolkit.\n")

    # Causa 2: Driver de NVIDIA
    print("\n  Verificando driver NVIDIA...")
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Extraer versión del driver de la salida
            for line in result.stdout.splitlines():
                if "Driver Version" in line or "CUDA Version" in line:
                    print(f"  {line.strip()}")
            print("\n  ✓ nvidia-smi funciona — el driver está instalado")
            print("  → El problema es incompatibilidad entre la versión CUDA")
            print("    del driver y la versión CUDA con la que se compiló PyTorch.")
            print()
            print("  ── SOLUCIÓN ─────────────────────────────────────────────")
            print("  Verifica qué versión CUDA soporta tu driver en la salida")
            print("  de nvidia-smi (campo 'CUDA Version') y luego instala")
            print("  PyTorch compilado para esa versión o inferior:")
            print()
            print("  https://pytorch.org/get-started/locally/")
            print("  ─────────────────────────────────────────────────────────")
        else:
            print("  ✗ nvidia-smi no encontrado o falló")
            print("  → El driver de NVIDIA NO está instalado correctamente")
            print()
            print("  ── SOLUCIÓN ─────────────────────────────────────────────")
            print("  1. Descargar driver desde: https://www.nvidia.com/drivers")
            print("  2. Instalar CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
            print("  3. Reiniciar el sistema")
            print("  4. Reinstalar PyTorch con soporte CUDA (ver arriba)")
            print("  ─────────────────────────────────────────────────────────")
    except FileNotFoundError:
        print("  ✗ nvidia-smi no encontrado en el PATH")
        print("  → Instalar drivers NVIDIA desde https://www.nvidia.com/drivers")
    except Exception as e:
        print(f"  ✗ Error ejecutando nvidia-smi: {e}")

    # Causa 3: Entorno virtual con PyTorch CPU
    print("\n  ── VERIFICACIÓN ADICIONAL ───────────────────────────────────")
    print(f"  Ejecutable Python: {sys.executable}")
    print("  Si estás en un entorno virtual (venv/conda), asegúrate de")
    print("  haber instalado PyTorch-CUDA DENTRO de ese entorno, no globalmente.")
    print("  ─────────────────────────────────────────────────────────────")

# =============================================================================
# 4. Versión de torchvision
# =============================================================================
try:
    import torchvision
    print(f"\n[torchvision] {torchvision.__version__}")
except ImportError:
    print("\n[torchvision] NO instalado — ejecuta:")
    print("  pip install torchvision --index-url https://download.pytorch.org/whl/cu121")

# =============================================================================
# 5. Resumen final
# =============================================================================
print("\n" + "=" * 60)
if torch.cuda.is_available():
    print("  ✅ LISTO — PyTorch detecta la GPU correctamente.")
    print("     Puedes ejecutar train_faster_rcnn.py")
else:
    print("  ❌ GPU NO detectada — sigue las instrucciones de arriba")
    print("     y vuelve a ejecutar este script para verificar.")
print("=" * 60)
