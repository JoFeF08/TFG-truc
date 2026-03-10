# =============================================================================
# Script de compilació amb Nuitka per al joc del Truc
# =============================================================================
# Canvis respecte la comanda original:
#
#   1. --include-package=onnxruntime
#      → Força Nuitka a incloure el paquet onnxruntime.
#        Inclou automàticament:
#          - El .pyd (onnxruntime_pybind11_state.pyd) com a extensió
#          - Les DLLs (onnxruntime.dll, onnxruntime_providers_shared.dll)
#        NOTA: Nuitka extreu les DLLs a l'arrel del directori temporal,
#        però el .pyd queda a onnxruntime/capi/. Per això, demo.py crida
#        os.add_dll_directory() ABANS d'importar onnxruntime (Python 3.8+).
#
#   2. --nofollow-import-to=onnxruntime.quantization/tools/transformers
#      → Exclou subpaquets pesats d'onnxruntime no necessaris per inferència.
#
#   3. models/best.onnx + best.onnx.data → model ONNX i pesos externs.
#
#   4. resource_path + add_dll_directory a demo.py (ja corregits).
# =============================================================================

.\.venv\Scripts\python.exe -m nuitka `
    --standalone `
    --onefile `
    --plugin-enable=tk-inter `
    --include-package=onnxruntime `
    --nofollow-import-to=onnxruntime.quantization `
    --nofollow-import-to=onnxruntime.tools `
    --nofollow-import-to=onnxruntime.transformers `
    --include-data-files=models/best.onnx=models/best.onnx `
    --include-data-files=models/best.onnx.data=models/best.onnx.data `
    --include-data-dir=vista/vista_desktop/img_iu=vista/vista_desktop/img_iu `
    --include-package-data=rlcard `
    --nofollow-import-to=torch `
    --nofollow-import-to=torchvision `
    --nofollow-import-to=torchaudio `
    --windows-icon-from-ico=icona.ico `
    --output-dir=dist_nuitka `
    --remove-output `
    --assume-yes-for-downloads `
    demo.py

