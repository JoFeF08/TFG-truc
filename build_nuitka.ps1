.\.venv\Scripts\python.exe -m nuitka `
    --standalone `
    --onefile `
    --plugin-enable=tk-inter `
    --plugin-enable=numpy `
    --include-package=pkg_resources `
    --nofollow-import-to=torch `
    --nofollow-import-to=torchvision `
    --nofollow-import-to=torchaudio `
    --nofollow-import-to=onnxruntime `
    --include-data-files="RL/models/best.npz=RL/models/best.npz" `
    --include-data-files="icona.ico=icona.ico" `
    --include-data-dir="joc/vista/vista_desktop/img_iu=joc/vista/vista_desktop/img_iu" `
    --include-package-data=rlcard `
    --windows-icon-from-ico="icona.ico" `
    --output-filename="ManillIA.exe" `
    --output-dir="dist_nuitka" `
    --remove-output `
    --assume-yes-for-downloads `
    demo.py
