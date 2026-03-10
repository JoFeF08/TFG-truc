.\.venv\Scripts\python.exe -m nuitka `
    --standalone `
    --onefile `
    --plugin-enable=tk-inter `
    --include-data-dir=models=models `
    --include-data-dir=vista/vista_desktop/img_iu=vista/vista_desktop/img_iu `
    --include-package-data=rlcard `
    --windows-icon-from-ico=icona.ico `
    --output-dir=dist_nuitka `
    --remove-output `
    --assume-yes-for-downloads `
    demo.py

