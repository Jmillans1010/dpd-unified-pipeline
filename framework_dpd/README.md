# Framework DPD independiente

Abre `framework_dpd.code-workspace`.

```powershell
.\ejecutar_dpd.ps1
.\.venv\Scripts\python.exe -m unittest discover -s tests_dpd -v
```

[Documentación completa](README_DPD.md).

Este proyecto usa su propio entorno y copias de entrada en `data/input/`. El preprocesador está separado en `../procesador_iq_pa/`.
