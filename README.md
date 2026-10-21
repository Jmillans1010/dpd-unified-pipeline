# Pipeline unificado de predistorsión digital

Lee dos conjuntos independientes de cuatro CSV I/Q y un `input` Spectre, diagnostica y alinea cada conjunto, entrena MP/EMP/GMP/Pruned Volterra/LUT, selecciona por validación y exporta PWL para Cadence. Las constelaciones finales contienen **solo instantes de símbolo**, con ejes I/Q en voltios.

Python >=3.10. Instalación en Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\ejecutar.ps1
```

En Linux/macOS usar `.venv/bin/python run_pipeline.py`. Las rutas del JSON se resuelven respecto a ese JSON, no respecto al directorio desde el que se ejecuta. Abrir `dpd-unified-pipeline.code-workspace` para trabajar en VS Code. El lanzador Windows también acepta `DPD_PYTHON` apuntando a un entorno preparado. Si falta `.venv`, puede reutilizar el entorno existente de `../procesador_iq_pa/.venv`, anunciándolo al ejecutar.

## Datos oficiales y configuración

```text
dataset/
  training/
    I_in_training.csv
    Q_in_training.csv
    I_out_training.csv
    Q_out_training.csv
    input
  validation/
    I_in_validation.csv
    Q_in_validation.csv
    I_out_validation.csv
    Q_out_validation.csv
    input
```

Cada CSV tiene dos columnas: tiempo en segundos y voltaje, con cabeceras Cadence como `/I_IN X,/I_IN Y` (la X es tiempo, no otro canal). También se admite la cabecera explícita `time_s,voltage_V`. Las unidades se declaran en `pipeline.json`; no se deducen de la amplitud. No se admite muestreo irregular ni se remuestrea silenciosamente.

`input` es texto Spectre, obligatorio, no contiene las muestras. Se reutiliza `InputSCSReader`: extrae la fuente y el transitorio, conserva literales e inclusiones y **no ejecuta** netlists ni archivos referenciados. El parser soporta el subconjunto documentado por el proyecto original, no todo el lenguaje Spectre. Varios wsource/tran o expresiones esenciales no resueltas requieren desambiguación.

La fuente confirmada es QAM16, Data rate **en bits/s**, OFDM=No. Se calcula Rs=Rb/log2(M); Fs se mide en los CSV. Con Rb=1G y Ts=250ps resulta L=16 muestras por símbolo. Se valida Fs/Rs entero y la coherencia con strobeperiod; `oversample` de la fuente no se usa como L del CSV. La conversión automática rechaza OFDM.

La copia local de este proyecto usa `training_data` y `Validation_1`. **Validation_0 está excluido.** Los originales no se alteran: cada ejecución conserva un snapshot de los cinco archivos y sus hashes. El usuario declaró configuración común y semillas diferentes; los valores exactos se conservan como desconocidos salvo que se aporten en `seed_provenance` o estén en los parámetros del netlist. Semillas distintas no prueban independencia de PN9.

`dataset/` y `runs/` son locales y están excluidos de Git: no se publica el netlist del circuito, PDK, rutas de terceros ni resultados privados. Para un clon nuevo se necesitan los cinco archivos de cada conjunto. La demo siguiente funciona sin datos privados.

## Demo reproducible

```powershell
.\.venv\Scripts\python.exe make_demo.py dataset/demo
.\.venv\Scripts\python.exe run_pipeline.py --config dataset/demo/demo.json
```

Genera dos secuencias sintéticas distintas, con metadatos mínimos. **No son mediciones del PA ni una simulación de Spectre.** La carpeta de destino debe ser nueva.

## Flujo y selección

1. Validación de archivos, cabeceras, finitud, unidades declaradas y rejillas temporales.
2. Diagnóstico crudo con formas de onda, espectros de amplitud FFT, picos, RMS y PAPR.
3. Alineación por correlación compleja, opcionalmente fraccional; recorte/DC configurables y documentados. Se preservan los cuatro módulos originales y su interfaz `aligned_waveform.csv` + `metadata.json`.
4. Fase de símbolo y EVM inicial usando entrada medida con ganancia/fase ajustada; no se compara con QAM ideal. Se conservan la forma de onda completa y los símbolos en archivos separados.
5. Ganancia G calculada exclusivamente con las muestras de entrenamiento válidas. Identificación inversa `F(y/G) ≈ x`.
6. **BIC individual por familia sobre training únicamente.** El nuevo buscador recibe solo la partición de entrenamiento; nunca consulta validation. Mínimos cuadrados, bases, poda e interpolación LUT reutilizan los entrenadores originales.
7. Evaluación de los cinco modelos seleccionados sobre **toda validation utilizable**, con soporte común y guardas. No se divide validation ni se crea un test final.
8. Ganador: menor `nmse_waveform_db` de validación. Desempate por número de coeficientes y nombre. Es el conjunto usado para seleccionar, no una prueba independiente posterior.
9. Constelaciones: mismos índices de símbolo, fase compartida y NMSE de símbolos en los títulos. También se exporta NMSE de toda la onda, errores RMS relativos y hardware estimado. No se ocultan errores/extremos en instantes de símbolo; vista global y zoom con conteo fuera de vista.
10. Predistorsión `F(x)`, límite de voltaje y exportación; incluye el caso sin DPD con la misma interpolación.

La fase se obtiene del registro original de validación, antes de comparar reconstrucciones; no se ajusta una fase por modelo para favorecerlo. La selección de símbolos solo afecta figuras y métricas de símbolos: se mantienen todas las muestras válidas en el entrenamiento y la inferencia con memoria.

## Explorar estructuras mayores

Los máximos predeterminados son MP/EMP P15 M12, GMP P15 M12 Q4, Volterra P9 M6 D2 y LUT hasta 1024. M es máximo retardo (incluye 0..M); los órdenes son impares. El límite de términos es 4096.

`search.plateau_extra_depth=2` explora hasta dos niveles adicionales después de un nivel sin mejora; `beam_width=3` conserva tres rutas válidas por nivel. Se mantiene siempre el mejor BIC encontrado: no obliga a escoger el candidato más grande. `0` desactiva esa extensión. Sigue siendo búsqueda acotada/local, no óptimo global.

Hay presupuestos de 150 iteraciones, 180 candidatos por familia y una estimación de memoria de 1024 MiB (6 matrices complejas N×K). Es una estimación preventiva, no un límite duro de RAM. Se registran candidatos descartados por recursos, rango o condicionamiento. Una frontera inválida se informa como incompleta; no se confunde con un mínimo global.

Modificar los máximos de memoria cambia el contexto común y el número de muestras puntuadas. Consulte los índices y guardas del manifest antes de comparar ejecuciones diferentes.

## PWL y límite de 3 V

`export.overflow_policy=zero_pair`: después de `voltage_scale`, si abs(I)>3 V **o** abs(Q)>3 V se ponen **I=Q=0** para esa muestra. Exactamente ±3 V se conserva. No es clipping. Se conservan valores previos, valores exportados, máscara e índices afectados; los valores no finitos se rechazan. La política `reject` permite bloquear una exportación que exceda el límite.

`exports/<modelo>/I.txt` y `Q.txt`: dos columnas ASCII, segundos y voltios. Incluye `samples.csv` con señal deseada, predistorsión original, exportada e índices de símbolo. `metadata.json` guarda hashes, estadísticas, origen temporal, escala, contexto y puntos afectados. `export.models` admite `all` o `winner`.

Políticas `zero_pad` (N muestras, bordes identificados) y `trim` (solo contexto válido). GMP puede usar muestras futuras: se registra el adelanto y es generación offline. La interpolación puede ser `linear` o `zoh` con `0 < transition_s < Ts`. ZOH usa una rampa finita; no hay tiempos repetidos. El archivo reinicia el tiempo en 0; el origen previo se conserva en metadatos. No hay repetición automática del segmento.

Conectar los PWL a las entradas I/Q del modulador de Cadence, sustituyendo la fuente original; no conectarlos directamente al PA RF ni poner fuentes ideales en paralelo. El paso RF del simulador y el muestreo baseband son distintos. Consulte `framework_dpd/cadence_bridge/README.md` para detalles del adaptador original; sus ejemplos históricos de límite 2 V no sustituyen los 3 V del pipeline nuevo.

## Ejecutar solo una etapa o recargar modelos

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --stage preprocess
.\.venv\Scripts\python.exe run_pipeline.py --models-dir runs/run_FECHA --stage compare
.\.venv\Scripts\python.exe run_pipeline.py --models-dir runs/run_FECHA
.\.venv\Scripts\python.exe run_pipeline.py --no-plots
```

`--models-dir` acepta una ejecución unificada, su carpeta `models`, o una ejecución del framework original con manifest y cinco JSON. Reutiliza G original y copia los modelos sin cambiar coeficientes. Sin esa opción, `compare` incluye entrenamiento y comparación, pero no PWL. No hay evaluación de test escondida.

Cada ejecución crea `runs/run_<UTC>/` y actualiza `latest_run.json` solo al terminar. Un error deja un manifest `failed`; no modifica resultados previos. `preprocessed/` contiene snapshots, alineación, EVM, estadísticas y gráficos; `models/` contiene los modelos y trazas; `comparison/` contiene las figuras finales y métricas; `exports/` contiene los PWL. `report.md` resume resultados.

## Retorno de Cadence

Después de simular, exportar cuatro CSV (I/Q de entrada y salida) y su `input`, con nombres de validation, a una carpeta nueva:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py --feedback-dir ruta/retorno --reference-run runs/run_FECHA --feedback-model MP
```

La referencia se recupera de `exports/MP/samples.csv`: es la señal **deseada**, no la predistorsionada medida en I_IN/Q_IN. Se alinea la salida, se usa G fijo del entrenamiento y los índices de símbolo de la referencia. Produce NMSE, EVM, AM/AM, AM/PM y constelación de símbolos. El recorte de asentamiento se controla explícitamente con la configuración. Comparar con un retorno `--feedback-model sin_dpd` bajo condiciones equivalentes. No lanza Spectre ni reentrena automáticamente.

## Pruebas y alcance

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Se comprueban cinco modelos de extremo a extremo, inferencia del reloj, índices comunes de símbolos, BIC de entrenamiento, exploración que cruza un candidato peor, límites de voltaje y PWL, recarga sin reajustar coeficientes, fuentes intactas, errores de entrada y retorno sintético con ganancia conocida. Los tests originales se conservan en las carpetas de ambos proyectos; algunos requieren sus datasets históricos locales.

La validación sobre PA real simulado en Spectre sigue pendiente. No se reporta ACLR sin una medición y definición de bandas. El hardware es estimado, sin síntesis ni cuantización. El pipeline completo requiere Python moderno; solo el evaluador portable del adaptador original mantiene su diseño compatible con Python 3.6.
