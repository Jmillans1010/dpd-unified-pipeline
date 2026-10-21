# Procesador I/Q para PA

Proyecto independiente que lee cuatro CSV Cadence y un netlist Spectre, alinea entrada/salida por correlación cruzada compleja y calcula EVM respecto de la entrada medida con ganancia y fase constantes ajustadas. Conserva las muestras completas para posterior modelado MP/GMP/Volterra.

## Estructura

```text
procesador_iq_pa/
├── run.py                       # Punto de entrada
├── ejecutar.ps1                 # Lanzador para Windows
├── config.json                  # Rutas relativas y parámetros
├── requirements.txt
├── iq_processor/
│   ├── models.py                # Configuración, contratos y resultados
│   ├── readers.py               # CSV, netlist y resolución de metadatos
│   ├── processing.py            # Validación, correlación, selección y EVM
│   └── outputs.py               # CSV, JSON y gráficas
├── tests/test_processor.py      # Pruebas sintéticas y con los CSV adjuntos
└── data/
    ├── input/                  # Copias de los cinco archivos proporcionados
    └── output/                 # Una carpeta nueva por ejecución
```

Los originales de la unidad de red no se modifican. Los archivos incluidos son `I_in_training.csv`, `Q_in_training.csv`, `I_out_training.csv`, `Q_out_training.csv` e `input`. El netlist no necesita extensión `.scs`.

## Ejecutar

En Windows, desde esta carpeta:

```powershell
.\ejecutar.ps1
```

El lanzador utiliza exclusivamente `.venv/Scripts/python.exe`. El entorno de este equipo se ha creado con Python 3.14 y dependencias compiladas para esa versión. No guarda una ruta absoluta de usuario en el código. La carpeta histórica `.runtime_deps/` no se utiliza: contiene extensiones de Python 3.12 que no son compatibles con Python 3.14.

En VS Code, abre `procesador_iq_pa.code-workspace`. Selecciona **Python: Select Interpreter** y elige `.venv/Scripts/python.exe` si VS Code conserva el intérprete anterior. El perfil **Procesador I/Q (.venv)** de Ejecutar y depurar especifica explícitamente ese entorno. La selección de intérprete en VS Code requiere la extensión Python habitual.

El mensaje `_multiarray_umath.cp312-win_amd64.pyd` ejecutando Python 3.14 indica una mezcla de versiones binarias, no un error del algoritmo. No añadas `.runtime_deps` a `PYTHONPATH`. Si cambias de versión de Python, crea un entorno nuevo e instala allí `requirements.txt`; no copies paquetes compilados de otro intérprete. Los entornos virtuales se recrean al trasladar el proyecto a otro equipo.

Para una instalación convencional y portátil (Python 3.10 o posterior):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

Con dependencias ya instaladas también funciona `python run.py`. En macOS/Linux, el intérprete del entorno virtual está en `.venv/bin/python`.

```text
python run.py --no-plots
python run.py --config otra_configuracion.json
```

`run.py` puede invocarse desde otra carpeta: la configuración predeterminada se busca junto al script, y las rutas de los datos se resuelven **desde la carpeta del archivo de configuración**, no desde el directorio actual de la terminal. No se necesita conexión a la unidad de red para procesar las copias.

## Flujo y métodos

1. **CSVReader** lee tiempo en segundos y valor, con cabecera y separador coma. Construye `x = I_in + j Q_in` e `y = I_out + j Q_out`. No une I/Q con tiempos diferentes.
2. **InputSCSReader** extrae valores por instancia y análisis, une líneas continuadas y respeta comentarios/cadenas. Conserva literales e inclusiones sin ejecutarlas. No interpreta Spectre completo ni evalúa expresiones arbitrarias. Si hay varias fuentes `wsource` o varios análisis `tran`, solicita desambiguación mediante error explícito.
3. **MetadataResolver** combina netlist, JSON anterior asociado a los mismos CSV y configuración manual; detecta conflictos. `previous_metadata` solo acepta coincidencia de SHA-256 de los cuatro CSV. No reutiliza EVM ni retardos anteriores.
4. **SignalValidator** verifica valores finitos, orden temporal, uniformidad y compatibilidad de las rejillas. Registra el muestreo observado y las tolerancias. No remuestrea silenciosamente.
5. **SignalPreprocessor** aplica solo el recorte inicial y la eliminación de DC habilitados. Por defecto ambos están desactivados.
6. **SampleAligner** calcula correlación lineal por FFT con ceros suficientes, equivalente a `R[l] = sum(conj(x[n])*y[n+l])`, y maximiza `abs(R)` dentro del intervalo configurado. Un `D` positivo significa que se empareja `x[n]` con `y[n+D]`. No hay desplazamiento circular.
7. **FractionalDelayRefiner**, opcional, ajusta una parábola a la magnitud del pico y aplica interpolación sinc con ventana Hann. Se descartan bordes sin soporte. Está desactivado en la configuración entregada. Es una aproximación local, no una búsqueda de retardo mediante EVM.
8. **SymbolTimingSelector** evalúa `p=0,...,L-1` sobre el mismo número de símbolos completos. Para cada fase usa `X[k]=x_aligned[p+kL]` e `Y[k]=y_aligned[p+kL]`. Selecciona el mínimo EVM; con empate dentro de la tolerancia elige la fase menor. `symbol_phase` permite fijar una fase manual sin perder los diagnósticos de las demás.
9. **EVMCalculator** estima `alpha = sum(conj(X)*Y)/sum(abs(X)**2)` y calcula `100*sqrt(sum(abs(Y-alpha*X)**2)/sum(abs(alpha*X)**2))`. La referencia es la entrada medida ajustada, no una constelación QAM ideal. Las referencias de energía nula o despreciable se rechazan.
10. **StatisticsCalculator**, **MetadataProcessor**, **CSVWriter**, **JSONWriter** y **Grapher** generan estadísticas, trazabilidad y archivos. La forma de onda conserva ganancia/fase; la compensación no se aplica al CSV completo.

El programa usa arrays NumPy tipados en sus objetos de datos en lugar de DataFrames: mantiene los contratos del diseño sin requerir pandas. Los datos tabulares se exportan como CSV.

## Parámetros que conviene conocer

| Opción | Valor entregado | Interpretación |
|---|---:|---|
| `samples_per_symbol` | 16 | Hipótesis explícita para este experimento. Puede ser `null` para resolver desde otras fuentes. |
| `samples_per_symbol_verified` | false | Se advierte que el significado físico de L sigue pendiente de verificación. |
| `source_data_rate_is_bit_rate` | false | No interpreta `MSource_DataRate` como bits/s automáticamente. |
| `max_delay_samples` | 128 | Búsqueda inicial configurable de −128 a +128 muestras. |
| `minimum_overlap_fraction` | 0.8 | Exige al menos 80 % de la longitud del registro más corto. |
| `fractional_refinement` | false | Alineación entera en la ejecución predeterminada. |
| `additional_initial_crop_s` | 0 | No elimina otros 8 ns: los CSV ya empiezan en 8 ns. |
| `remove_dc` | false | No altera medias automáticamente. |
| `time_rtol` | 1e-5 | Tolerancia relativa al intervalo de muestreo, no al tiempo absoluto. |
| `time_atol_s` | 1e-18 | Tolerancia absoluta adicional en segundos. |
| `minimum_symbols` | 16 | Mínimo para ejecutar la búsqueda de fase. |
| `symbol_phase` | null | Búsqueda automática; un entero permite fase manual. |

En estos CSV el intervalo mediano es 250 ps. Hay pequeñas variaciones de hasta aproximadamente 1.81 fs entre pasos, comunes a los cuatro canales; se admite una tolerancia de unos 2.501 fs y se conservan los tiempos originales. No se elimina ninguna fila por esa variación. El retardo temporal exportado es la mediana de las diferencias de tiempos de las parejas alineadas; también se conserva el retardo nominal obtenido del muestreo.

`oversample=4` se conserva como parámetro de la fuente y **no** se usa como muestras por símbolo del CSV. Las 16 muestras por símbolo son coherentes con 1 Gbit/s, 16-QAM y 250 ps, si se confirma esa interpretación de DataRate. Una `symbol_rate_hz` manual puede aportarse dentro de `manual_metadata.signal_config`; se rechaza si contradice L. Para guardar una relación declarada como verificada se requiere modificar explícitamente la configuración.

## Archivos de salida

Cada ejecución crea `data/output/run_<fecha UTC>/`. `latest_run.json` apunta a la última exportación terminada. Las ejecuciones anteriores se conservan.

| Archivo | Contenido |
|---|---|
| `aligned_waveform.csv` | Todas las parejas alineadas, índices de origen, tiempos e I/Q. |
| `selected_symbols.csv` | Símbolos, referencia `alpha*X` y error complejo. |
| `correlation_diagnostics.csv` | Correlación, diagnóstico normalizado y solapamiento por retardo. |
| `symbol_phase_candidates.csv` | EVM, alfa y estado de cada fase. |
| `metadata.json` | Exactamente los cuatro bloques acordados. |
| `summary.txt` | Resumen legible, resultado y advertencias. |
| `01_cross_correlation.png` | Búsqueda del retardo. |
| `02_aligned_waveforms.png` | Primeras 512 muestras alineadas, sin normalizar amplitudes. |
| `03_evm_vs_phase.png` | EVM de cada fase. |
| `04_constellations.png` | Entrada medida y comparación salida/referencia lineal. |
| `05_symbol_error.png` | Magnitud del error por símbolo. |

JSON contiene `simulation_config`, `signal_config`, `processing` y `statistics`. Las rutas de procedencia son relativas a la configuración. También guarda SHA-256, opciones efectivas, versiones, orígenes de campos, candidatos y estado de cada etapa. Las rutas internas que aparezcan en literales del netlist son documentación de la simulación; el procesador no las abre.

Sin netlist, la alineación puede continuar y L puede venir de configuración manual. Si falta L, se exportan la forma de onda y la correlación, y se marca el EVM como omitido. Los errores de datos esenciales devuelven código de salida 1. Una exportación interrumpida lleva `FAILED.txt` y no reemplaza el puntero `latest_run.json`.

## Interpretación de los resultados

La fase y alfa se optimizan y evalúan sobre el mismo registro: este EVM es un resultado ajustado, **no una validación independiente**. No se usa el antiguo criterio de elegir primero la fase por cercanía a QAM ideal, ni la partición 70/30 del script anterior, porque el procedimiento acordado define entrada medida y búsqueda por error entrada/salida. La distorsión dependiente de amplitud y la memoria permanecen en el residuo.

El resultado caracteriza todo el trayecto entre los nodos I/Q exportados. Si incluye filtros o moduladores, no debe atribuirse exclusivamente al PA sin una caracterización adicional.

La primera ejecución verificada sobre las copias entregadas, sin refinamiento, obtuvo:

- 8.769 muestras por canal; 8.766 parejas alineadas.
- Retardo: 3 muestras, aproximadamente 0.75 ns.
- Fase elegida: 6 respecto del inicio alineado; 547 símbolos.
- EVM RMS: aproximadamente 22.29665 %, usando L=16 sin verificar.

Estos valores son de la configuración entregada. Cambiar recortes, fase, referencia, DC o refinamiento puede cambiar las métricas.

## Pruebas

En un entorno virtual con los requisitos instalados:

```text
python -m unittest discover -s tests -v
```

Sin activar el entorno, ejecuta `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`. No añadas la carpeta antigua `.runtime_deps` a `PYTHONPATH`.

Las pruebas comprueban correlación contra la suma directa, retardos positivos/negativos, rotación, orígenes temporales distintos, un EVM analítico conocido, energía nula, persistencia de no linealidad, tiempos inválidos, fase conocida, refinamiento fraccional, parsing del netlist, configuración incoherente, metadatos previos incompatibles y reconstrucción del EVM desde los CSV exportados.
