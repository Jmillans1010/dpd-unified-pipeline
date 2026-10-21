# Framework de identificación inversa DPD

Implementa MP, EMP, GMP, Pruned Volterra y LUT memoryless con las responsabilidades del diagrama: `TrainingPipeline`, `CSVReader`, `DataProcessor`, `Trainer`, `SearchPolicy`, `HyperparameterOptimizer`, `HardwareComplexityEstimator` y `JSONWriter`. Los contratos de datos y resultados son dataclasses en `dpd/contracts.py`.

## Ejecutar en este equipo

Abre `framework_dpd.code-workspace`. Desde `framework_dpd`:

```powershell
.\ejecutar_dpd.ps1
```

Equivalente:

```powershell
.\.venv\Scripts\python.exe run_dpd.py --config dpd_config.json
```

El lanzador también funciona desde otra carpeta. Las rutas de entrada/salida se resuelven respecto del archivo de configuración. Usa su propio entorno `.venv` y depende únicamente de NumPy. Para otro equipo: Python >= 3.10, ejecutar `python -m venv .venv` y `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.

Este workspace es independiente de `procesador_iq_pa`. El preprocesador conserva allí `run.py`, `iq_processor/`, sus pruebas y su entorno. DPD tiene código, pruebas, dependencias y resultados propios. No importa código del preprocesador ni modifica sus CSV.

La interfaz entre ambas etapas es `aligned_waveform.csv` más `metadata.json`. Se incluyen copias de los datos alineados en `data/input/` para probar DPD sin ejecutar ni tener disponible la primera etapa. `data/input/snapshot_manifest.json` registra procedencia y hashes. Las copias no se actualizan automáticamente: para un nuevo experimento, reemplazarlas explícitamente o configurar otras rutas. Esta interfaz permite unir ambas etapas en un pipeline futuro.

## Particiones acordadas

- `data/input/training/aligned_waveform.csv`: entrenamiento completo, salvo guardas/contexto.
- Primera mitad cronológica de `data/input/Validation/aligned_waveform.csv`: validación diagnóstica en modo BIC; selección de hiperparámetros solo en el modo histórico.
- Segunda mitad cronológica de ese mismo CSV: test reservado.
- La carpeta `data/output/test` queda excluida y no se borra físicamente. Su CSV era idéntico al de training.

`validation_fraction=0.5` fija la división. Se deja una separación explícita de `guard_samples` en el punto de corte y se recorta además esa cantidad en cada borde de cada partición. `null` calcula la guarda automáticamente a partir del mayor contexto permitido. Con los límites actuales es de 11 muestras; se usa soporte común de 11 muestras pasadas y 3 futuras para todos los candidatos. Se descartan los extremos sin contexto. No se rellena con ceros, no hay desplazamiento circular ni mezcla de estados entre particiones.

Los registros son simulaciones separadas y sus tiempos se reinician: se conserva el orden dentro de cada registro, sin afirmar una cronología global entre ellos. El uso de PN9 puede producir correlación entre registros; dividirlos no demuestra independencia estadística.

El `CSVReader` lee la forma de onda completa, no `selected_symbols.csv`. Se preserva la alineación que ya contiene el archivo. `DataProcessor.align_signals()` ofrece alineación entera por correlación para integrar registros nuevos; no se vuelve a estimar retardo sobre estas entradas ya alineadas. También ofrece `split_chronologically()` para un registro único.

La ganancia se calcula **solo con las muestras retenidas de training**:

```text
G = vdot(x_train, y_train) / vdot(x_train, x_train)
z = y / G
modelo: z -> x_hat; target: x
```

Se verifica orden/uniformidad temporal, valores finitos, muestreo compatible y metadatos de modulación/portadora. Los metadatos originales completos y hashes se conservan en `manifest.json`; no se interpretan como instrucciones ni se ejecutan sus rutas. La compatibilidad de esos campos no certifica que dos simulaciones tengan exactamente el mismo PA.

Para tres carpetas independientes en otro experimento, seleccionar `split_mode: "provided_partitions"` y añadir `inputs.test`. El programa rechaza particiones completamente duplicadas. Esta comprobación no detecta todas las formas de solapamiento parcial. La opción explícita `duplicate_policy: "split_training"` permite reservar el último 25 % de training si test lo duplica; **no es la configuración usada en esta entrega**.

## Matemática y convenciones

`M` es el máximo retardo, de modo que los índices abarcan `0..M`. `P` incluye solo órdenes impares. Cada modelo conserva la lista exacta y el orden de sus bases junto a sus coeficientes.

| Modelo | Base |
|---|---|
| MP | `z[n-m] * abs(z[n-m])**(p-1)` |
| EMP | `z[n] * abs(z[n-m])**(p-1)` |
| GMP aligned | Bases MP |
| GMP lagging | `z[n-m] * abs(z[n-m-q])**(p-1)` |
| GMP leading | `z[n-m] * abs(z[n-m+q])**(p-1)` |
| Volterra | Producto de `(p+1)/2` factores `z` y `(p-1)/2` factores conjugados |
| LUT | `z[n] * ganancia_compleja(abs(z[n]))` |

EMP conserva **un solo término lineal** `z[n]`: repetirlo por cada retardo produciría columnas idénticas. Por eso su cuenta real es `1 + (M+1)*(P-1)/2`. En GMP las ramas cruzadas comienzan en `P=3, M=1, Q=1`, con `m=0..M` y `q=1..Q`. Leading puede necesitar muestras futuras: el contexto se respeta y una implementación causal exigiría la latencia documentada en hardware. La organización aligned/lagging/leading sigue el modelo de [Morgan et al., 2006](https://doi.org/10.1109/TSP.2006.879264).

Volterra enumera combinaciones con repetición dentro de cada grupo, evitando permutaciones de productos idénticos. `MaxDelaySpread` retiene términos con `max(retardos)-min(retardos) <= D`. La generación y `PruningRule` están separadas; las bases retenidas se guardan para reproducir incluso una poda personalizada al recargar el modelo.

Los polinomios se ajustan con `numpy.linalg.lstsq` después de equilibrar las normas de columnas. Se exportan los coeficientes en la base original, rank, valores singulares de **Phi original**, condición original y condición de la matriz equilibrada. El rechazo usa rango y condición equilibrada, porque las distintas unidades de los monomios pueden inflar la condición original sin implicar dependencia. No se forma ni invierte `Phiᴴ Phi`. `rcond` y `condition_limit` son configurables.

La LUT utiliza centros obtenidos por amplitud uniforme, potencia uniforme o cuantiles. Nearest selecciona el centro más próximo; linear interpola entre centros y satura en los extremos. Los nodos se estiman solo con training y se reutilizan en validación/test. El ajuste nearest es LS exacto por celdas disjuntas; linear usa `lstsq` con la interpolación incluida en los regresores. Las celdas nearest vacías se interpolan desde celdas entrenadas y se registra el rango efectivo. Una LUT linear de rango deficiente se rechaza. Se maneja `z=0` sin dividir por la muestra.

## Búsqueda con BIC de Becerra et al. (2020)

La configuración activa usa `optimization.stopping_rule: "bic"`, siguiendo [On the Optimum Number of Coefficients of Sparse Digital Predistorters: A Bayesian Approach](https://doi.org/10.1109/LMWC.2020.3027878), ecuaciones (6), (9) y (13). Aquí `N` es el número de muestras complejas puntuadas de **training** y `n` el número de coeficientes complejos independientes ajustados:

```text
penalty_db_per_coefficient = 10*log10(2*N)/N
J = NMSE_train_dB + n*penalty_db_per_coefficient
aceptar candidato si J_candidate < J_current
equivalente: mejora_NMSE_train_dB > delta_n*penalty_db_per_coefficient
```

Esta es la forma BIC en unidades de NMSE de la ecuación (13), equivalente al BIC de (6) salvo escala positiva y constante compartida. No es BIC calculado con validación. La forma RSS supone residual de media cero; el registro de la ejecución advierte que la correlación temporal y periodicidad PN9 afectan su interpretación estadística. No se estima automáticamente un tamaño de muestra efectivo.

En cada paso se ajustan los vecinos y se elige el menor J; hardware deshace empates exactos. La búsqueda no entrena referencias máximas ni usa `target_fraction`, `min_improvement`, `plateau_patience` o `tie_tolerance` del modo histórico. Se atraviesan configuraciones polinomiales sin cambio de base (por ejemplo EMP lineal al crecer M) para probar ampliaciones que sí incorporan términos. No se implementa DOMP: este cambio aplica el criterio del paper a la búsqueda de estructuras tipo hill-climbing existente.

La complejidad penalizada es el rango de coeficientes ajustados, no el coste hardware. En polinomios válidos coincide con el número de términos. La aplicación a LUT es una extensión condicional a su rejilla estimada con training: penaliza el rango de las ganancias ajustadas, pero no la estimación de los nodos. No es una reproducción del experimento GMP del artículo.

Paradas: `bic_minimum_reached` indica que ninguno de los vecinos válidos mejora J; `search_limit_reached` indica que se agotaron los vecinos permitidos; `max_iterations` indica límite de iteraciones; `numerical_instability` indica que no hay vecinos estables; `numerical_frontier_incomplete` indica que los vecinos estables no mejoran pero otros no pudieron evaluarse de forma estable. Estas últimas no se presentan como una meseta BIC demostrada. La parada BIC es local y depende del conjunto de términos y los límites explorados; no prueba un óptimo global ni detecta geométricamente un codo.

El resultado exporta `bic_score_db`, coeficientes y umbral por coeficiente; la traza incluye cambios de NMSE, de coeficientes y de J, así como candidatos rechazados. `progress_fraction` y `target_fraction` son `null`: no se presenta un porcentaje de capacidad. Validación se informa sin intervenir en la selección; test se calcula una sola vez para el modelo seleccionado, sin reentrenamiento con validación.

## Modo histórico de porcentaje de referencia

Disponible solo al configurar `optimization.stopping_rule: "reference_fraction"`.

Cada arquitectura tiene baseline y referencias propios. Las referencias son configuraciones grandes **dentro de los límites configurados**, no una garantía de óptimo global. Una referencia mal condicionada se registra, pero no se utiliza para fijar el objetivo. Si ninguna referencia es válida, se conserva baseline y la parada es `numerical_instability`.

```text
error = sum(abs(x - x_hat)**2) / sum(abs(x)**2)
NMSE_dB = 10*log10(error)
progress = (error_baseline - error_candidate) / (error_baseline - error_reference)
```

No se interpola en dB. Si la referencia no mejora baseline, la parada es `performance_plateau` y se informa progreso 0 con una explicación. El progreso puede superar 1 si un candidato mejora la referencia estimada.

Cada vecino aumenta una dimensión de configuración; LUT también puede cambiar binning a igual coste. Se entrenan todos los vecinos válidos y se reutilizan resultados ya calculados. Se elige por mejora de progreso / incremento de complejidad. Entre alternativas de igual complejidad se prioriza menor error de validación; si el desempeño es equivalente dentro de `tie_tolerance * error_baseline`, se comparan multiplicadores equivalentes reales, memoria y sumadores equivalentes reales, por ese orden. Se registran las diferencias y la razón de cada decisión. Las configuraciones ya recorridas no se revisitan.

`min_improvement` mide la disminución de error dividida por error baseline; `plateau_patience` permite pasos neutros limitados. Esto es relevante cuando crecer en memoria/poda todavía no añade una base útil. Si la exploración empeora, se devuelve el mejor estado **recorrido**, no automáticamente una referencia grande. El frente Pareto se calcula sobre error de validación y complejidad estructural de todos los candidatos válidos evaluados; no certifica el frente global del espacio.

Posibles paradas: `target_reached`, `performance_plateau`, `max_iterations`, `no_valid_neighbors`, `numerical_instability`. **Solo el modelo finalmente seleccionado se evalúa sobre test, una vez por arquitectura.** No se reentrena con validation.

## Configuración entregada

| Arquitectura | Límites de referencia |
|---|---|
| MP / EMP | P <= 11, M <= 8 |
| GMP | P <= 11, M <= 8, Q <= 3 |
| Pruned Volterra | P <= 7, M <= 4, D <= 2 |
| LUT | 8 a 512 entradas, duplicación geométrica |

Los límites se ampliaron respecto de P=7/M=3 y los anteriores límites GMP/Volterra/LUT para permitir explorar más términos. Siguen siendo límites de recursos, no capacidades máximas demostradas. `max_basis_terms` comprueba las bases polinomiales máximas para preparar el contexto; no es un límite general de RAM. Volterra crece combinatoriamente.

## Estimación de hardware

Los resultados incluyen número de parámetros, multiplicadores y sumadores complejos/reales, retardos, LUTs, entradas, bits, ciclos/muestra y multiplicador de reloj estimado. **Los contadores reales excluyen los operadores complejos**, por lo que no deben sumarse sin conversión. Un multiplicador complejo se convierte aquí en 4 reales y 2 sumas reales. Las palabras de 18 bits son una hipótesis de estimación; el entrenamiento es en coma flotante y no simula cuantización.

- `DirectHardwareEstimator`: recorre las bases reales; comparte potencias de envolvente por retardo, cuenta productos Volterra por término y la acumulación. Incluye memoria de coeficientes y retardos. En LUT incluye ganancias, nodos, anchos inversos para linear, búsqueda y raíz cuadrada como operaciones auxiliares.
- `LUTAssistedHardwareEstimator`: sustituye productos de potencias superiores de envolvente por tablas parametrizadas. Es un diseño conceptual cuyo error de cuantización no se simula.
- `MultiplierSharingEstimator`: dimensiona bancos compartidos, añade registros intermedios y da una cota serial conservadora de ciclos con operadores supuestos de un ciclo. No es un planificador óptimo ni una estimación de Fmax.

Los dos últimos son alternativas conceptuales; no reutilizan fórmulas específicas de una arquitectura Guan–Zhu para otras bases. Para activarlos, cambiar `hardware.mode` a `lut_assisted` o `multiplier_sharing`. Se puede añadir respectivamente `power_lut_entries` o `sharing_factor`.

No se incluyen interfaces, control, normalización `y/G` ni síntesis FPGA real. El reloj requerido es un factor de ciclos sobre tasa de muestras, no una demostración de viabilidad temporal. Complejidad estructural y recursos hardware se mantienen separados.

## Resultados y recarga

Cada ejecución crea `data/dpd_results/run_<UTC>/`, sin sobrescribir resultados previos:

- `manifest.json`: configuración, G, versiones, particiones, guardas, metadatos y hashes.
- `MP.json`, `EMP.json`, `GMP.json`, `PrunedVolterra.json`, `LUT.json`: modelo, coeficientes/LUT, métricas y `optimization_trace` completo.
- `summary.json` y `summary.md`: comparación compacta.
- `data/dpd_results/latest_run.json`: ruta de la última ejecución terminada.

El historial guarda candidatos una sola vez, referenciados por su configuración canónica en baseline, referencias e iteraciones. Los diagnósticos de fitting están en `candidate.model.diagnostics`. Incluye candidatos rechazados y el motivo. Un baseline inválido produce resultado sin modelo/test, marcado con `numerical_instability`.

JSON es estricto (`allow_nan=False`): números complejos, arrays y valores infinitos usan objetos etiquetados. Por ejemplo, un ajuste exacto tiene NMSE `-inf`, representado como `{"__nonfinite__": "-inf"}`. No se usa pickle.

```python
from pathlib import Path
from dpd.pipeline import TrainingPipeline
from dpd.serialization import JSONWriter

pipeline = TrainingPipeline("dpd_config.json")
data = pipeline.prepare_data()
latest = JSONWriter.read("data/dpd_results/latest_run.json")
model = JSONWriter.load_model(Path(latest["directory"]) / "MP.json")
x_hat = model.predict(data.test)
# x_hat corresponde a data.test.indices. G se conserva en manifest.json.
```

Estas métricas son de **identificación inversa/postdistorsión**. No son EVM ni ACPR/ACLR del sistema linealizado: no se ha aplicado físicamente DPD al PA.

## Extensión y pruebas

Una sexta arquitectura implementa `Trainer` y `SearchPolicy`, y se registra en `TRAINERS` y `POLICIES`. Puede aportar un estimador hardware propio cuando su base no encaje en las descripciones existentes. No requiere cambios en `HyperparameterOptimizer`. Para poda personalizada, compartir la misma `PruningRule` entre el entrenador Volterra y su política.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests_dpd -v
```

Las pruebas cubren regresores, poda, índices e interpolación LUT, LS, NMSE, progreso lineal, vecinos, costes, condiciones de parada, rechazo numérico, ganancia solo de training, guardas, no utilización de test en la selección, hardware y recarga de JSON. La prueba de integración usa los CSV alineados y verifica los cinco modelos recargados.

Las pruebas históricas permanecen en `../procesador_iq_pa/tests/`, fuera de este workspace. En la verificación de esta entrega, 15 de sus 16 pruebas pasaron; `test_real_data_export_reconstructs_evm_and_preserves_sources` falla con los CSV actuales de `data/input` porque `SignalValidator` rechaza su rejilla temporal. No se ha cambiado ese preprocesador ni relajado sus tolerancias para ocultar el fallo.
