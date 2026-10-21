# Puente DPD → Cadence (módulo nuevo)

Este módulo se añade al framework existente. No modifica dpd/, los coeficientes,
los optimizadores ni el preprocesador. Python genera el estímulo; no se ha ejecutado Spectre.

## Análisis de los componentes existentes

- `dpd/contracts.py`: RawIQData contiene x, y y tiempo; DataPartition guarda x,
  z=y/G, tiempo e índices. Su target es x[indices].
- `dpd/serialization.py`: JSONWriter.load_model reconstruye TrainedModel y sus
  coeficientes complejos. El JSON codifica arrays con __ndarray__ y números
  complejos como __complex__: [real, imag].
- `TrainedModel.predict(DataPartition)` usa el registro TRAINERS. MP/EMP/GMP
  evalúan basis_terms de tipo envelope; Volterra, productos directos y conjugados.
  LUT utiliza nodos, ganancias complejas e interpolación nearest o linear.
- `dpd/trainers.py` exige contexto válido: no rellena ni recorta automáticamente.
  Los retardos se extraen de los términos guardados, no solo de la etiqueta M.
- `procesador_iq_pa/` ya importa/alinea señales, exporta aligned_waveform.csv y
  conserva metadatos. Reutilizarlo para la salida nueva de Cadence en el ordenador
  moderno; no se ha portado este preprocesador a Python 3.6.

Datos usados: framework_dpd/data/input/Validation/aligned_waveform.csv, 8766
muestras complejas, Ts≈250 ps, Fs≈4 GHz. El archivo empieza a 8 ns. Se conserva
ese origen en metadata y se reinicia EXPLÍCITAMENTE el estímulo exportado en 0 s.
El último punto lineal está en 2.19125 us. No se reconstruyen las muestras anteriores
a 8 ns. No asumir que simular este segmento produce el mismo estado inicial del PA.

Las muestras de entrada son voltios (confirmado por el usuario y coherente con
los nodos I_IN/Q_IN del netlist). No se normalizan por pico/RMS ni por [-1,1].
La normalización y/G se usó para identificar el inverso. Para usarlo como DPD:
`x_dpd=F(x)`, sin dividir x por G. Se espera idealmente PA(F(x))≈G*x; esto debe
validarse con una nueva simulación, no con las métricas de postdistorsión.

## Interfaz y archivos nuevos

`cadence_bridge/core.py`: DPDModel.apply, carga IQ, estadísticas y exportador común.
`cadence_bridge/__main__.py`: CLI, metadatos y gráficos de verificación.
`cadence_bridge/feedback.py`: NMSE, error RMS, AM/AM, AM/PM y EVM con índices explícitos.
`cadence_bridge/test_bridge.py`: equivalencia con modelos originales, PWL y validaciones.

Desde framework_dpd:

```python
from cadence_bridge import DPDModel, load_iq, export_to_cadence
x, Fs, source_t0 = load_iq('data/input/Validation/aligned_waveform.csv')
model = DPDModel('data/dpd_results/run_20261001T174609_515799Z/MP.json')
x_dpd = model.apply(x)
V_SCALE = 1.0  # V/V porque los datos ya son voltios
metadata = export_to_cadence(x_dpd, Fs, 'nuevo/I_DPD.txt', 'nuevo/Q_DPD.txt',
                             voltage_scale=V_SCALE, max_abs_voltage=2.0)
```

CLI (usar siempre un directorio nuevo):

```text
python -m cadence_bridge --model data/dpd_results/run_20261001T174609_515799Z/MP.json --input data/input/Validation/aligned_waveform.csv --output data/cadence_exports/mi_exportacion_MP --voltage-scale 1 --max-abs-voltage 2
python -m unittest cadence_bridge.test_bridge -v
```

Cambiar MP.json por EMP, GMP, PrunedVolterra o LUT. `--diagnose-only` genera datos,
gráficos y estadísticas sin PWL. `--backend portable` fuerza el evaluador portable.
`--no-plots` evita matplotlib. No se modifican modelos al ejecutar.

El usuario especificó límite de 2 V de valor absoluto POR CANAL (no 2 Vpp).
Se verifica después de V_SCALE. No hay clipping, reescalado automático ni omisión
de muestras extremas. Se rechazan datos no finitos, reloj no uniforme y archivos
de salida existentes. Las estadísticas se calculan antes de exportar.
Una PWL lineal no excede los extremos de sus puntos por canal.

## Memoria, adelantos y tiempos

- `zero_pad` por defecto: ceros antes y después según los retardos reales del
  modelo, N salidas y t[n]=n/Fs. No hay desplazamiento temporal de las muestras.
  El principio/final que depende del padding queda registrado. El PA aún requiere
  su propio asentamiento; el padding digital no garantiza el estado analógico.
- `trim`: elimina pasado/futuro sin contexto. Registra first_original_index y
  last_original_index, y reinicia el tiempo a cero expresamente. La referencia
  de los gráficos se recorta con los mismos índices.
- GMP usa adelantos. Offline se dispone de ellos; streaming requeriría buffering
  y latencia igual al máximo adelanto. Se registra offline_lookahead_s y latencia.
  No se simula un procesador causal en tiempo real.

## Interpolación

Se exportó inicialmente PWL LINEAL para una primera reproducción del segmento de
voltajes muestreado: cada intervalo interpola entre x[n] y x[n+1]. Es una elección
explícita de reconstrucción, no prueba de identidad con la wsource original entre
muestras. Para evaluar DPD justamente, ejecutar también el caso SIN DPD usando el
mismo exportador, reloj, interfaz y método de interpolación.

ZOH sirve para aproximar un DAC con retención, preferiblemente acompañado por su
filtro de reconstrucción si ese es el hardware objetivo. Se solicita así:
`--interpolation zoh --transition-s 1e-12` (1 ps es un ejemplo, no una especificación
del DAC). El programa requiere 0<transition_s<Ts. Para x0,x1,x2 se emiten:

```text
0          x0
Ts         x0
Ts+eps     x1
2Ts        x1
2Ts+eps    x2
3Ts        x2
```

Los tiempos son estrictamente crecientes: retiene la muestra anterior hasta el
reloj y completa el cambio eps después. Es ZOH con rampa finita, no discontinuidad
ideal ni timestamps duplicados. La simulación debe resolver esas transiciones.
Lineal termina en (N-1)Ts; ZOH incluye el último intervalo hasta NTs. Después se
mantiene el último voltaje; no hay repetición ni retorno automático a cero.

## Potencia y espectros

Se calculan picos I/Q/envolvente, RMS complejo=sqrt(mean(I²+Q²)), mean_square_V2 y
PAPR=10log10(max(|x|²)/mean(|x|²)). No se llama vatios a V². Con resistencia
explícita `--resistance-ohm R`, se añade potencia RF ideal mean(|x|²)/(2R), válida
solo bajo vRF=I*cos-Q*sin sin ganancia, pérdidas ni desadaptación. No es una medida
de potencia entregada al PA de este testbench.
El espectro usa FFT compleja con ventana Hann y amplitud referida a 1 V, no dBm,
PSD ni ACLR. Los gráficos no alteran las muestras.

## Conexión determinada a partir del netlist disponible

Fuente inspeccionada: ../procesador_iq_pa/data/input/input, celda
TO2608_DPDTESTING/dpd_testbench. Es la copia local disponible, no una lectura de la
sesión viva de Virtuoso. Confirma que corresponde a tu revisión antes de editarla.

```text
W0 (I_IN 0 Q_IN 0) wsource ...
I1 (I_IN Q_IN RF_IN net11) IQ_modulator ... flo=240e9
I2 (RF_IN net8 net7) balun ...
I0 (net8 net7 net4 net1 vcc!) PA_COMPONENT_EMCoreCorrect_FULLEM_tightlayinv
```

CASO A: testbench con interfaz IQ. En Virtuoso sustituir/desactivar W0 (no poner
dos fuentes ideales en paralelo). Añadir dos analogLib/vsource con type=pwl y
Waveform Entry Method=File, o vpwlf si está disponible. Terminal negativo a 0;
positivos a I_IN y Q_IN respectivamente. Introducir rutas ABSOLUTAS del servidor
en el campo file. Copiar ambos TXT al servidor antes de simular. Mantener I1,
balun, PA, demodulador y filtros. No conectar IQ directamente a net8/net7.

```spectre
V_I_DPD (I_IN 0) vsource type=pwl file="/ruta/servidor/I_DPD.txt"
V_Q_DPD (Q_IN 0) vsource type=pwl file="/ruta/servidor/Q_DPD.txt"
```

No añadir los TXT como model/include files: se referencian en la fuente. No añadir
factores de escala/offset adicionales al CDF sin reflejarlos en metadata. Verificar
I_IN y Q_IN contra los TXT: la impedancia del modulador y la impedancia original
de wsource pueden afectar la reproducción y potencia, aunque coincida el voltaje.

El netlist actual usa tran stop=2.2u, outputstart=8n, strobeperiod=250p y
maxstep=100f (RF a 240 GHz). Como el archivo nuevo comienza en 0, no reutilizar
el offset de 8 ns como si fuera parte del archivo. Para capturar todo el segmento
usar outputstart=0 y stop igual al end_time_s del metadata; si se añade asentamiento,
aplicarlo de forma explícita e idéntica al caso base y reflejarlo al alinear.
No aumentar maxstep a 250 ps para la simulación RF: el periodo de 240 GHz es ~4.17 ps.
El strobe controla la salida guardada, no resuelve por sí solo la portadora.

CASO B: PA expuesto solo por RF. Hace falta un modulador IQ o fuente comportamental
que produzca vRF(t)=I_DPD(t)cos(2*pi*fc*t)-Q_DPD(t)sin(2*pi*fc*t), con ganancias y
signo de Q coherentes con el demodulador. Para este circuito se conserva I1 existente
y su balun diferencial. No se exporta RF a 240 GHz con la rejilla baseband de 4 GHz;
la mezcla se hace dentro de Cadence. No se ha supuesto el signo interno del modelo
rfLib sin leer su Verilog-A: verificarlo con una prueba de tono IQ si se reemplaza I1.

Referencias Cadence consultadas:
- https://community.cadence.com/cadence_technology_forums/f/custom-ic-design/35602/using-a-file-for-vpwlf-source
- https://community.cadence.com/cadence_technology_forums/f/custom-ic-design/11426/syntax-of-vpwlf-file

## Retorno de Cadence a Python

1. Guardar I_IN, Q_IN, I_OUT, Q_OUT y sus tiempos. Conservar también RF_IN/RF_OUT
   si se va a evaluar espectro RF. Simular una referencia SIN DPD usando PWL idéntico.
2. Exportar waveforms y procesarlas en una carpeta NUEVA con procesador_iq_pa;
   no sobrescribir el dataset de entrenamiento. Conservar retardo/alineación y hashes.
3. Comparar la salida demodulada alineada con la señal DESEADA original (la referencia
   en samples.csv), no con la señal predistorsionada. Unir por tiempo/índices y aplicar
   explícitamente el recorte de asentamiento; igual longitud no implica alineación.
4. `analyze_feedback(reference, measured, gain, symbol_indices=None)` da NMSE y
   AM/AM-AM/PM. Reutilizar G original para una comparación absoluta; cualquier nueva
   compensación de ganancia debe registrarse y reportarse por separado.
5. EVM solo con instantes de símbolo verificados y referencia adecuada. El config
   existente tiene samples_per_symbol=16 pero samples_per_symbol_verified=false:
   no se puede prometer EVM de símbolos ideal sin resolver esta verificación.
6. Reentrenar es una acción separada usando el pipeline existente y el nuevo dataset;
   este módulo no actualiza coeficientes automáticamente ni demuestra convergencia ILA.

## CentOS 7 / Python 3.6.8

El framework existente usa dataclasses, anotaciones modernas y numpy.typing: no se
importa en 3.6. El adaptador usa automáticamente el evaluador portable de los términos
SERIALIZADOS (sin rediseñar/entrenar modelos). En Python>=3.10 usa predict nativo.
Copiar cadence_bridge/, los JSON y el CSV a una carpeta común del servidor. Desde
ella se ejecuta `python3 -m cadence_bridge ... --backend portable`.

Dependencias para Python 3.6: numpy==1.19.5; matplotlib==3.3.4 si se quieren gráficos.
En equipo moderno usar numpy/matplotlib del entorno actual. Pruebas locales:
equivalencia nativo/portable sobre los cinco modelos y ambas políticas de borde,
ASCII lineal/ZOH, límites, feedback y parseo de sintaxis Python 3.6.
No se ha ejecutado en un intérprete Python 3.6 ni en CentOS/Spectre: la validación
real de ese entorno queda pendiente; no se afirma lo contrario.
