# Verificación de esta versión

2026-10-21. Python 3.14, NumPy 2.5.3 y Matplotlib 3.11.2, Windows.

- Ocho pruebas automatizadas pasan: flujo completo sintético con cinco familias, BIC solo de entrenamiento, exploración más allá de un candidato peor, índices comunes de símbolos, límite de 3 V y PWL, entrada inválida, recarga sin modificar modelos y retorno sintético respecto a la señal deseada.
- Flujo completo ejecutado con los conjuntos locales `training_data` y `Validation_1`: lectura, diagnóstico, alineación, entrenamiento, comparación, figuras y seis exportaciones PWL (cinco modelos y referencia sin DPD).
- Ambos conjuntos produjeron 8766 muestras alineadas, con retardo entero de 3 muestras. Se verificaron 16 muestras por símbolo a partir de los parámetros de la fuente y del reloj CSV.
- La prueba con datos locales usó límites pequeños: MP/EMP P5 M2, GMP P3 M1 Q1, Pruned Volterra P3 M2 D1 y LUT de 8 a 16 nodos. Se compararon 8759 muestras y 547 símbolos comunes. Las figuras finales de esa ejecución fueron inspeccionadas y muestran solo instantes de símbolo.
- Se comprobó que los máximos ampliados del JSON predeterminado generan un contexto válido (16 muestras pasadas, 4 futuras). **La búsqueda ampliada completa no se ejecutó como parte de esta verificación.**
- No se ejecutó Spectre ni se midió una mejora del PA con los PWL generados. El retorno se verificó con un caso sintético de ganancia y retardo conocidos.

Los datos y resultados locales no se incluyen en Git. Los tests usan datos sintéticos reproducibles. En la máquina original, el lanzador reutiliza el entorno ya instalado del procesador; la creación de otro entorno se interrumpió por falta de espacio en disco y se retiró únicamente esa instalación incompleta.
