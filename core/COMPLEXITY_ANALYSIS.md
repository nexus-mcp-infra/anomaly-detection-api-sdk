# Anomaly Detection API - Complejidad Computacional

## Endpoints y Métodos Público

### Endpoint `detect_anomalies`

- **Complejidad Temporal**: O(n log n)
  - La estimación de la densidad de kernel adaptativo (KDE) para calcular Transfer Entropy tiene una complejidad O(n log n), donde n es el tamaño del conjunto de datos temporal.
  
- **Complejidad Espacial**: O(n)
  - Se requiere espacio para almacenar los valores del conjunto de datos y sus transformaciones temporales.
  
- **Caso Mejor**: O(1) (Para conjuntos de datos pequeños o ventanas deslizantes muy cortas donde la estimación KDE es rápida)
- **Caso Promedio**: O(log n)
- **Caso Peor**: O(n)
  
- **Cuello de Botella**: Estimación de la densidad de kernel adaptativo (KDE)

### Endpoint `get_anomaly_causal_map`

- **Complejidad Temporal**: O(m^2 * log m)
  - Se requiere calcular Transfer Entropy entre cada par de variables en el conjunto, lo que lleva a una complejidad cuadrática.
  
- **Complejidad Espacial**: O(m^2)
  - Almacena todas las aristas del grafo causal.

- **Caso Mejor**: O(1) (Para conjuntos de variables pequeños donde la cantidad de pares es pequeña)
- **Caso Promedio**: O(log m)
- **Caso Peor**: O(m^2)
  
- **Cuello de Botella**: Cálculo de Transfer Entropy entre cada par de variables

## Punto de Saturación Estimado

El punto de saturación estimado para este servicio depende del tamaño del conjunto de datos y la complejidad de las transformaciones temporales. En una arquitectura con un procesador moderno (4 núcleos, 8 hilos), se podrían manejar alrededor de **50-100 peticiones por segundo** para conjuntos de datos pequeños a medianas.

## Estrategia de Optimización para Escalar Más Allá

### Paralelización
- Utilizar múltiples núcleos y hilos para paralelizar el cálculo de Transfer Entropy en cada ventana temporal.

### Caché In-Process
- Implementar un caché in-proceso para aliviar la carga durante picos de tráfico. Los resultados recientes se almacenan en memoria y se reutilizan si se solicitan nuevamente dentro de una cierta ventana de tiempo.

### Reducción del Tamaño de Ventana
- Permite a los clientes configurar el tamaño de las ventanas temporales más pequeñas para reducir la complejidad computacional.

### Optimización de KDE
- Usar estimadores KDE de alta eficiencia que minimicen la complejidad al trabajar con grandes conjuntos de datos.

### Caching de Resultados
- Almacenar resultados de Transfer Entropy previamente calculados y reutilizarlos en lugar de recalcularlos si se solicitan nuevamente para el mismo conjunto de datos.

Estas estrategias ayudan a manejar la complejidad computacional y permiten escalar el servicio más allá del punto de saturación estimado.