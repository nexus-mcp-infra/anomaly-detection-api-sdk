### 1. Por qué máximo 5 endpoints (Hick's Law: $T = b \cdot \log_2(n+1)$)

El principio de Hick sugiere que el tiempo requerido para seleccionar una opción aumenta con la cantidad de opciones disponibles. Con un máximo de 5 endpoints, se facilita a los desarrolladores la elección y reducen el costo cognitivo asociado con entender y utilizar la API.

### 2. Por qué pricing per-call vs por asiento (Elasticidad precio-demanda)

La elasticidad del precio-demanda mide cómo las cantidades demandadas reaccionan a un cambio en los precios. Un pricing por operación permite que los clientes solo pagueen por lo que usan, fomentando la eficiencia económica y adaptabilidad.

### 3. Por qué esta estructura de datos específica (Complejidad algorítmica $O(n \log n)$)

Transfer Entropy requiere una complejidad $O(n \log n)$ con estimadores KDE adaptativos para mantener el rendimiento en tiempo real. Esta complejidad permite procesar ventanas deslizantes sin re-entrenamiento, optimizando la eficiencia computacional.

### 4. El invariante matemático que hace esta solución correcta

El AnomalyCausalMap se basa en la diferencia de Transfer Entropy ($TE(X->Y) - TE(Y->X)$), lo cual es un invariante matemático que identifica el nodo origen de la propagación de anomalías. Este cálculo garantiza que la solución sea correcta y generalizable.

### 5. Límites teóricos del sistema (Qué no puede hacer y por qué)

El sistema tiene límites teóricos debido a las restricciones computacionales y el costo de calcular Transfer Entropy con KDE adaptativos en tiempo real. No puede procesar grandes volúmenes de datos sin un equilibrio entre velocidad y eficiencia, lo que impone limitaciones en su escalabilidad.

**Markdown:**

1. **Hick's Law:** El tiempo requerido para seleccionar una opción aumenta con la cantidad de opciones disponibles. Con un máximo de 5 endpoints, se facilita a los desarrolladores la elección y reducen el costo cognitivo asociado con entender y utilizar la API.

2. **Elasticidad precio-demanda:** La elasticidad del precio-demanda mide cómo las cantidades demandadas reaccionan a un cambio en los precios. Un pricing por operación permite que los clientes solo paguen por lo que usan, fomentando la eficiencia económica y adaptabilidad.

3. **Complejidad algorítmica:** Transfer Entropy requiere una complejidad $O(n \log n)$ con estimadores KDE adaptativos para mantener el rendimiento en tiempo real. Esta complejidad permite procesar ventanas deslizantes sin re-entrenamiento, optimizando la eficiencia computacional.

4. **Invariante matemático:** El AnomalyCausalMap se basa en la diferencia de Transfer Entropy ($TE(X->Y) - TE(Y->X)$), lo cual es un invariante matemático que identifica el nodo origen de la propagación de anomalías. Este cálculo garantiza que la solución sea correcta y generalizable.

5. **Límites teóricos:** El sistema tiene límites teóricos debido a las restricciones computacionales y el costo de calcular Transfer Entropy con KDE adaptativos en tiempo real. No puede procesar grandes volúmenes de datos sin un equilibrio entre velocidad y eficiencia, lo que impone limitaciones en su escalabilidad.