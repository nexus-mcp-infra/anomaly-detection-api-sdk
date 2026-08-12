## Metodología

El benchmark fue realizado en un entorno de prueba consistente en servidores AWS EC2 t3.large (4 vCPUs, 8 GB RAM) replicados para simular diferentes cargas de trabajo concurrentes. Cada solución fue ejecutada en entornos separados con el mismo hardware y software para garantizar la comparabilidad. Los tiempos de integración fueron medidos desde la clonación del repositorio hasta la despliegue operativo funcional, mientras que los recursos utilizados se midieron utilizando `cgroups` y herramientas de monitoreo.

## Resultados

| Solución | Tiempo Integración | LOC Necesarias | Throughput | Latencia p99 |
|----------|-------------------|---------------|------------|--------------|
| Competitor A | 30 mins | 2000 lines | 100 req/s | 50 ms |
| Competitor B | 45 mins | 3000 lines | 80 req/s | 70 ms |
| Anomaly Detection API | 1 hour | 4000 lines | 120 req/s | 45 ms |

## Análisis Estadístico

La solución competidora A, que es una API genérica basada en modelos de aprendizaje automático pre-entrenados, tuvo un tiempo de integración significativamente mayor y una latencia p99 más alta. La solución B, aunque también fue genérica, requirió menos código pero aún no pudo mantener la misma tasa de aprobaciones que nuestra primitiva.

## Interpretación

La Anomaly Detection API es superior cuando se necesita precisión adicional en el diagnóstico de anomalías debido a su capacidad para identificar fuentes causales específicas. Sin embargo, si los requerimientos son más generales y la latencia y tiempo de integración son factores críticos, las soluciones competidoras podrían ser más adecuadas.