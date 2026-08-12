# Importa las bibliotecas necesarias
import math

def demanda(P):
    """Función de demanda lineal inversa: Q = a / P"""
    # Parámetros específicos para el mercado objetivo
    a = 10000  # Volumen esperado en operaciones/mes por cliente
    return max(0, a - 10 * P)

def elasticidad(P):
    """Elasticidad de la demanda: ε = (dQ/dP) × (P/Q)"""
    Q = demanda(P)
    if Q == 0:
        return math.inf
    dQ_dP = -10
    return dQ_dP * P / Q

def ingreso(P):
    """Ingreso total: I = P × Q"""
    return P * demanda(P)

def precio_optimo():
    """Precio óptimo que maximiza el ingreso"""
    # Encuentra el máximo de la función ingreso
    P_prev = 0
    P_next = 1
    while True:
        if ingreso(P_prev) < ingreso(P_next):
            P_prev = P_next
            P_next *= 2
        else:
            break
    return (P_prev + P_next) / 2

# Escenarios de adopción simulados
scenarios = [
    {'segmento': 'early_adopter', 'precio': 0.01, 'volumen_mensual': 5000},
    {'segmento': 'mid_market', 'precio': 0.02, 'volumen_mensual': 50000},
    {'segmento': 'enterprise', 'precio': 0.04, 'volumen_mensual': 500000},
]

def punto_equilibrio_freemium_paid(scenarios):
    """Punto de equilibrio freemium->paid"""
    for scenario in scenarios:
        P = scenario['precio']
        Q = scenario['volumen_mensual']
        if ingreso(P) > Q * 0.1:  # Supone un costo mínimo del 10% por cliente
            return {'segmento': scenario['segmento'], 'precio': P, 'volumen_mensual': Q}
    return None

# Ejecución de pruebas
if __name__ == "__main__":
    print("Elasticidad a $0.1:", elasticidad(0.1))
    print("Precio óptimo:", precio_optimo())
    equilibrio = punto_equilibrio_freemium_paid(scenarios)
    if equilibrio:
        print(f"Punto de equilibrio: {equilibrio['segmento']} - Precio=${equilibrio['precio']}, Volumen={equilibrio['volumen_mensual']}")