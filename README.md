# 🏎️ CARL — Coches Autónomos con Red neuronal y aLgoritmo genético

**CARL** es una simulación de autos de carrera que aprenden a conducir solos usando una **red neuronal artificial** entrenada mediante un **algoritmo genético evolutivo**. Todo está construido con Python y Pygame, sin librerías de machine-learning externas.

---

## 📋 Tabla de contenidos

1. [Descripción general](#descripción-general)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Cómo ejecutar](#cómo-ejecutar)
4. [Cómo funciona](#cómo-funciona)
   - [Red neuronal](#red-neuronal)
   - [Algoritmo genético](#algoritmo-genético)
   - [Sensores (raycasting)](#sensores-raycasting)
   - [Sistema de fitness](#sistema-de-fitness)
   - [Mapa y pista](#mapa-y-pista)
5. [Estructura del proyecto](#estructura-del-proyecto)
6. [Configuración y parámetros](#configuración-y-parámetros)
7. [Visualización](#visualización)

---

## Descripción general

En cada "generación", una **población de 50 autos** es lanzada sobre la pista. Cada auto está controlado por su propia red neuronal que recibe información del entorno (sensores de distancia a los muros) y decide cuándo acelerar y hacia dónde girar. Los autos que llegan más lejos —o completan el circuito más rápido— transfieren sus "genes" (pesos de la red neuronal) a la siguiente generación. Con el paso de las generaciones, los autos aprenden a conducir de forma progresivamente más eficiente.

---

## Requisitos e instalación

### Requisitos del sistema

- Python **3.8** o superior
- Sistema operativo: Windows, macOS o Linux

### Dependencias

| Paquete   | Uso                                      |
|-----------|------------------------------------------|
| `pygame`  | Visualización gráfica y bucle de juego   |
| `Pillow`  | Carga alternativa de la imagen de pista  |

### Instalación paso a paso

1. **Clona el repositorio:**

   ```bash
   git clone https://github.com/justlebadura/carl.git
   cd carl
   ```

2. **Crea un entorno virtual (recomendado):**

   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En macOS/Linux:
   source venv/bin/activate
   ```

3. **Instala las dependencias:**

   ```bash
   pip install pygame Pillow
   ```

---

## Cómo ejecutar

```bash
python car_game.py
```

Se abrirá una ventana gráfica con la pista y los autos. No se requiere ningún argumento adicional.

Para cerrar la simulación, cierra la ventana o presiona **Alt + F4**.

---

## Cómo funciona

### Red neuronal

Cada auto tiene su propia red neuronal de tipo **Perceptrón Multicapa (MLP)** implementada desde cero (sin TensorFlow ni PyTorch).

```
Entradas (8)  →  Capa oculta (8 neuronas)  →  Salidas (2)
```

| Capa    | Descripción                                                              |
|---------|--------------------------------------------------------------------------|
| Entrada | 7 lecturas de sensores de distancia + 1 valor de progreso en la pista   |
| Oculta  | 8 neuronas con función de activación **tanh**                            |
| Salida  | Neurona 1 → Acelerar / Neurona 2 → Girar (negativo = izquierda, positivo = derecha) |

Los pesos de conexión entre capas se inicializan aleatoriamente en el rango `[-1, 1]`.

---

### Algoritmo genético

Al finalizar cada generación (cuando todos los autos han muerto), se aplica la siguiente estrategia evolutiva para crear la siguiente población:

| Grupo               | % de población | Descripción                                                   |
|---------------------|----------------|---------------------------------------------------------------|
| **Élite**           | ~4% (2 copias) | Copias exactas del mejor auto de la generación anterior       |
| **Sucesores**       | ~40%           | Copias del mejor auto con mutación mínima (`rate = 0.01`)     |
| **Exploradores**    | ~30%           | Copias con mutación moderada (`rate = 0.1`)                   |
| **Aleatorios**      | ~26%           | Autos nuevos con mutación agresiva (`rate = 0.2`)             |

La **mutación** modifica cada peso con una pequeña variación gaussiana (`σ = 0.1`), según la tasa de mutación configurada para cada grupo.

---

### Sensores (raycasting)

Cada auto emite **7 rayos** desde su centro hacia distintos ángulos relativos a su dirección de movimiento:

```
Ángulos: 0°, -30°, 30°, -60°, 60°, -90°, 90°
Longitud máxima: 250 píxeles
```

Cada rayo avanza píxel a píxel hasta topar con un muro (píxel negro) o llegar al límite de longitud. La distancia resultante se normaliza con `tanh` y se usa como entrada de la red neuronal.

El rayo del primer auto vivo se **dibuja en pantalla** (color amarillo tenue) para facilitar la visualización.

---

### Sistema de fitness

El **fitness** (puntuación) determina cuáles autos son los "mejores" y se heredan a la siguiente generación. Se calcula en tiempo real de la siguiente forma:

| Evento                             | Efecto en fitness          |
|------------------------------------|----------------------------|
| Avanzar en la pista (nuevo récord personal) | +progreso obtenido        |
| Alcanzar el 25%, 50% o 75% del circuito     | +1 000 puntos (bono)      |
| Completar la vuelta completa (>80% previo)  | +10 000 × multiplicador de tiempo |
| Pasar tiempo sin avanzar           | -0.05 por frame            |
| Penalización por tiempo (60 FPS)   | -0.033 por frame           |
| Acercarse demasiado a un muro      | -0.1 por frame             |
| Conducir en sentido contrario (retroceso > 50 unidades) | Muerte instantánea |

El **multiplicador de tiempo** premia la velocidad: si el auto bate el récord histórico, la recompensa se amplifica proporcionalmente.

#### Causas de muerte

| Causa          | Color del auto | Descripción                          |
|----------------|----------------|--------------------------------------|
| `collision`    | 🔴 Rojo        | Chocó con un muro                    |
| `inactivity`   | ⚫ Gris        | Sin movimiento durante 10 segundos   |
| `finished`     | 🟢 Verde       | Completó el circuito                 |
| `wrong_way`    | 🟠 Naranja     | Retrocedió más de 50 unidades        |

---

### Mapa y pista

La pista se define mediante una imagen PNG (`track.png`). Los colores tienen los siguientes significados:

| Color         | RGB              | Rol                         |
|---------------|------------------|-----------------------------|
| **Blanco**    | `(255, 255, 255)` | Zona de carretera válida    |
| **Negro**     | `(0, 0, 0)`       | Muro / fuera de pista       |
| **Rojo**      | `(255, 0, 0)`     | Línea de meta / salida      |

Al iniciar el programa, se realiza un **BFS (Búsqueda en Anchura)** sobre los píxeles blancos desde el lado de llegada de la meta. Esto genera un **mapa de distancias** que indica, para cada píxel de la pista, a cuántos pasos está de la meta. Este mapa se usa para:

- Determinar el punto y ángulo de inicio de los autos.
- Calcular el progreso de cada auto en tiempo real.
- Detectar si un auto está yendo en sentido contrario.

Para añadir pistas personalizadas, simplemente reemplaza `track.png` con una nueva imagen que respete la convención de colores anterior y coloca la imagen dentro de la carpeta `tracks/`.

---

## Estructura del proyecto

```
carl/
├── car_game.py      # Código principal (red neuronal, algoritmo genético, simulación)
├── track.png        # Imagen de la pista activa
└── tracks/          # Carpeta con pistas adicionales
    └── track.png
```

---

## Configuración y parámetros

Los siguientes parámetros se pueden ajustar directamente en `car_game.py`:

| Parámetro          | Valor por defecto | Descripción                                      |
|--------------------|-------------------|--------------------------------------------------|
| `population_size`  | `50`              | Número de autos por generación                   |
| `RAY_ANGLES`       | `[0,-30,30,-60,60,-90,90]` | Ángulos de los 7 sensores              |
| `RAY_LENGTH`       | `250`             | Longitud máxima de cada rayo (en píxeles)        |
| `CAR_WIDTH`        | `60`              | Ancho del auto (para detección de colisiones)    |
| `CAR_HEIGHT`       | `30`              | Alto del auto                                    |
| `max_speed`        | `5`               | Velocidad máxima del auto                        |
| `acceleration`     | `0.2`             | Aceleración por frame                            |
| `rotation_speed`   | `4`               | Velocidad angular de giro                        |
| `BEST_TIME_EVER`   | `30000` ms        | Referencia de tiempo inicial para récords        |

---

## Visualización

Durante la simulación, la consola muestra:

```
Gen: 5 | Vivos: 23/50
Generación 5 terminada. Mejor fitness: 2847.3
¡Meta alcanzada! Tiempo: 18.43s | Multiplicador: x1.63
¡NUEVO RÉCORD HISTÓRICO: 18430.0ms!
```

En la ventana gráfica:

- **Autos azules** 🔵 → Activos y circulando.
- **Autos rojos** 🔴 → Chocados.
- **Autos verdes** 🟢 → Completaron el circuito.
- **Autos naranja** 🟠 → Eliminados por ir en sentido contrario.
- **Autos grises** ⚫ → Eliminados por inactividad.
- **Líneas amarillas** → Rayos del sensor del primer auto vivo.
