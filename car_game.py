import pygame
import math
import random

# Variables globales para seguimiento de récords
BEST_TIME_EVER = 30000.0 # 30 segundos iniciales como referencia

# --- Clase para la Red Neuronal (Perceptrón Multicapa Simple) ---
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # Inicialización de pesos aleatorios entre -1 y 1
        self.weights_ih = [[random.uniform(-1, 1) for _ in range(self.input_nodes)] for _ in range(self.hidden_nodes)]
        self.weights_ho = [[random.uniform(-1, 1) for _ in range(self.hidden_nodes)] for _ in range(self.output_nodes)]
        
        # Bias
        self.bias_h = [random.uniform(-1, 1) for _ in range(self.hidden_nodes)]
        self.bias_o = [random.uniform(-1, 1) for _ in range(self.output_nodes)]


    def predict(self, inputs):
        # Capa oculta
        hidden = []
        for i in range(self.hidden_nodes):
            summation = self.bias_h[i]
            for j in range(self.input_nodes):
                summation += inputs[j] * self.weights_ih[i][j]
            hidden.append(math.tanh(summation))
            
        # Capa de salida
        outputs = []
        for i in range(self.output_nodes):
            summation = self.bias_o[i]
            for j in range(self.hidden_nodes):
                summation += hidden[j] * self.weights_ho[i][j]
            outputs.append(math.tanh(summation))
            
        return outputs

    def mutate(self, rate):
        # Función para algoritmos genéticos: muta los pesos y biases
        def mutate_val(val):
            if random.random() < rate:
                return val + random.gauss(0, 0.1) # Pequeña variación gaussiana
            return val

        self.weights_ih = [[mutate_val(w) for w in row] for row in self.weights_ih]
        self.weights_ho = [[mutate_val(w) for w in row] for row in self.weights_ho]
        self.bias_h = [mutate_val(b) for b in self.bias_h]
        self.bias_o = [mutate_val(b) for b in self.bias_o]

#-------------------- mapa - imagen --------------------
import collections
import os

TRACK_IMG = "track.png"

def process_track_data(surface):
    """Procesa la superficie de la pista para encontrar la meta, generar el mapa de distancias y el punto de inicio."""
    width, height = surface.get_width(), surface.get_height()
    
    # --- Detección de meta (rojo) ---
    meta_pixels = []
    pxarray = pygame.PixelArray(surface)
    for i in range(width):
        for j in range(height):
            color = surface.unmap_rgb(pxarray[i, j])
            if color.r == 255 and color.g == 0 and color.b == 0:
                meta_pixels.append((i, j))
    pxarray.close()
    
    if not meta_pixels:
        raise ValueError("No se encontró meta (zona roja) en la pista!")

    # Calcular el centroide
    meta_x = int(sum([x for x, y in meta_pixels]) / len(meta_pixels))
    meta_y = int(sum([y for x, y in meta_pixels]) / len(meta_pixels))

    # --- Generación de mapa de distancias UNIDIRECCIONAL (BFS) ---
    dist_map = [[None for _ in range(height)] for _ in range(width)]
    max_dist = 0
    queue = collections.deque()
    
    # 1. Encontrar el lado de LLEGADA de la meta
    target_node = None
    for px, py in meta_pixels:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = px+dx, py+dy
            if 0 <= nx < width and 0 <= ny < height:
                color = surface.get_at((nx, ny))
                if color.r == 255 and color.g == 255 and color.b == 255:
                    target_node = (nx, ny)
                    break
        if target_node: break

    if not target_node:
        raise ValueError("Error: No hay pista blanca junto a la meta!")

    dist_map[target_node[0]][target_node[1]] = 0
    queue.append(target_node)
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height:
                if dist_map[nx][ny] is None:
                    color = surface.get_at((nx, ny))
                    # BLOQUEO: Los ROJOS son muros para el BFS. Obliga a dar toda la vuelta.
                    if color.r == 255 and color.g == 255 and color.b == 255:
                        dist_map[nx][ny] = dist_map[x][y] + 1
                        max_dist = max(max_dist, dist_map[nx][ny])
                        queue.append((nx, ny))
    
    # 2. Encontrar el punto de INICIO (el otro lado de la meta)
    start_x, start_y = target_node
    highest_dist = 0
    for px, py in meta_pixels:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = px+dx, py+dy
            if 0 <= nx < width and 0 <= ny < height:
                if dist_map[nx][ny] is not None and dist_map[nx][ny] > highest_dist:
                    highest_dist = dist_map[nx][ny]
                    start_x, start_y = nx, ny
    
    # 3. Orientación inicial: Mirar hacia donde baja la distancia
    start_angle = 0
    for dx, dy in [(-10,0),(10,0),(0,-10),(0,10)]:
        nx, ny = start_x+dx, start_y+dy
        if 0 <= nx < width and 0 <= ny < height and dist_map[nx][ny] is not None:
            if dist_map[nx][ny] < highest_dist:
                start_angle = math.degrees(math.atan2(ny - start_y, nx - start_x))
                break

    return meta_pixels, (meta_x, meta_y), dist_map, max_dist, (start_x, start_y, start_angle)

# Carga de imagen principal
try:
    print("Intentando cargar imagen con Pygame...")
    track_surface = pygame.image.load(TRACK_IMG)
    WIDTH, HEIGHT = track_surface.get_width(), track_surface.get_height()
except pygame.error:
    print("Fallo Pygame, intentando con Pillow...")
    try:
        from PIL import Image
        pil_img = Image.open(TRACK_IMG).convert('RGBA')
        raw_str = pil_img.tobytes()
        WIDTH, HEIGHT = pil_img.size
        track_surface = pygame.image.frombuffer(raw_str, (WIDTH, HEIGHT), 'RGBA')
    except Exception as e:
        print(f"Error crítico al cargar imagen: {e}")
        exit(1)

print(f"Imagen cargada correctamente: {WIDTH}x{HEIGHT}")

# Procesar datos de la pista
try:
    meta_pixels, meta_center, dist_map, max_dist, start_info = process_track_data(track_surface)
    meta_x, meta_y = meta_center
    start_x, start_y, start_angle = start_info
    print(f"Meta encontrada en {meta_x},{meta_y}. Distancia máxima: {max_dist}")
except ValueError as e:
    print(e)
    exit(1)


CAR_WIDTH, CAR_HEIGHT = 60, 30

# Rayos: solo delanteros para forzar avance
RAY_ANGLES = [0, -30, 30, -60, 60, -90, 90] 
RAY_LENGTH = 250



pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


# --- Clase Auto ---
class Car:
    def __init__(self, brain=None):
        self.x = start_x
        self.y = start_y
        self.angle = start_angle # Orientación inicial hacia la pista
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 5
        self.rotation_speed = 5
        self.alive = True
        self.death_reason = None # "collision" o "inactivity"
        self.fitness = 0.0 # Usamos float para evitar errores de tipo
        self.max_reached_fitness = 0.0 # Récord personal de progreso en esta vida
        self.distance_traveled = 0.0
        self.last_pista_dist = dist_map[int(self.x)][int(self.y)] if dist_map[int(self.x)][int(self.y)] else max_dist
        self.last_move_time = pygame.time.get_ticks()
        self.start_time = pygame.time.get_ticks() # Momento en que nace el coche
        
        # Red Neuronal: 7 sensores + 1 progreso = 8 entradas
        # 8 neuronas ocultas para procesar mejor la dirección, 2 salidas
        if brain:
            self.brain = brain
        else:
            self.brain = NeuralNetwork(8, 8, 2)

    def think(self, rays_norm, dist_norm):
        # 7 rayos + 1 progreso = 8 entradas
        inputs = rays_norm + [dist_norm]
        output = self.brain.predict(inputs)
        
        # SOLO AVANCE: Forzamos que 'up' sea más fácil de activar para evitar quietud
        keys = {
            "up": output[0] > -0.5,   # Umbral muy bajo para incentivar el movimiento inicial
            "down": False,            # Reversa DESACTIVADA
            "left": output[1] < -0.2, # Giro Izquierda
            "right": output[1] > 0.2  # Giro Derecha
        }
        return keys

    def move(self, keys):
        if not self.alive: return

        # Aplicación de fuerzas (SOLO POSITIVAS)
        if keys["up"]:
            self.speed += self.acceleration
        else:
            # Pequeño empuje constante para que la inercia los obligue a aprender
            self.speed += self.acceleration * 0.2
            self.speed *= 0.98 # Menor fricción
        
        # Velocidad mínima es 0, máxima es 5
        self.speed = max(0.1, min(self.max_speed, self.speed)) # Mínimo 0.1 para que siempre rueden
        
        # El giro es más efectivo si el carro se mueve (realismo)
        actual_rotation = self.rotation_speed * (abs(self.speed) / self.max_speed + 0.5)
        if keys["left"]:
            self.angle -= actual_rotation if self.speed != 0 else 0
        if keys["right"]:
            self.angle += actual_rotation if self.speed != 0 else 0

        rad = math.radians(self.angle)
        new_x = self.x + math.cos(rad) * self.speed
        new_y = self.y + math.sin(rad) * self.speed

        collision = False
        for dx, dy in [(-CAR_WIDTH/2, -CAR_HEIGHT/2), (CAR_WIDTH/2, -CAR_HEIGHT/2),
                       (CAR_WIDTH/2, CAR_HEIGHT/2), (-CAR_WIDTH/2, CAR_HEIGHT/2)]:
            cx = new_x + dx * math.cos(rad) - dy * math.sin(rad)
            cy = new_y + dx * math.sin(rad) + dy * math.cos(rad)
            if not (0 <= int(cx) < WIDTH and 0 <= int(cy) < HEIGHT):
                collision = True; break
            color = track_surface.get_at((int(cx), int(cy)))
            if color.r == 0 and color.g == 0 and color.b == 0:
                collision = True; break
            if not ((color.r == 255 and color.g == 255 and color.b == 255) or 
                    (color.r == 255 and color.g == 0 and color.b == 0)):
                collision = True; break

        if not collision:
            old_pos = (self.x, self.y)
            self.x = new_x
            self.y = new_y
            
            dist_movida = math.sqrt((self.x - old_pos[0])**2 + (self.y - old_pos[1])**2)
            if dist_movida > 0.5:
                self.last_move_time = pygame.time.get_ticks()

            # Cálculo de recompensa (fitness) PROTEGIDO contra reward hacking
            ix, iy = int(self.x), int(self.y)
            current_pista_dist = dist_map[ix][iy]
            
            if current_pista_dist is not None:
                # El progreso es inverso a la distancia a meta (1000 = meta, 0 = inicio)
                progreso_actual = (1 - current_pista_dist / max_dist) * 1000
                
                # PENALIZACIÓN POR TIEMPO: Pierde 2 puntos por segundo para incentivar la velocidad
                self.fitness -= 0.033 # Aprox 2 puntos por segundo a 60 FPS
                
                # Checkpoint Fitness: Bonos por tramos completados (25%, 50%, 75% del circuito)
                for threshold in [250, 500, 750]:
                    if progreso_actual > threshold and self.max_reached_fitness <= threshold:
                        self.fitness += 1000 # Bono de hito mayor
                            
                # REGLA UNIDIRECCIONAL: Solo premiar si supera su récord de progreso en esta vida
                if progreso_actual > self.max_reached_fitness:
                    avance = progreso_actual - self.max_reached_fitness
                    self.fitness += avance
                    self.max_reached_fitness = progreso_actual
                else:
                    # PENALIZACIÓN POR QUEDARSE QUIETO: Si no progresa, pierde fitness lentamente
                    self.fitness -= 0.05

                    # MUERTE POR SENTIDO CONTRARIO: 
                    # Si el coche retrocede significativamente respecto a su mejor progreso, muere.
                    if progreso_actual < self.max_reached_fitness - 50:
                        self.alive = False
                        self.death_reason = "wrong_way"

                # DETECCIÓN DE PASO POR META (SENTIDO CORRECTO)
                # Un coche solo puede ganar si ha recorrido al menos el 80% de la pista
                if current_pista_dist < 25:
                    if self.max_reached_fitness > 800: # Ha completado el 80% del progreso
                        global BEST_TIME_EVER
                        time_taken = pygame.time.get_ticks() - self.start_time
                        
                        time_multiplier = max(1.0, BEST_TIME_EVER / max(1, time_taken))
                        
                        if time_taken < BEST_TIME_EVER:
                            BEST_TIME_EVER = float(time_taken)
                            print(f"¡NUEVO RÉCORD HISTÓRICO: {BEST_TIME_EVER/1000:.2f}s!")
                        
                        self.fitness += 10000 * time_multiplier
                        self.alive = False
                        self.death_reason = "finished"
                        print(f"¡Meta alcanzada! Tiempo: {time_taken/1000:.2f}s | Multiplicador: x{time_multiplier:.2f}")
                    else:
                        # Si toca la meta sin haber dado la vuelta (trampa)
                        # No muere necesariamente, pero no gana el bono de meta.
                        # Si retrocede mucho, ya tenemos la lógica de "wrong_way" abajo.
                        pass

                self.last_pista_dist = current_pista_dist
        else:
            self.alive = False
            self.death_reason = "collision"
            self.speed = 0

    def draw(self, surface):
        rad = math.radians(self.angle)
        # Calcular esquinas del auto para rotar el rectángulo
        points = []
        for dx, dy in [(-CAR_WIDTH/2, -CAR_HEIGHT/2), (CAR_WIDTH/2, -CAR_HEIGHT/2),
                       (CAR_WIDTH/2, CAR_HEIGHT/2), (-CAR_WIDTH/2, CAR_HEIGHT/2)]:
            x = self.x + dx * math.cos(rad) - dy * math.sin(rad)
            y = self.y + dx * math.sin(rad) + dy * math.cos(rad)
            points.append((x, y))
        
        # Color según estado
        if self.alive:
            color = (0, 120, 255) # Azul si está vivo
        else:
            if self.death_reason == "collision":
                color = (200, 0, 0) # Rojo si chocó
            elif self.death_reason == "inactivity":
                color = (100, 100, 100) # Gris si fue por inactividad
            elif self.death_reason == "finished":
                color = (0, 255, 0) # Verde si completó el circuito
            elif self.death_reason == "wrong_way":
                color = (255, 165, 0) # Naranja si fue en sentido contrario
            else:
                color = (50, 50, 50)
        
        pygame.draw.polygon(surface, color, points)

    def get_ray_origins(self):
        # 5 rayos saliendo del centro del auto, por ángulos relativos
        origins = []
        for offset_angle in RAY_ANGLES:
            ray_angle = self.angle + offset_angle
            x = self.x
            y = self.y
            origins.append((x, y, ray_angle))
        return origins

# --- Raycasting ---
def cast_ray(start_x, start_y, angle_deg):
    angle_rad = math.radians(angle_deg)
    step = 2  # pixels
    for length in range(0, RAY_LENGTH, step):
        px = int(start_x + math.cos(angle_rad) * length)
        py = int(start_y + math.sin(angle_rad) * length)
        # Si está fuera de pista, termina el rayo
        if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
            return (px, py), length
        color = track_surface.get_at((px, py))
        if color.r == 0 and color.g == 0 and color.b == 0:
            # muro
            return (px, py), length
    # Si no colisiona, devuelve el final
    px = int(start_x + math.cos(angle_rad) * RAY_LENGTH)
    py = int(start_y + math.sin(angle_rad) * RAY_LENGTH)
    return (px, py), RAY_LENGTH

# --- Main ---
population_size = 50
generation = 1
best_brain = None

def create_generation(size, best_brain=None):
    cars = []
    import copy
    
    if best_brain is None:
        return [Car() for _ in range(size)]

    # 1. ÉLITE (El Campeón): 2 copias EXACTAS
    cars.append(Car(brain=copy.deepcopy(best_brain)))
    cars.append(Car(brain=copy.deepcopy(best_brain)))

    # 2. SUCESORES DIRECTOS: 40% de la población.
    # Mutación mínima (0.01) para perfeccionar al ganador.
    num_sucessors = int(size * 0.4)
    for _ in range(num_sucessors):
        child_brain = copy.deepcopy(best_brain)
        child_brain.mutate(0.01)
        cars.append(Car(brain=child_brain))

    # 3. EXPLORADORES: 30% de la población.
    # Mutación moderada (0.1) para buscar alternativas.
    num_explorers = int(size * 0.3)
    for _ in range(num_explorers):
        child_brain = copy.deepcopy(best_brain)
        child_brain.mutate(0.1)
        cars.append(Car(brain=child_brain))

    # 4. REFUERZO ALEATORIO: El resto (10-20%).
    # Con mutación agresiva (0.2)
    while len(cars) < size:
        new_car = Car()
        new_car.brain.mutate(0.2)
        cars.append(new_car)

    return cars

population_size = 50 # Aumentamos población a 50 carros
cars = create_generation(population_size)
running = True
last_print_time = 0

while running:
    screen.fill((30, 30, 50))
    screen.blit(track_surface, (0, 0))

    # Controlar si todos los carros han muerto
    all_dead = True
    current_time = pygame.time.get_ticks()

    for car in cars:
        # Dibujar siempre, estén vivos o muertos (para ver los colores)
        car.draw(screen)

        if car.alive:
            # Ampliamos el tiempo de inactividad a 10 segundos para dar tiempo 
            # a que el carro frene, de reversa y maniobre sin ser cortado.
            if current_time - car.last_move_time > 10000:
                car.alive = False
                car.death_reason = "inactivity"
                continue

            all_dead = False
            
            # 1. Obtener datos de sensores
            ray_origins = car.get_ray_origins()
            hit_points = []
            rays_norm = []
            for sx, sy, angle in ray_origins:
                end, dist = cast_ray(sx, sy, angle)
                rays_norm.append(round(math.tanh((dist - RAY_LENGTH/2)/(RAY_LENGTH/8)), 4))
                hit_points.append((end, dist))
            
            cx, cy = int(car.x), int(car.y)
            pista_dist = dist_map[cx][cy] if 0 <= cx < WIDTH and 0 <= cy < HEIGHT else None
            dist_norm = round(1 - pista_dist / max_dist, 4) if (pista_dist is not None and max_dist > 0) else 0
            
            # Identificar carros vivos para visualización
            alive_cars = [c for c in cars if c.alive]

            # 2. Red Neuronal piensa y mueve
            nn_keys = car.think(rays_norm, dist_norm)
            car.move(nn_keys)

            # Penalización por cercanía excesiva a muros (Reward Shaping)
            min_ray = min(rays_norm)
            if min_ray < 0.2:
                car.fitness -= 0.1 
            
            # Dibujar rayos del primer carro vivo para visualizar
            alive_cars = [c for c in cars if c.alive]
            if len(alive_cars) > 0 and car == alive_cars[0]:
                for sx, sy, angle in ray_origins:
                    end, _ = cast_ray(sx, sy, angle)
                    pygame.draw.line(screen, (255, 255, 100, 100), (sx, sy), end, 1)

    # Si todos chocaron, pasar a la siguiente generación
    if all_dead:
        # Elegir el mejor basado en fitness
        cars.sort(key=lambda x: x.fitness, reverse=True)
        best_car = cars[0]
        best_brain = best_car.brain
        
        print(f"Generación {generation} terminada. Mejor fitness: {best_car.fitness}")
        
        generation += 1
        cars = create_generation(population_size, best_brain)

    # Imprimir info cada segundo
    curr_time = pygame.time.get_ticks()
    if curr_time - last_print_time > 1000:
        vivos = len([c for c in cars if c.alive])
        print(f"Gen: {generation} | Vivos: {vivos}/{population_size}")
        last_print_time = curr_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
