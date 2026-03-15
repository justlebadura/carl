import pygame
import math
import random

# --- Clase para la Red Neuronal (Perceptrón Multicapa Simple) ---
class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        # Inicialización de pesos aleatorios entre -1 y 1
        self.weights_ih = [[random.uniform(-1, 1) for _ in range(self.input_nodes)] for _ in range(self.hidden_nodes)]
        self.weights_ho = [[random.uniform(-1, 1) for _ in range(self.hidden_nodes)] for _ in range(self.output_nodes)]
        
        # Biases
        self.bias_h = [random.uniform(-1, 1) for _ in range(self.hidden_nodes)]
        self.bias_o = [random.uniform(-1, 1) for _ in range(self.output_nodes)]

    def _activate(self, x):
        # Usamos la función sigmoide definida globalmente
        return sigmoid(x)

    def predict(self, inputs):
        # Capa oculta (mantenemos sigmoide)
        hidden = []
        for i in range(self.hidden_nodes):
            summation = self.bias_h[i]
            for j in range(self.input_nodes):
                summation += inputs[j] * self.weights_ih[i][j]
            hidden.append(1 / (1 + math.exp(-max(-10, min(10, summation)))))
            
        # Capa de salida (usamos Tanh para el giro)
        outputs = []
        for i in range(self.output_nodes):
            summation = self.bias_o[i]
            for j in range(self.hidden_nodes):
                summation += hidden[j] * self.weights_ho[i][j]
            # Aplicamos Tanh: (e^x - e^-x) / (e^x + e^-x)
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

def sigmoid(x):
    '''Función sigmoide estándar.'''
    return 1 / (1 + math.exp(-x))


# --- Configuración inicial ---
import collections

# --- Aquí solo configuraciones; la detección de meta y mapa de distancias va después de cargar la imagen ---

import os
TRACK_IMG = "track.png"  # Cambia a tu archivo

# Cargar imagen de pista
try:
    track_surface = pygame.image.load(TRACK_IMG)
    WIDTH, HEIGHT = track_surface.get_width(), track_surface.get_height()
    print("Imagen de pista cargada con Pygame.")

    # --- Detección de meta (rojo)
    # Buscar todos los píxeles rojos (meta) y calcular el centro
    meta_pixels = []
    pxarray = pygame.PixelArray(track_surface)
    for i in range(WIDTH):
        for j in range(HEIGHT):
            color = track_surface.unmap_rgb(pxarray[i, j])
            if color.r == 255 and color.g == 0 and color.b == 0:
                meta_pixels.append((i, j))
    pxarray.close()
    if not meta_pixels:
        print("No se encontró meta (zona roja) en la pista!")
        exit(1)
    # Calcular el centroide
    meta_x = int(sum([x for x, y in meta_pixels]) / len(meta_pixels))
    meta_y = int(sum([y for x, y in meta_pixels]) / len(meta_pixels))

    # --- Generación de mapa de distancias UNIDIRECCIONAL (BFS) ---
    import collections
    dist_map = [[None for _ in range(HEIGHT)] for _ in range(WIDTH)]
    max_dist = 0
    queue = collections.deque()
    
    # Para hacerla unidireccional, elegimos un píxel blanco ADYACENTE a la meta como el "0"
    # y el resto de la meta como un MURO para el BFS.
    start_white_node = None
    for px, py in meta_pixels:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = px+dx, py+dy
            if 0<=nx<WIDTH and 0<=ny<HEIGHT:
                color = track_surface.get_at((nx, ny))
                if color.r == 255 and color.g == 255 and color.b == 255:
                    start_white_node = (nx, ny)
                    break
        if start_white_node: break

    if not start_white_node:
        print("Error: No hay pista blanca junto a la meta!")
        exit(1)

    # El BFS empieza en el píxel blanco "detrás" de la meta
    dist_map[start_white_node[0]][start_white_node[1]] = 0
    queue.append(start_white_node)
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<WIDTH and 0<=ny<HEIGHT:
                if dist_map[nx][ny] is None:
                    color = track_surface.get_at((nx, ny))
                    # Solo permitimos pasar por PIXELS BLANCOS. 
                    # Los ROJOS (meta) actúan como muro para el BFS, obligando a dar la vuelta.
                    if color.r == 255 and color.g == 255 and color.b == 255:
                        dist_map[nx][ny] = dist_map[x][y] + 1
                        max_dist = max(max_dist, dist_map[nx][ny])
                        queue.append((nx, ny))
    
    # El auto inicia en el punto blanco "delante" de la meta (el que tiene la distancia más alta cerca de la meta)
    # o simplemente en el punto blanco encontrado antes para no complicar.
    start_x, start_y = start_white_node[0], start_white_node[1]
    # (Opcional) Si quieres que aparezca al otro lado de la meta:
    best_start_dist = 0
    for px, py in meta_pixels:
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = px+dx, py+dy
            if 0<=nx<WIDTH and 0<=ny<HEIGHT:
                if dist_map[nx][ny] is not None and dist_map[nx][ny] > best_start_dist:
                    best_start_dist = dist_map[nx][ny]
                    start_x, start_y = nx, ny

except pygame.error as e:
    print(f"Error al cargar imagen con Pygame: {e}")
    print("Intentando con Pillow...")
    try:
        from PIL import Image
        pil_img = Image.open(TRACK_IMG).convert('RGBA')
        raw_str = pil_img.tobytes()
        WIDTH, HEIGHT = pil_img.size
        track_surface = pygame.image.frombuffer(raw_str, (WIDTH, HEIGHT), 'RGBA')
        print("Imagen de pista cargada con Pillow!")
        # --- Detección de meta (rojo) (Pillow fallback) ---
        meta_pixels = []
        pxarray = pygame.PixelArray(track_surface)
        for i in range(WIDTH):
            for j in range(HEIGHT):
                color = track_surface.unmap_rgb(pxarray[i, j])
                if color.r == 255 and color.g == 0 and color.b == 0:
                    meta_pixels.append((i, j))
        pxarray.close()
        if not meta_pixels:
            print("No se encontró meta (zona roja) en la pista!")
            exit(1)
        
        meta_x = int(sum([x for x, y in meta_pixels]) / len(meta_pixels))
        meta_y = int(sum([y for x, y in meta_pixels]) / len(meta_pixels))

        # --- Generación de mapa de distancias UNIDIRECCIONAL (BFS) ---
        dist_map = [[None for _ in range(HEIGHT)] for _ in range(WIDTH)]
        max_dist = 0
        queue = collections.deque()
        
        # Elegimos el primer píxel blanco adyacente a la meta como el final del circuito (distancia 0)
        start_white_node = None
        for px, py in meta_pixels:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = px+dx, py+dy
                if 0<=nx<WIDTH and 0<=ny<HEIGHT:
                    color = track_surface.get_at((nx, ny))
                    if color.r == 255 and color.g == 255 and color.b == 255:
                        start_white_node = (nx, ny)
                        break
            if start_white_node: break

        if not start_white_node:
            print("Error: No hay pista blanca junto a la meta!")
            exit(1)

        dist_map[start_white_node[0]][start_white_node[1]] = 0
        queue.append(start_white_node)
        
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = x+dx, y+dy
                if 0<=nx<WIDTH and 0<=ny<HEIGHT:
                    if dist_map[nx][ny] is None:
                        color = track_surface.get_at((nx, ny))
                        # Solo BLANCOS. ROJOS (meta) bloquean el paso para obligar a dar la vuelta completa
                        if color.r == 255 and color.g == 255 and color.b == 255:
                            dist_map[nx][ny] = dist_map[x][y] + 1
                            max_dist = max(max_dist, dist_map[nx][ny])
                            queue.append((nx, ny))
        
        # El auto inicia en el punto blanco con mayor distancia al final (el otro lado de la meta)
        start_x, start_y = start_white_node[0], start_white_node[1]
        best_start_dist = 0
        for px, py in meta_pixels:
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                nx, ny = px+dx, py+dy
                if 0<=nx<WIDTH and 0<=ny<HEIGHT:
                    if dist_map[nx][ny] is not None and dist_map[nx][ny] > best_start_dist:
                        best_start_dist = dist_map[nx][ny]
                        start_x, start_y = nx, ny


    except Exception as e2:
        print(f"Error al cargar imagen con Pillow: {e2}")
        print("Asegúrate de que el archivo tiene formato PNG válido y está en el directorio.")
        exit(1)

CAR_WIDTH, CAR_HEIGHT = 60, 30

# Rayos: delanteros y traseros
RAY_ANGLES = [
    0, -45, 45, -90, 90,          # Adelante
    180, 180-45, 180+45, 180-90, 180+90   # Atrás
]
RAY_LENGTH = 250



pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


# --- Clase Auto ---
class Car:
    def __init__(self, brain=None):
        self.x = start_x
        self.y = start_y
        self.angle = 0  # en grados
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.2
        self.rotation_speed = 4
        self.alive = True
        self.death_reason = None # "collision" o "inactivity"
        self.fitness = 0
        self.distance_traveled = 0
        self.last_pista_dist = dist_map[int(self.x)][int(self.y)] if dist_map[int(self.x)][int(self.y)] else max_dist
        self.last_move_time = pygame.time.get_ticks()
        
        # Red Neuronal: 10 sensores + 1 distancia meta = 11 entradas
        # 4 neuronas ocultas, 2 salidas (acel/fren, giro)
        if brain:
            self.brain = brain
        else:
            self.brain = NeuralNetwork(10, 6, 2)

    def think(self, rays_norm, dist_norm):
        inputs = rays_norm + [dist_norm]
        output = self.brain.predict(inputs)
        
        # Con Tanh en la salida:
        # output[0] controla Aceleración/Freno
        # output[1] controla Giro (Izquierda < 0, Derecha > 0)
        keys = {
            "up": output[0] > 0,      # Acelera si es positivo
            "down": output[0] < -0.3, # Frena si es muy negativo
            "left": output[1] < -0.1, # Gira izquierda si es negativo
            "right": output[1] > 0.1  # Gira derecha si es positivo
        }
        return keys

    def move(self, keys):
        if not self.alive: return

        if keys["up"]:
            self.speed += self.acceleration
        elif keys["down"]:
            self.speed -= self.acceleration
        else:
            self.speed *= 0.98

        self.speed = max(-self.max_speed, min(self.max_speed, self.speed))
        if keys["left"]:
            self.angle -= self.rotation_speed if self.speed != 0 else 0
        if keys["right"]:
            self.angle += self.rotation_speed if self.speed != 0 else 0

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
            # Guardamos posición anterior para verificar movimiento real
            old_pos = (self.x, self.y)
            self.x = new_x
            self.y = new_y
            
            # Si el carro ha cambiado de posición significativamente (más de 0.5 píxeles)
            dist_movida = math.sqrt((self.x - old_pos[0])**2 + (self.y - old_pos[1])**2)
            if dist_movida > 0.5:
                self.last_move_time = pygame.time.get_ticks()

            # Cálculo de recompensa (fitness) mejorado
            ix, iy = int(self.x), int(self.y)
            current_pista_dist = dist_map[ix][iy]
            
            if current_pista_dist is not None:
                # 1. Recompensa por progreso real (sentido correcto)
                if self.last_pista_dist is not None and current_pista_dist < self.last_pista_dist:
                    progreso = (self.last_pista_dist - current_pista_dist)
                    # El progreso vale más si vas rápido (evita estancamiento)
                    self.fitness += progreso * (1 + self.speed / self.max_speed)
                
                # 2. Pequeña recompensa constante por seguir vivo (solo si se mueve)
                if abs(self.speed) > 1.0:
                    self.fitness += 0.1
                
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
population_size = 20
generation = 1
best_brain = None

def create_generation(size, best_brain=None):
    cars = []
    for i in range(size):
        if best_brain:
            # Clonar el mejor cerebro y mutarlo
            import copy
            new_brain = copy.deepcopy(best_brain)
            if i == 0:
                # El mejor de la anterior pasa sin cambios (Elitismo)
                pass
            elif i > size - 3:
                # Los últimos 2 son totalmente aleatorios (Sangre nueva)
                new_brain = NeuralNetwork(10, 6, 2)
            else:
                # El resto muta con intensidad variable
                intensity = 0.2 if i > size // 2 else 0.05
                new_brain.mutate(intensity)
            cars.append(Car(brain=new_brain))
        else:
            cars.append(Car())
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
            # 0. Verificar si el carro se ha quedado quieto demasiado tiempo (3 segundos)
            if current_time - car.last_move_time > 3000:
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
                rays_norm.append(round(sigmoid((dist - RAY_LENGTH/2)/(RAY_LENGTH/8)), 4))
                hit_points.append((end, dist))
            
            cx, cy = int(car.x), int(car.y)
            pista_dist = dist_map[cx][cy] if 0 <= cx < WIDTH and 0 <= cy < HEIGHT else None
            dist_norm = round(1 - pista_dist / max_dist, 4) if pista_dist is not None and max_dist > 0 else 0
            
            # 2. Red Neuronal piensa y mueve
            nn_keys = car.think(rays_norm, dist_norm)
            car.move(nn_keys)

            # Penalización por cercanía excesiva a muros (Reward Shaping)
            min_ray = min(rays_norm)
            if min_ray < 0.2:  # Si algún sensor detecta un muro muy cerca
                car.fitness -= 0.5  # Penalización leve por conducción peligrosa
            
            # 3. Dibujar (Movido fuera para ver los muertos)
            # car.draw(screen) 
            
            # Dibujar rayos del primer carro vivo para visualizar
            alive_cars = [c for c in cars if c.alive]
            if len(alive_cars) > 0 and car == alive_cars[0]:
                for sx, sy, angle in ray_origins:
                    end, _ = cast_ray(sx, sy, angle)
                    pygame.draw.line(screen, (255, 255, 100, 100), (sx, sy), end, 1)

        # Dibujar UI
        # Eliminado debido a error en módulo pygame.font
        # font = pygame.font.SysFont("Arial", 24)
        # gen_text = font.render(f"Generación: {generation}", True, (255, 255, 255))
        # alive_text = font.render(f"Vivos: {len([c for c in cars if c.alive])}/{population_size}", True, (255, 255, 255))
        # screen.blit(gen_text, (10, 10))
        # screen.blit(alive_text, (10, 40))

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
