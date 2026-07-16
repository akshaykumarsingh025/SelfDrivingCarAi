import pygame
import numpy as np
import math
import random
import sys
import os
import pickle

MIN_WIDTH = 800
MIN_HEIGHT = 500
DEFAULT_WIDTH = 1200
DEFAULT_HEIGHT = 800

CANVAS_COLOR = (30, 30, 30)
WALL_COLOR = (200, 200, 200)
CAR_COLOR = (0, 180, 255)
SENSOR_COLOR = (255, 50, 50)
UI_BG_COLOR = (20, 20, 20)
UI_TEXT_COLOR = (220, 220, 220)
BUTTON_COLOR = (55, 55, 55)
BUTTON_HOVER = (75, 75, 75)
BUTTON_ACTIVE = (0, 140, 75)
BUTTON_ACTIVE_RED = (180, 40, 40)
BUTTON_TOOL = (50, 50, 90)
BUTTON_TOOL_ACTIVE = (80, 80, 170)
SEPARATOR_COLOR = (50, 50, 50)
GRID_COLOR = (40, 40, 40)
TOP_BAR_H = 44
BOTTOM_BAR_H = 58
TRAIL_COLOR = (0, 100, 200)

DRAW_MODE = 0
TRAIN_MODE = 1
RUN_MODE = 2
TOOL_BRUSH = 0
TOOL_ERASER = 1
TOOL_CAR = 2

NUM_SENSORS = 3
SENSOR_LENGTH = 120
SENSOR_ANGLES = [-40, 0, 40]

CAR_WIDTH = 20
CAR_HEIGHT = 36
CAR_SPEED = 3
CAR_TURN_SPEED = 8

STATE_BINS = 5
ACTION_LEFT = 0
ACTION_STRAIGHT = 1
ACTION_RIGHT = 2
NUM_ACTIONS = 3

LEARNING_RATE = 0.15
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9995

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "saves")


class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((STATE_BINS, STATE_BINS, STATE_BINS, NUM_ACTIONS))
        self.epsilon = EPSILON_START
        self.lr = LEARNING_RATE
        self.gamma = DISCOUNT_FACTOR
        self.total_episodes = 0

    def discretize(self, sensor_values):
        state = []
        for val in sensor_values:
            bin_idx = int(np.clip(val * STATE_BINS, 0, STATE_BINS - 1))
            state.append(bin_idx)
        return tuple(state)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, NUM_ACTIONS - 1)
        return int(np.argmax(self.q_table[state[0], state[1], state[2]]))

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state[0], state[1], state[2], action]
        max_next_q = np.max(self.q_table[next_state[0], next_state[1], next_state[2]])
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state[0], state[1], state[2], action] = new_q

    def decay_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def reset_exploration(self):
        self.epsilon = EPSILON_START

    def save(self, filepath):
        data = {
            "q_table": self.q_table,
            "epsilon": self.epsilon,
            "lr": self.lr,
            "gamma": self.gamma,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = data["q_table"]
        self.epsilon = data.get("epsilon", EPSILON_START)
        self.lr = data.get("lr", LEARNING_RATE)
        self.gamma = data.get("gamma", DISCOUNT_FACTOR)


class Car:
    def __init__(self, x, y, angle=0, bounds_w=1200, bounds_h=800):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = CAR_SPEED
        self.sensor_readings = [1.0] * NUM_SENSORS
        self.sensor_endpoints = []
        self.alive = True
        self.distance_traveled = 0
        self.steps_alive = 0
        self.bounds_w = bounds_w
        self.bounds_h = bounds_h
        self.trail = []
        self.trail_max = 2000

    def get_vertices(self):
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        hw = CAR_WIDTH / 2
        hh = CAR_HEIGHT / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        vertices = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + self.x
            ry = cx * sin_a + cy * cos_a + self.y
            vertices.append((rx, ry))
        return vertices

    def update_sensors(self, wall_surface):
        self.sensor_endpoints = []
        self.sensor_readings = []
        rad = math.radians(self.angle)

        for i, s_angle in enumerate(SENSOR_ANGLES):
            total_rad = rad + math.radians(s_angle)
            dx = math.sin(total_rad)
            dy = -math.cos(total_rad)

            reading = 1.0
            end_x = self.x + dx * SENSOR_LENGTH
            end_y = self.y + dy * SENSOR_LENGTH

            for dist in range(5, SENSOR_LENGTH, 2):
                px = int(self.x + dx * dist)
                py = int(self.y + dy * dist)

                if 0 <= px < self.bounds_w and 0 <= py < self.bounds_h:
                    try:
                        pixel = wall_surface.get_at((px, py))
                        if pixel[0] > 100 and pixel[1] > 100 and pixel[2] > 100:
                            reading = dist / SENSOR_LENGTH
                            end_x = self.x + dx * dist
                            end_y = self.y + dy * dist
                            break
                    except IndexError:
                        reading = dist / SENSOR_LENGTH
                        end_x = self.x + dx * dist
                        end_y = self.y + dy * dist
                        break
                else:
                    reading = dist / SENSOR_LENGTH
                    end_x = self.x + dx * dist
                    end_y = self.y + dy * dist
                    break

            self.sensor_readings.append(reading)
            self.sensor_endpoints.append((end_x, end_y))

    def check_collision(self, wall_surface):
        vertices = self.get_vertices()
        for vx, vy in vertices:
            px, py = int(vx), int(vy)
            if px < 0 or px >= self.bounds_w or py < 0 or py >= self.bounds_h:
                return True
            try:
                pixel = wall_surface.get_at((px, py))
                if pixel[0] > 100 and pixel[1] > 100 and pixel[2] > 100:
                    return True
            except IndexError:
                return True

        for check_dist in range(2, int(CAR_HEIGHT / 2), 3):
            rad = math.radians(self.angle)
            cx = self.x + math.sin(rad) * check_dist
            cy = self.y - math.cos(rad) * check_dist
            px, py = int(cx), int(cy)
            if 0 <= px < self.bounds_w and 0 <= py < self.bounds_h:
                try:
                    pixel = wall_surface.get_at((px, py))
                    if pixel[0] > 100 and pixel[1] > 100 and pixel[2] > 100:
                        return True
                except IndexError:
                    return True
        return False

    def move(self, action):
        if action == ACTION_LEFT:
            self.angle -= CAR_TURN_SPEED
        elif action == ACTION_RIGHT:
            self.angle += CAR_TURN_SPEED

        rad = math.radians(self.angle)
        self.x += math.sin(rad) * self.speed
        self.y -= math.cos(rad) * self.speed
        self.steps_alive += 1
        self.distance_traveled += self.speed
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > self.trail_max:
            self.trail.pop(0)

    def draw(self, surface, show_trail=False, show_sensors=True):
        if not self.alive and not show_trail:
            return

        if show_trail and len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                alpha = int(180 * (i / len(self.trail)))
                color = (0, max(40, min(255, 60 + alpha // 3)), max(80, min(255, 120 + alpha // 2)))
                pygame.draw.line(surface, color, self.trail[i - 1], self.trail[i], 1)

        if not self.alive:
            return

        if show_sensors:
            for i, (ex, ey) in enumerate(self.sensor_endpoints):
                intensity = int(255 * (1 - self.sensor_readings[i]))
                color = (255, max(0, 255 - intensity * 2), 0)
                pygame.draw.line(surface, color, (int(self.x), int(self.y)), (int(ex), int(ey)), 2)
                pygame.draw.circle(surface, SENSOR_COLOR, (int(ex), int(ey)), 4)

        vertices = self.get_vertices()
        pygame.draw.polygon(surface, CAR_COLOR, vertices)
        pygame.draw.polygon(surface, (0, 220, 255), vertices, 2)

        rad = math.radians(self.angle)
        front_x = self.x + math.sin(rad) * (CAR_HEIGHT / 2 + 4)
        front_y = self.y - math.cos(rad) * (CAR_HEIGHT / 2 + 4)
        pygame.draw.circle(surface, (255, 255, 255), (int(front_x), int(front_y)), 3)


class Button:
    def __init__(self, text, width=None, color=BUTTON_COLOR, active_color=BUTTON_ACTIVE):
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.text = text
        self.base_text = text
        self.color = color
        self.active_color = active_color
        self.hover = False
        self.active = False
        self.width = width

    def set_pos(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, surface, font):
        ac = self.active_color if self.active else (BUTTON_HOVER if self.hover else self.color)
        pygame.draw.rect(surface, ac, self.rect, border_radius=5)
        pygame.draw.rect(surface, (80, 80, 80), self.rect, 1, border_radius=5)
        text_surf = font.render(self.text, True, UI_TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class StatsTracker:
    def __init__(self, max_history=200):
        self.max_history = max_history
        self.gen_rewards = []
        self.gen_distances = []
        self.best_distance = 0

    def record(self, gen, reward, distance):
        self.gen_rewards.append((gen, reward))
        self.gen_distances.append((gen, distance))
        if distance > self.best_distance:
            self.best_distance = distance
        if len(self.gen_rewards) > self.max_history:
            self.gen_rewards.pop(0)
            self.gen_distances.pop(0)

    def draw(self, surface, x, y, w, h):
        pygame.draw.rect(surface, (15, 15, 15), (x, y, w, h))
        pygame.draw.rect(surface, (50, 50, 50), (x, y, w, h), 1)

        if len(self.gen_rewards) < 2:
            font = pygame.font.SysFont("Consolas", 12)
            t = font.render("Training...", True, (80, 80, 80))
            surface.blit(t, (x + w // 2 - t.get_width() // 2, y + h // 2 - 6))
            return

        font = pygame.font.SysFont("Consolas", 10)
        label = font.render("Distance", True, (100, 180, 100))
        surface.blit(label, (x + 4, y + 2))

        margin = 14
        graph_h = h - margin - 4
        graph_w = w - 8

        distances = [d for _, d in self.gen_distances]
        max_d = max(max(distances), 1)

        points = []
        for i, (gen, dist) in enumerate(self.gen_distances):
            px = x + 4 + int(i / max(len(self.gen_distances) - 1, 1) * graph_w)
            py = y + margin + graph_h - int((dist / max_d) * graph_h)
            points.append((px, py))

        if len(points) > 1:
            pygame.draw.lines(surface, (80, 200, 80), False, points, 1)


class Game:
    def __init__(self):
        pygame.init()
        self.width = DEFAULT_WIDTH
        self.height = DEFAULT_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Self-Driving Car AI - Q-Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 13)
        self.title_font = pygame.font.SysFont("Consolas", 16, bold=True)
        self.small_font = pygame.font.SysFont("Consolas", 11)
        self.tiny_font = pygame.font.SysFont("Consolas", 10)

        self.mode = DRAW_MODE
        self.tool = TOOL_BRUSH
        self.wall_surface = pygame.Surface((self.width, self.height))
        self.wall_surface.fill((0, 0, 0))
        self.drawing = False
        self.last_mouse_pos = None
        self.brush_size = 6
        self.eraser_size = 20

        self.car = None
        self.agent = QLearningAgent()
        self.current_state = None
        self.episode_reward = 0
        self.episode_count = 0
        self.generation = 0
        self.auto_train = False
        self.train_speed = 1
        self.show_trail = True
        self.show_sensors = True
        self.show_stats = True

        self.stats = StatsTracker()

        self.car_start_x = self.width // 2
        self.car_start_y = self.height // 2
        self.car_start_angle = 0

        self.is_fullscreen = False
        self.notification = ""
        self.notification_timer = 0

        os.makedirs(SAVE_DIR, exist_ok=True)

        self._create_buttons()
        self.layout_buttons()

    def _create_buttons(self):
        self.btn_brush = Button("Pen", color=BUTTON_TOOL, active_color=BUTTON_TOOL_ACTIVE)
        self.btn_eraser = Button("Eraser", color=BUTTON_TOOL, active_color=BUTTON_TOOL_ACTIVE)
        self.btn_place_car = Button("Car", color=BUTTON_TOOL, active_color=BUTTON_TOOL_ACTIVE)
        self.btn_size_dn = Button("-", width=26)
        self.btn_size_up = Button("+", width=26)
        self.btn_train = Button("Train", active_color=BUTTON_ACTIVE)
        self.btn_run = Button("Run AI", active_color=BUTTON_ACTIVE)
        self.btn_auto_train = Button("Auto Train", active_color=BUTTON_ACTIVE_RED)
        self.btn_speed = Button("1x", width=36)
        self.btn_save_model = Button("Save", width=42)
        self.btn_load_model = Button("Load", width=42)
        self.btn_save_track = Button("Trk+", width=36)
        self.btn_load_track = Button("Trk-", width=36)
        self.btn_clear = Button("Clear", width=42, active_color=BUTTON_ACTIVE_RED)
        self.btn_reset_q = Button("Reset Q", width=52, active_color=BUTTON_ACTIVE_RED)
        self.btn_trail = Button("Trail", width=42, color=BUTTON_TOOL, active_color=BUTTON_TOOL_ACTIVE)
        self.btn_sensors = Button("Sens", width=42, color=BUTTON_TOOL, active_color=BUTTON_TOOL_ACTIVE)
        self.btn_stats = Button("Stats", width=42, color=BUTTON_TOOL, active_color=BUTTON_TOOL_ACTIVE)
        self.btn_fullscreen = Button("\u25A1", width=28)

        self.btn_brush.active = True
        self.btn_trail.active = True
        self.btn_sensors.active = True
        self.btn_stats.active = True

        self.buttons = [
            self.btn_brush, self.btn_eraser, self.btn_place_car,
            self.btn_size_dn, self.btn_size_up,
            self.btn_train, self.btn_run, self.btn_auto_train, self.btn_speed,
            self.btn_save_model, self.btn_load_model,
            self.btn_save_track, self.btn_load_track,
            self.btn_clear, self.btn_reset_q,
            self.btn_trail, self.btn_sensors, self.btn_stats,
            self.btn_fullscreen,
        ]

    def layout_buttons(self):
        pad = 4
        btn_h = min(28, TOP_BAR_H - 12)
        btn_y = (TOP_BAR_H - btn_h) // 2
        x = pad

        def place(btn, w=None):
            nonlocal x
            bw = w or btn.width or 50
            btn.set_pos(x, btn_y, bw, btn_h)
            x += bw + pad

        def sep():
            nonlocal x
            x += pad

        place(self.btn_brush, 40)
        place(self.btn_eraser, 52)
        place(self.btn_place_car, 40)
        sep()
        place(self.btn_size_dn)
        place(self.btn_size_up)
        sep()
        place(self.btn_train, 48)
        place(self.btn_run, 52)
        place(self.btn_auto_train, 72)
        place(self.btn_speed)
        sep()
        place(self.btn_save_model)
        place(self.btn_load_model)
        place(self.btn_save_track)
        place(self.btn_load_track)
        sep()
        place(self.btn_clear)
        place(self.btn_reset_q)
        sep()
        place(self.btn_trail)
        place(self.btn_sensors)
        place(self.btn_stats)

        place(self.btn_fullscreen)

    def notify(self, msg, duration=120):
        self.notification = msg
        self.notification_timer = duration

    def handle_resize(self, w, h):
        w = max(w, MIN_WIDTH)
        h = max(h, MIN_HEIGHT)
        old_w, old_h = self.width, self.height
        old_wall = self.wall_surface.copy()

        self.width = w
        self.height = h
        self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)

        new_wall = pygame.Surface((w, h))
        new_wall.fill((0, 0, 0))
        scale_x = w / old_w
        scale_y = h / old_h
        scaled = pygame.transform.scale(old_wall, (w, h))
        new_wall.blit(scaled, (0, 0))
        self.wall_surface = new_wall

        self.car_start_x = int(self.car_start_x * scale_x)
        self.car_start_y = int(self.car_start_y * scale_y)

        if self.car:
            self.car.x = int(self.car.x * scale_x)
            self.car.y = int(self.car.y * scale_y)
            self.car.bounds_w = w
            self.car.bounds_h = h
            self.car.update_sensors(self.wall_surface)

        self.layout_buttons()

    def spawn_car(self):
        self.car = Car(self.car_start_x, self.car_start_y, self.car_start_angle, self.width, self.height)
        self.car.update_sensors(self.wall_surface)
        self.current_state = self.agent.discretize(self.car.sensor_readings)
        self.episode_reward = 0
        self.episode_count += 1

    def set_mode(self, mode):
        self.mode = mode
        self.btn_train.active = (mode == TRAIN_MODE)
        self.btn_run.active = (mode == RUN_MODE)

        if mode == TRAIN_MODE:
            self.spawn_car()
            self.auto_train = True
            self.btn_auto_train.text = "Stop"
            self.btn_auto_train.active = True
        elif mode == RUN_MODE:
            self.spawn_car()
            self.auto_train = False
            self.btn_auto_train.active = False
            self.btn_auto_train.text = "Auto Train"
        else:
            self.auto_train = False
            self.btn_auto_train.active = False
            self.btn_auto_train.text = "Auto Train"

    def set_tool(self, tool):
        self.tool = tool
        self.btn_brush.active = (tool == TOOL_BRUSH)
        self.btn_eraser.active = (tool == TOOL_ERASER)
        self.btn_place_car.active = (tool == TOOL_CAR)

    def paint(self, pos, erase=False):
        x, y = pos
        size = self.eraser_size if erase else self.brush_size
        color = (0, 0, 0) if erase else WALL_COLOR
        if self.last_mouse_pos:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            dist = max(1, math.sqrt(dx * dx + dy * dy))
            steps = int(dist / 2)
            for i in range(steps + 1):
                t = i / max(steps, 1)
                ix = int(self.last_mouse_pos[0] + dx * t)
                iy = int(self.last_mouse_pos[1] + dy * t)
                pygame.draw.circle(self.wall_surface, color, (ix, iy), size)
        else:
            pygame.draw.circle(self.wall_surface, color, (x, y), size)
        self.last_mouse_pos = pos

    def draw_grid(self, surface):
        for x in range(0, self.width, 40):
            pygame.draw.line(surface, GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, 40):
            pygame.draw.line(surface, GRID_COLOR, (0, y), (self.width, y))

    def step_training(self):
        if not self.car or not self.car.alive:
            self.stats.record(self.generation, self.episode_reward, self.stats.best_distance if self.car else 0)
            self.spawn_car()
            self.generation += 1
            return

        action = self.agent.choose_action(self.current_state)
        self.car.move(action)
        self.car.update_sensors(self.wall_surface)

        collision = self.car.check_collision(self.wall_surface)

        if collision:
            reward = -200
            self.car.alive = False
        else:
            min_sensor = min(self.car.sensor_readings)
            if min_sensor < 0.2:
                reward = -10
            elif min_sensor < 0.4:
                reward = -2
            else:
                reward = 3
            if action == ACTION_STRAIGHT:
                reward += 1

        next_state = self.agent.discretize(self.car.sensor_readings)
        self.agent.learn(self.current_state, action, reward, next_state)
        self.current_state = next_state
        self.episode_reward += reward

        if collision:
            if self.car.distance_traveled > self.stats.best_distance:
                self.stats.best_distance = self.car.distance_traveled
            self.stats.record(self.generation, self.episode_reward, self.car.distance_traveled)
            self.spawn_car()
            self.generation += 1

        self.agent.decay_epsilon()

    def step_running(self):
        if not self.car or not self.car.alive:
            self.spawn_car()
            return

        state = self.agent.discretize(self.car.sensor_readings)
        action = int(np.argmax(self.agent.q_table[state[0], state[1], state[2]]))
        self.car.move(action)
        self.car.update_sensors(self.wall_surface)

        if self.car.check_collision(self.wall_surface):
            self.car.alive = False

    def draw_ui(self):
        pygame.draw.rect(self.screen, UI_BG_COLOR, (0, 0, self.width, TOP_BAR_H))
        pygame.draw.line(self.screen, SEPARATOR_COLOR, (0, TOP_BAR_H), (self.width, TOP_BAR_H))

        for btn in self.buttons:
            btn.draw(self.screen, self.font)

        size = self.eraser_size if self.tool == TOOL_ERASER else self.brush_size
        size_text = self.tiny_font.render(f"{size}px", True, (140, 140, 140))
        sx = self.btn_size_dn.rect.left
        self.screen.blit(size_text, (sx - size_text.get_width() - 2, (TOP_BAR_H - size_text.get_height()) // 2))

        right_x = self.btn_fullscreen.rect.left - 10
        if right_x > self.btn_stats.rect.right + 20:
            mode_names = {DRAW_MODE: "DRAW", TRAIN_MODE: "TRAIN", RUN_MODE: "RUN"}
            mode_colors = {DRAW_MODE: (100, 200, 255), TRAIN_MODE: (255, 200, 50), RUN_MODE: (50, 255, 100)}
            mt = self.small_font.render(mode_names[self.mode], True, mode_colors[self.mode])
            self.screen.blit(mt, (right_x - mt.get_width(), (TOP_BAR_H - mt.get_height()) // 2))

            stat_str = f"Gen:{self.generation} Best:{self.stats.best_distance:.0f} \u03b5:{self.agent.epsilon:.3f}"
            st = self.tiny_font.render(stat_str, True, (140, 140, 140))
            self.screen.blit(st, (right_x - mt.get_width() - st.get_width() - 6, (TOP_BAR_H - st.get_height()) // 2))

    def draw_bottom_bar(self):
        bar_y = self.height - BOTTOM_BAR_H
        pygame.draw.rect(self.screen, (15, 15, 15), (0, bar_y, self.width, BOTTOM_BAR_H))
        pygame.draw.line(self.screen, SEPARATOR_COLOR, (0, bar_y), (self.width, bar_y))

        if self.car:
            sensor_text = " | ".join([f"S{i + 1}:{self.car.sensor_readings[i]:.2f}" for i in range(NUM_SENSORS)])
            s = self.tiny_font.render(f"Sensors: {sensor_text}", True, (160, 160, 160))
            self.screen.blit(s, (10, bar_y + 4))

            alive_color = (50, 255, 50) if self.car.alive else (255, 50, 50)
            alive_text = self.font.render(f"{'ALIVE' if self.car.alive else 'CRASHED'}", True, alive_color)
            self.screen.blit(alive_text, (10, bar_y + 20))

            dist_text = self.tiny_font.render(f"Dist:{self.car.distance_traveled:.0f} Steps:{self.car.steps_alive}", True, (160, 160, 160))
            self.screen.blit(dist_text, (90, bar_y + 22))

            if self.car.alive:
                indicator_x = max(10, self.width - 200)
                indicator_y = bar_y + 4
                for i in range(NUM_SENSORS):
                    val = self.car.sensor_readings[i]
                    r = int(255 * (1 - val))
                    g = int(255 * val)
                    bar_w = 45
                    bx = indicator_x + i * (bar_w + 6)
                    pygame.draw.rect(self.screen, (40, 40, 40), (bx, indicator_y, bar_w, 8))
                    pygame.draw.rect(self.screen, (r, g, 0), (bx, indicator_y, int(bar_w * val), 8))
                    label = self.tiny_font.render(f"S{i + 1}:{val:.2f}", True, (140, 140, 140))
                    self.screen.blit(label, (bx, indicator_y + 10))

        tool_names = {TOOL_BRUSH: "Pen", TOOL_ERASER: "Eraser", TOOL_CAR: "Place Car"}
        tool_text = self.tiny_font.render(f"Tool: {tool_names[self.tool]}", True, (120, 120, 120))
        self.screen.blit(tool_text, (10, bar_y + 40))

        controls = "LMB:Use Tool  Scroll:Angle  C:Clear  Space:Pause  F11:Fullscreen  Ctrl+S:Save  Ctrl+L:Load"
        c_text = self.tiny_font.render(controls, True, (80, 80, 80))
        c_text_x = max(tool_text.get_width() + 20, self.width - c_text.get_width() - 10)
        self.screen.blit(c_text, (c_text_x, bar_y + 40))

    def draw_start_marker(self):
        x, y = int(self.car_start_x), int(self.car_start_y)
        rad = math.radians(self.car_start_angle)
        hw = CAR_WIDTH / 2
        hh = CAR_HEIGHT / 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        vertices = []
        for cx, cy in corners:
            rx = cx * cos_a - cy * sin_a + x
            ry = cx * sin_a + cy * cos_a + y
            vertices.append((rx, ry))
        pygame.draw.polygon(self.screen, (0, 60, 0), vertices)
        pygame.draw.polygon(self.screen, (0, 160, 0), vertices, 2)
        front_x = x + math.sin(rad) * (hh + 6)
        front_y = y - math.cos(rad) * (hh + 6)
        pygame.draw.circle(self.screen, (0, 200, 0), (int(front_x), int(front_y)), 4)
        label = self.tiny_font.render(f"{self.car_start_angle}\u00b0", True, (0, 180, 0))
        self.screen.blit(label, (x + 16, y - 14))

    def draw_cursor_preview(self):
        if self.mode != DRAW_MODE:
            return
        mx, my = pygame.mouse.get_pos()
        if my <= TOP_BAR_H or my >= self.height - BOTTOM_BAR_H:
            return

        if self.tool == TOOL_CAR:
            rad = math.radians(self.car_start_angle)
            hw = CAR_WIDTH / 2
            hh = CAR_HEIGHT / 2
            corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            vertices = []
            for cx, cy in corners:
                rx = cx * cos_a - cy * sin_a + mx
                ry = cx * sin_a + cy * cos_a + my
                vertices.append((rx, ry))
            preview = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.polygon(preview, (0, 180, 255, 80), vertices)
            pygame.draw.polygon(preview, (0, 220, 255, 180), vertices, 2)
            self.screen.blit(preview, (0, 0))
        else:
            size = self.eraser_size if self.tool == TOOL_ERASER else self.brush_size
            color = (255, 100, 100, 60) if self.tool == TOOL_ERASER else (200, 200, 200, 40)
            preview = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            pygame.draw.circle(preview, color, (size, size), size)
            pygame.draw.circle(preview, (180, 180, 180, 120), (size, size), size, 1)
            self.screen.blit(preview, (mx - size, my - size))

    def draw_notification(self):
        if self.notification_timer <= 0:
            return
        self.notification_timer -= 1
        alpha = min(255, self.notification_timer * 4)
        surf = self.font.render(self.notification, True, (255, 255, 200))
        bg = pygame.Surface((surf.get_width() + 20, surf.get_height() + 10), pygame.SRCALPHA)
        bg.fill((0, 0, 0, min(180, alpha)))
        x = self.width // 2 - bg.get_width() // 2
        y = TOP_BAR_H + 10
        self.screen.blit(bg, (x, y))
        self.screen.blit(surf, (x + 10, y + 5))

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            info = pygame.display.Info()
            self.handle_resize(info.current_w, info.current_h)
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.width, self.height = self.screen.get_size()
            self.layout_buttons()
        else:
            self.handle_resize(DEFAULT_WIDTH, DEFAULT_HEIGHT)
            self.screen = pygame.display.set_mode((DEFAULT_WIDTH, DEFAULT_HEIGHT), pygame.RESIZABLE)
            self.width = DEFAULT_WIDTH
            self.height = DEFAULT_HEIGHT
            self.layout_buttons()

    def save_model(self):
        filepath = os.path.join(SAVE_DIR, "model_latest.pkl")
        self.agent.save(filepath)
        self.notify(f"Model saved to {filepath}")

    def load_model(self):
        filepath = os.path.join(SAVE_DIR, "model_latest.pkl")
        if not os.path.exists(filepath):
            self.notify("No saved model found")
            return
        try:
            self.agent.load(filepath)
            self.notify(f"Model loaded (eps={self.agent.epsilon:.3f})")
        except Exception as e:
            self.notify(f"Load failed: {e}")

    def save_track(self):
        filepath = os.path.join(SAVE_DIR, "track_latest.png")
        pygame.image.save(self.wall_surface, filepath)
        self.notify(f"Track saved to {filepath}")

    def load_track(self):
        filepath = os.path.join(SAVE_DIR, "track_latest.png")
        if not os.path.exists(filepath):
            self.notify("No saved track found")
            return
        try:
            loaded = pygame.image.load(filepath)
            self.wall_surface = pygame.Surface((self.width, self.height))
            self.wall_surface.fill((0, 0, 0))
            self.wall_surface.blit(loaded, (0, 0))
            self.notify("Track loaded")
        except Exception as e:
            self.notify(f"Track load failed: {e}")

    def handle_button_click(self, btn):
        if btn == self.btn_brush:
            self.set_tool(TOOL_BRUSH)
        elif btn == self.btn_eraser:
            self.set_tool(TOOL_ERASER)
        elif btn == self.btn_place_car:
            self.set_tool(TOOL_CAR)
        elif btn == self.btn_size_dn:
            if self.tool == TOOL_ERASER:
                self.eraser_size = max(4, self.eraser_size - 2)
            else:
                self.brush_size = max(2, self.brush_size - 1)
        elif btn == self.btn_size_up:
            if self.tool == TOOL_ERASER:
                self.eraser_size = min(80, self.eraser_size + 2)
            else:
                self.brush_size = min(40, self.brush_size + 1)
        elif btn == self.btn_train:
            self.set_mode(TRAIN_MODE)
        elif btn == self.btn_run:
            self.agent.epsilon = 0
            self.set_mode(RUN_MODE)
        elif btn == self.btn_auto_train:
            if self.auto_train:
                self.auto_train = False
                self.btn_auto_train.active = False
                self.btn_auto_train.text = "Auto Train"
            else:
                if self.mode == DRAW_MODE:
                    self.set_mode(TRAIN_MODE)
                else:
                    self.auto_train = True
                    self.btn_auto_train.active = True
                    self.btn_auto_train.text = "Stop"
        elif btn == self.btn_speed:
            speeds = [1, 5, 20, 50, 100]
            idx = speeds.index(self.train_speed) if self.train_speed in speeds else 0
            self.train_speed = speeds[(idx + 1) % len(speeds)]
            self.btn_speed.text = f"{self.train_speed}x"
        elif btn == self.btn_save_model:
            self.save_model()
        elif btn == self.btn_load_model:
            self.load_model()
        elif btn == self.btn_save_track:
            self.save_track()
        elif btn == self.btn_load_track:
            self.load_track()
        elif btn == self.btn_clear:
            self.wall_surface.fill((0, 0, 0))
            self.notify("Canvas cleared")
        elif btn == self.btn_reset_q:
            self.agent = QLearningAgent()
            self.generation = 0
            self.episode_count = 0
            self.stats = StatsTracker()
            self.notify("Q-table reset")
        elif btn == self.btn_trail:
            self.show_trail = not self.show_trail
            self.btn_trail.active = self.show_trail
        elif btn == self.btn_sensors:
            self.show_sensors = not self.show_sensors
            self.btn_sensors.active = self.show_sensors
        elif btn == self.btn_stats:
            self.show_stats = not self.show_stats
            self.btn_stats.active = self.show_stats
        elif btn == self.btn_fullscreen:
            self.toggle_fullscreen()

    def run(self):
        running = True
        paused = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.VIDEORESIZE and not self.is_fullscreen:
                    self.handle_resize(event.w, event.h)

                if event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    if event.key == pygame.K_ESCAPE:
                        if self.is_fullscreen:
                            self.toggle_fullscreen()
                        else:
                            running = False
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    if event.key == pygame.K_c:
                        self.wall_surface.fill((0, 0, 0))
                        self.notify("Canvas cleared")
                    if event.key == pygame.K_F11:
                        self.toggle_fullscreen()
                    if event.key == pygame.K_s and mods & pygame.KMOD_CTRL:
                        self.save_model()
                    if event.key == pygame.K_l and mods & pygame.KMOD_CTRL:
                        self.load_model()
                    if event.key == pygame.K_b:
                        self.set_tool(TOOL_BRUSH)
                    if event.key == pygame.K_e:
                        self.set_tool(TOOL_ERASER)
                    if event.key == pygame.K_p:
                        self.set_tool(TOOL_CAR)
                    if event.key == pygame.K_t:
                        self.show_trail = not self.show_trail
                        self.btn_trail.active = self.show_trail
                    if event.key == pygame.K_LEFTBRACKET:
                        if self.tool == TOOL_ERASER:
                            self.eraser_size = max(4, self.eraser_size - 2)
                        else:
                            self.brush_size = max(2, self.brush_size - 1)
                    if event.key == pygame.K_RIGHTBRACKET:
                        if self.tool == TOOL_ERASER:
                            self.eraser_size = min(80, self.eraser_size + 2)
                        else:
                            self.brush_size = min(40, self.brush_size + 1)

                for btn in self.buttons:
                    if btn.handle_event(event):
                        self.handle_button_click(btn)

                if self.mode == DRAW_MODE:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if event.button == 1 and event.pos[1] > TOP_BAR_H and event.pos[1] < self.height - BOTTOM_BAR_H:
                            if self.tool == TOOL_CAR:
                                self.car_start_x = event.pos[0]
                                self.car_start_y = event.pos[1]
                                self.notify(f"Car placed at ({event.pos[0]}, {event.pos[1]})")
                            else:
                                self.drawing = True
                                self.last_mouse_pos = None
                    if event.type == pygame.MOUSEBUTTONUP:
                        if event.button == 1:
                            self.drawing = False
                            self.last_mouse_pos = None
                    if event.type == pygame.MOUSEWHEEL:
                        self.car_start_angle = (self.car_start_angle - event.y * 15) % 360

            if not paused:
                if self.mode == DRAW_MODE and self.drawing and self.tool in (TOOL_BRUSH, TOOL_ERASER):
                    mouse_pos = pygame.mouse.get_pos()
                    if mouse_pos[1] > TOP_BAR_H and mouse_pos[1] < self.height - BOTTOM_BAR_H:
                        erase = (self.tool == TOOL_ERASER)
                        self.paint(mouse_pos, erase=erase)

                if self.mode == TRAIN_MODE and self.auto_train:
                    for _ in range(self.train_speed):
                        self.step_training()

                if self.mode == RUN_MODE:
                    self.step_running()

            self.screen.fill(CANVAS_COLOR)
            self.draw_grid(self.screen)
            self.screen.blit(self.wall_surface, (0, 0))
            self.draw_start_marker()

            if self.car and self.mode != DRAW_MODE:
                self.car.draw(self.screen, show_trail=self.show_trail, show_sensors=self.show_sensors)

            self.draw_cursor_preview()
            self.draw_ui()
            self.draw_bottom_bar()

            if self.show_stats and self.mode != DRAW_MODE:
                stats_w = min(180, self.width // 4)
                stats_h = min(80, self.height // 6)
                stats_x = self.width - stats_w - 10
                stats_y = TOP_BAR_H + 10
                self.stats.draw(self.screen, stats_x, stats_y, stats_w, stats_h)

            self.draw_notification()

            if paused:
                overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 80))
                self.screen.blit(overlay, (0, 0))
                pause_text = self.title_font.render("PAUSED", True, (255, 255, 0))
                self.screen.blit(pause_text, (self.width // 2 - pause_text.get_width() // 2, self.height // 2 - 10))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    game = Game()
    game.run()
