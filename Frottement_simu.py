import pygame
import random
import sys
import json  # <--- Ajout indispensable
import os    # <--- Pour vérifier la présence du fichier
from typing import List, Tuple, Optional, Dict, Any # <--- Ajout Dict, Any
from dataclasses import dataclass

# --- Configuration & Constantes ---
WIDTH, HEIGHT = 1920, 1080
FPS = 60
GRAVITY_ACCELERATION = 9.81  # m/s^2
PIXELS_PER_METER = 200.0  # Échelle arbitraire pour la visualisation


# Positionnement des dessins 
INFO_X = 1500 # position horizontale des infos à droite

table_y = 900 # position verticale du sol de la simulation


# Configuration du Graphique
GRAPH_ORIGIN_X, GRAPH_ORIGIN_Y = 100, 600
GRAPH_WIDTH, GRAPH_HEIGHT = 800, 550

GRAPH_SCALE_X = 20.0  # Pixels par Newton (Axe X: Traction)
GRAPH_SCALE_Y = 50.0  # Pixels par Newton (Axe Y: Frottement)
GRID_STEP_X = 2.0     # Une ligne verticale tous les 5 Newtons
GRID_STEP_Y = 1.0     # Une ligne horizontale tous les 1 Newton
COLOR_GRID = (200, 200, 200)

# Palette de couleurs (RGB)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (153, 0, 0)
COLOR_BLUE = (0, 0, 153)
COLOR_TABLE_BODY = (133, 186, 230)
COLOR_TABLE_TOP = (102, 102, 102)
COLOR_STRING = (251, 241, 82)
COLOR_GREEN = (0, 150, 0)


@dataclass
class Block:
    """Représente l'objet physique en mouvement."""
    mass_kg: float
    mu_static: float    # Coefficient de frottement statique propre au couple Bloc/Surface
    mu_kinetic: float   # Coefficient de frottement cinétique propre au couple Bloc/Surface
    name: str           # Nom du matériau (pour affichage)

@dataclass
class PhysicsState:
    """
    State Pattern: contient l'état instantané de la simulation.
    L'objet 'block' encapsule maintenant les propriétés intrinsèques.
    """
    block: Block        # <-- Composition: Le state possède un Block
    pos_x: float
    velocity: float
    applied_force: float
    is_moving: bool
    surface_index: int  # On garde l'index pour la couleur et la sélection


class Simulation:
    """
    Gestionnaire principal de la simulation de frottement.
    Suit le pattern Game Loop standard.
    """

    def __init__(self) -> None:
        """Initialise pygame, charge les matériaux JSON et l'état initial."""
        pygame.init()
        # MODIFICATION ICI : Ajout de pygame.SCALED pour un meilleur redimensionnement
        # Cela permet à la fenêtre de s'étirer proprement en plein écran
        self.screen: pygame.Surface = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED | pygame.RESIZABLE, vsync=1)
        
        pygame.display.set_caption("Simulation de Frottement - Données Externes")
        self.clock = pygame.time.Clock()
        
        # UI Elements
        self.font = pygame.font.SysFont("Arial", 25)
        self.bold_font = pygame.font.SysFont("Arial", 30, bold=True)

        # Simulation parameters
        self.base_mass_g: float = 100.0
        self.added_mass_g: float = 500.0
        self.starting_x_pixel: float = 200.0
        self.force_ramp_rate: float = 1 

        # --- Chargement des données JSON ---
        self.materials: Dict[str, Any] = self.load_materials("materials.json")
        self.material_names: List[str] = list(self.materials.keys()) # Pour la navigation
        
        # Si le fichier est vide ou erreur, on met une valeur par défaut de sécurité
        if not self.materials:
            print("Erreur critique : Aucun matériau chargé. Utilisation fallback.")
            self.materials = {"Ski-Neige": {"static": 0.85,"kinetic": 0.67,
                                                    "color_block": [30, 30, 30],"color_surface": [50, 50, 50]
                                                    }
                            }
            self.material_names = ["Default"]

        self.state: Optional[PhysicsState] = None
        self.running_sim: bool = False
        self.graph_points: List[Tuple[float, float]] = []
        self.data_points: List[Tuple[float, float]] = []
        # Ajout : On reset les résultats
        self.final_results = None
        self.is_paused: bool = False
        # AJOUT : État de visibilité du panneau d'infos coefficients
        self.show_coeff_info: bool = False

        # Ajout : Variable pour stocker les résultats de fin de course
        self.final_results: Optional[Tuple[float, float]] = None

        # Nouvel état pour suivre le plein écran
        self.is_fullscreen: bool = False
        
        self.reset_simulation()

    def load_materials(self, filename: str) -> Dict[str, Any]:
        """Charge les propriétés des matériaux depuis un fichier JSON."""
        """
        Charge les propriétés des matériaux depuis un fichier JSON.
        Utilise un chemin absolu basé sur l'emplacement du script.
        """
        # 1. Récupère le dossier où se trouve ce script Python
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Construit le chemin complet vers le json
        full_path = os.path.join(script_dir, filename)
        
        if not os.path.exists(full_path):
            print(f"Erreur: Le fichier est introuvable ici : {full_path}")
            # On retourne un dict vide pour déclencher le fallback
            return {}
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Succès : {len(data)} matériaux chargés depuis {filename}.")
                return data
        except json.JSONDecodeError as e:
            print(f"Erreur de syntaxe JSON dans '{full_path}': {e}")
            return {}

    def reset_simulation(self) -> None:
        """Réinitialise les variables d'état et instancie le Block."""
        self.running_sim = False
        self.is_paused = False # Si vous avez ajouté la pause précédemment

        self.graph_points = []
        self.data_points: List[Tuple[float, float]] = []  # Stocke (Force Traction, Frottement)

        self.final_results = None
        # Calcul des valeurs initiales
        total_mass_kg = (self.base_mass_g + self.added_mass_g) / 1000.0
        default_idx = 2  # Index par défaut (Bois)
        
        # Sélection par défaut (le premier ou "Bois" si dispo)
        default_name = "Ski-Neige" if "Ski-Neige" in self.materials else self.material_names[0]
        
        # Récupération des données depuis le dict chargé du JSON
        mat_data = self.materials[default_name]
        
        # Création du Block
        initial_block = Block(
            mass_kg=total_mass_kg,
            mu_static=mat_data["static"],   # Clés correspondant au JSON
            mu_kinetic=mat_data["kinetic"],
            name=default_name
        )
        
        self.state = PhysicsState(
            block=initial_block,
            surface_index=self.material_names.index(default_name),
            pos_x=self.starting_x_pixel,
            velocity=0.0,
            applied_force=0.0,
            is_moving=False
        )
        
        # On s'assure que si on change la masse/surface via les touches, on garde l'index
        # Note: Dans une version plus complexe, on séparerait la config de l'état courant.

    def calculate_forces(self) -> Tuple[float, float, float]:
        """
        Calcule les seuils de force basés sur la configuration actuelle.
        
        Returns:
            Tuple[Force Normale, Frottement Statique Max, Frottement Cinétique]
        """
        if not self.state:
            return 0.0, 0.0, 0.0

        # On utilise les données encapsulées dans l'objet Block
        block = self.state.block 
        
        normal_force = block.mass_kg * GRAVITY_ACCELERATION
        static_limit = block.mu_static * normal_force
        kinetic_friction = block.mu_kinetic * normal_force
        
        return normal_force, static_limit, kinetic_friction

    def start(self) -> None:
        """Lance la boucle de calcul physique."""
        if not self.state:
            self.reset_simulation()
        self.running_sim = True
        self.state.applied_force = 0.0
        self.state.velocity = 0.0
        self.state.is_moving = False
        self.state.pos_x = self.starting_x_pixel
        self.graph_points = []
        self.start_ticks = pygame.time.get_ticks()

    def update_physics(self, dt_seconds: float) -> float:
        """
        Mise à jour de la physique selon la Seconde Loi de Newton.
        
        Args:
            dt_seconds: Temps écoulé depuis la dernière frame (en secondes).
            
        Returns:
            float: La valeur de la force de frottement effective (pour affichage).
        """
        # Si pas d'état, pas de simu, ou EN PAUSE, on ne fait rien
        if not self.state or not self.running_sim or self.is_paused:
            # On retourne la dernière valeur de frottement connue (approximatif pour l'affichage)
            _, _, k_f = self.calculate_forces()
            return k_f if self.state and self.state.is_moving else (self.state.applied_force if self.state else 0.0)

        _, static_limit, kinetic_friction = self.calculate_forces()

        # 1. Augmentation de la force (inchangé)
        if self.state.pos_x < 850:
            self.state.applied_force += self.force_ramp_rate * dt_seconds
        
        # Noise (inchangé)
        noise = random.uniform(-0.05, 0.05) * self.state.applied_force
        
        friction_force = 0.0

        # 2. Logique de mouvement (inchangé)
        if not self.state.is_moving:
            friction_force = min(self.state.applied_force, static_limit)
            if self.state.applied_force > static_limit:
                self.state.is_moving = True
        
        if self.state.is_moving:
            friction_force = kinetic_friction
            net_force = self.state.applied_force - kinetic_friction
            acceleration = net_force / self.state.block.mass_kg
            self.state.velocity += acceleration * dt_seconds
            self.state.pos_x += self.state.velocity * dt_seconds * PIXELS_PER_METER

        # 3. Conditions d'arrêt (inchangé)
        
        if self.state.pos_x > 850 :
            self.running_sim = False
            print("Simulation terminée.")

        # 4. Enregistrement Graphique (MODIFIÉ)
        # Axe X : Force de Traction (N) | Axe Y : Force de Frottement (N)
        self.data_points.append((self.state.applied_force, friction_force))

        # Calcul des PIXELS (comme avant, mais avec les variables d'échelle globales)
        graph_x = GRAPH_ORIGIN_X + (self.state.applied_force * GRAPH_SCALE_X)
        graph_y = GRAPH_ORIGIN_Y - (friction_force * GRAPH_SCALE_Y)
        
        if graph_x <= GRAPH_ORIGIN_X + GRAPH_WIDTH:
            self.graph_points.append((graph_x, graph_y))

        if self.state.pos_x > 850 :
            self.running_sim = False
            
            # --- CALCUL DES RÉSULTATS FINAUX ---
            normal_force = self.state.block.mass_kg * GRAVITY_ACCELERATION
            
            # F_static_max = mu_s * N
            f_static_max = self.state.block.mu_static * normal_force
            
            # F_kinetic = mu_k * N
            f_kinetic_const = self.state.block.mu_kinetic * normal_force
            
            self.final_results = (f_static_max, f_kinetic_const)
            print(f"Fin: Static Max={f_static_max:.2f}N, Kinetic={f_kinetic_const:.2f}N")

        return friction_force

    def draw(self, current_friction: float) -> None:
        """Rendu graphique avec grille et axes mis à jour."""
        self.screen.fill(COLOR_WHITE)
        
        if not self.state:
            return

        # --- 1. Graphique (Haut) ---
        
        # Dessin de la Grille (Nouveau)
        # Lignes Verticales (Force de traction)
        for i in range(0, int(GRAPH_WIDTH / GRAPH_SCALE_X) + 1, int(GRID_STEP_X)):
            x_pos = GRAPH_ORIGIN_X + (i * GRAPH_SCALE_X * int(GRID_STEP_X) / int(GRID_STEP_X)) # Alignement simple
            # Note: Pour simplifier la boucle, on itère sur les Newtons directement
        
        # Meilleure approche boucle grille :
        # Grille X (Traction)
        max_force_x = int(GRAPH_WIDTH / GRAPH_SCALE_X)
        for f in range(0, max_force_x + 1, int(GRID_STEP_X)):
            gx = GRAPH_ORIGIN_X + f * GRAPH_SCALE_X
            pygame.draw.line(self.screen, COLOR_GRID, (gx, GRAPH_ORIGIN_Y), (gx, GRAPH_ORIGIN_Y - GRAPH_HEIGHT), 1)
            # Petit label numérique
            val_surf = self.font.render(str(f), True, COLOR_TABLE_TOP)
            self.screen.blit(val_surf, (gx - 5, GRAPH_ORIGIN_Y + 5))

        # Grille Y (Frottement)
        max_force_y = int(GRAPH_HEIGHT / GRAPH_SCALE_Y)
        for f in range(0, max_force_y + 1, int(GRID_STEP_Y)):
            gy = GRAPH_ORIGIN_Y - f * GRAPH_SCALE_Y
            pygame.draw.line(self.screen, COLOR_GRID, (GRAPH_ORIGIN_X, gy), (GRAPH_ORIGIN_X + GRAPH_WIDTH, gy), 1)
            # Petit label numérique
            if f > 0: # Ne pas écrire 0 deux fois
                val_surf = self.font.render(str(f), True, COLOR_TABLE_TOP)
                self.screen.blit(val_surf, (GRAPH_ORIGIN_X - 30, gy - 10))

        # Axes Principaux
        pygame.draw.line(self.screen, COLOR_BLACK, (GRAPH_ORIGIN_X, GRAPH_ORIGIN_Y - GRAPH_HEIGHT), (GRAPH_ORIGIN_X, GRAPH_ORIGIN_Y), 2)  # Axe Y
        pygame.draw.line(self.screen, COLOR_BLACK, (GRAPH_ORIGIN_X, GRAPH_ORIGIN_Y), (GRAPH_ORIGIN_X + GRAPH_WIDTH, GRAPH_ORIGIN_Y), 2)  # Axe X
        
        # Labels Axes (Modifiés)
        label_y = self.font.render("Frottement (N)", True, COLOR_BLACK)
        # Rotation du texte Y pour faire "pro" (Optionnel, sinon laisser à plat)
        label_y = pygame.transform.rotate(label_y, 90)
        
        label_x = self.font.render("Force de Traction (N)", True, COLOR_BLACK)
        
        self.screen.blit(label_y, (GRAPH_ORIGIN_X - 65, 50))
        self.screen.blit(label_x, (GRAPH_ORIGIN_X + 750, GRAPH_ORIGIN_Y + 40))

        # Tracé de la courbe
        if len(self.graph_points) > 1:
            pygame.draw.lines(self.screen, COLOR_RED, False, self.graph_points, 2)
            
        # --- DESSIN DES RÉSULTATS FINAUX (Lignes et Texte) ---
        if self.final_results:
            f_static, f_kinetic = self.final_results
            
            # Conversion en pixels Y
            y_static = GRAPH_ORIGIN_Y - (f_static * GRAPH_SCALE_Y)
            y_kinetic = GRAPH_ORIGIN_Y - (f_kinetic * GRAPH_SCALE_Y)
            
            # Définition de la zone du graphique
            x_start = GRAPH_ORIGIN_X
            x_end = GRAPH_ORIGIN_X + GRAPH_WIDTH

            # 1. Ligne ROUGE (Frottement Statique Max)
            # On utilise une ligne pointillée simulée ou pleine fine
            pygame.draw.line(self.screen, COLOR_RED, (x_start, y_static), (x_end, y_static), 1)
            label_stat = self.bold_font.render(f"Force Max : {f_static:.2f} N", True, COLOR_RED)
            self.screen.blit(label_stat, (x_end + 10, y_static - 10))

            # 2. Ligne VERTE (Frottement Cinétique Constant)
            pygame.draw.line(self.screen, COLOR_GREEN, (x_start, y_kinetic), (x_end, y_kinetic), 1)
            label_kin = self.bold_font.render(f"Force constante: {f_kinetic:.2f} N", True, COLOR_GREEN)
            self.screen.blit(label_kin, (x_end + 10, y_kinetic - 10))

        # Indicateur de Pause
        if self.is_paused:
            pause_surf = self.bold_font.render("PAUSE", True, COLOR_RED)
            pygame.draw.rect(self.screen, COLOR_WHITE, (WIDTH//2 - 50, HEIGHT//2 - 20, 100, 40))
            pygame.draw.rect(self.screen, COLOR_BLACK, (WIDTH//2 - 50, HEIGHT//2 - 20, 100, 40), 2)
            self.screen.blit(pause_surf, (WIDTH//2 - 35, HEIGHT//2 - 12))

        # --- 2. Scène Physique (Bas) --- 
        # (Le reste de la méthode draw reste identique à votre code original 
        # à partir de `table_y = 500`...)
        
        # Récupération des données visuelles depuis le JSON
        surf_name = self.state.block.name
        mat_data = self.materials.get(surf_name, {})
        
        col_surface = tuple(mat_data.get("color_surface", [128, 128, 128]))
        col_block = tuple(mat_data.get("color_block", [100, 100, 200]))
        
        # Récupération des nouveaux types (avec valeurs par défaut)
        shape_type = mat_data.get("shape", "rect")
        block_name_str = mat_data.get("type_block", "Bloc")
        surface_name_str = mat_data.get("type_surface", "Surface")

        # 1. Dessin de la Table (Surface)
        pygame.draw.rect(self.screen, COLOR_TABLE_BODY, (0, table_y + 20, WIDTH, 100))
        pygame.draw.rect(self.screen, col_surface, (0, table_y, WIDTH, 20))
        
        # Affichage du nom de la surface
        surf_label = self.font.render(surface_name_str, True, (255, 255, 255))
        self.screen.blit(surf_label, (10, table_y + 80)) # En bas à gauche dans le corps de table

        # 2. Dessin du Bloc (DYNAMIQUE SELON LA FORME)
        block_x = int(self.state.pos_x)
        
        # Dimensions par défaut
        block_w, block_h = 100, 60 
        
        # Définition du point d'attache de la corde (par défaut)
        rope_attach_x, rope_attach_y = block_x + block_w, table_y - 30

        if shape_type == "wardrobe":
            # Forme : Armoire (Plus haute que large)
            block_w, block_h = 80, 140
            rect_arm = (block_x, table_y - block_h, block_w, block_h)
            
            # Corps
            pygame.draw.rect(self.screen, col_block, rect_arm)
            pygame.draw.rect(self.screen, COLOR_BLACK, rect_arm, 2)
            # Détails (Portes)
            pygame.draw.line(self.screen, (100, 50, 0), (block_x + block_w//2, table_y - block_h + 5), (block_x + block_w//2, table_y - 5), 2)
            # Poignées
            pygame.draw.circle(self.screen, (255, 215, 0), (block_x + block_w//2 - 5, table_y - block_h//2), 3)
            pygame.draw.circle(self.screen, (255, 215, 0), (block_x + block_w//2 + 5, table_y - block_h//2), 3)
            
            # Le poids additionnel est caché dans l'armoire ou posé dessus (ici, petit carré bleu dessus)
            pygame.draw.rect(self.screen, COLOR_BLUE, (block_x + 20, table_y - block_h - 15, 40, 15))
            
            rope_attach_x, rope_attach_y = block_x + block_w, table_y - block_h // 2

        elif shape_type == "car":
            # Forme : Voiture vue de profil
            car_w = 160  # Longueur totale
            car_h = 70   # Hauteur totale (roues incluses)
            
            # Dimensions relatives
            wheel_radius = 20
            body_bottom_y = table_y - wheel_radius + 5 # Le bas de la caisse (un peu au dessus du sol)
            main_body_h = 35 # Hauteur du bas de caisse
            cabin_h = 30     # Hauteur de l'habitacle
            
            # Coordonnées de base
            front_x = block_x + car_w
            back_x = block_x
            
            # 1. Les Roues (Pneus noirs et jantes grises)
            wheel_centers = [
                (back_x + 35, table_y - wheel_radius), # Roue arrière
                (front_x - 35, table_y - wheel_radius)  # Roue avant
            ]
            for wc in wheel_centers:
                # Pneu
                pygame.draw.circle(self.screen, (40, 40, 40), wc, wheel_radius)
                # Jante
                pygame.draw.circle(self.screen, (180, 180, 180), wc, wheel_radius - 8)
                # Ecrou central
                pygame.draw.circle(self.screen, (80, 80, 80), wc, 5)

            # 2. La Carrosserie Basse (Main body) - Utilise col_block du JSON
            # Forme légèrement profilée à l'avant
            body_points = [
                (back_x, body_bottom_y),                            # Arrière bas
                (front_x - 10, body_bottom_y),                      # Avant bas
                (front_x, body_bottom_y - main_body_h // 2),        # Nez
                (front_x - 20, body_bottom_y - main_body_h),        # Capot avant
                (back_x + 10, body_bottom_y - main_body_h),         # Coffre arrière
            ]
            pygame.draw.polygon(self.screen, col_block, body_points)
            pygame.draw.polygon(self.screen, COLOR_BLACK, body_points, 2) # Contour

            # 3. L'Habitacle (Cabin) - Dessus de la voiture
            cabin_start_x = back_x + 30
            cabin_end_x = front_x - 50
            cabin_points = [
                (cabin_start_x, body_bottom_y - main_body_h),           # Bas pare-brise arrière
                (cabin_end_x, body_bottom_y - main_body_h),             # Bas pare-brise avant
                (cabin_end_x - 15, body_bottom_y - main_body_h - cabin_h), # Haut toit avant
                (cabin_start_x + 20, body_bottom_y - main_body_h - cabin_h) # Haut toit arrière
            ]
            pygame.draw.polygon(self.screen, col_block, cabin_points)
            pygame.draw.polygon(self.screen, COLOR_BLACK, cabin_points, 2)

            # 4. Les Vitres (Bleu clair)
            window_color = (200, 230, 255)
            # Vitre latérale (trapèze simple pour l'effet)
            window_points = [
                (cabin_start_x + 15, body_bottom_y - main_body_h - 5),
                (cabin_end_x - 10, body_bottom_y - main_body_h - 5),
                (cabin_end_x - 18, body_bottom_y - main_body_h - cabin_h + 5),
                (cabin_start_x + 25, body_bottom_y - main_body_h - cabin_h + 5)
            ]
            pygame.draw.polygon(self.screen, window_color, window_points)
            pygame.draw.polygon(self.screen, (100, 150, 200), window_points, 2) # Contour vitre

            # 5. Poids additionnel (Coffre de toit ou aileron bleu)
            pygame.draw.rect(self.screen, COLOR_BLUE, (cabin_start_x + 30, body_bottom_y - main_body_h - cabin_h - 8, 40, 8))

            # 6. Point d'attache de la corde
            # À l'avant, à mi-hauteur de la carrosserie basse pour que ce soit réaliste
            rope_attach_x = front_x
            rope_attach_y = body_bottom_y - (main_body_h // 2)

        elif shape_type == "sled":
            # Forme : Luge
            block_w, block_h = 110, 40
            # Patins
            pygame.draw.lines(self.screen, (150, 150, 150), False, 
                              [(block_x, table_y), (block_x + block_w, table_y), 
                               (block_x + block_w + 10, table_y - 15)], 4)
            # Assise
            pygame.draw.rect(self.screen, col_block, (block_x + 10, table_y - 35, block_w - 20, 15))
            # Piliers
            pygame.draw.line(self.screen, (100, 50, 0), (block_x + 30, table_y), (block_x + 30, table_y - 20), 3)
            pygame.draw.line(self.screen, (100, 50, 0), (block_x + 80, table_y), (block_x + 80, table_y - 20), 3)
            
            # Poids bleu sur l'assise
            pygame.draw.rect(self.screen, COLOR_BLUE, (block_x + 35, table_y - 50, 40, 15))
            
            rope_attach_x, rope_attach_y = block_x + block_w + 10, table_y - 15
            
        elif shape_type == "flat":
            # Forme : Palet (très plat)
            block_w, block_h = 60, 20
            pygame.draw.rect(self.screen, col_block, (block_x, table_y - block_h, block_w, block_h))
            pygame.draw.rect(self.screen, COLOR_BLACK, (block_x, table_y - block_h, block_w, block_h), 2)
            
            # Poids bleu
            pygame.draw.rect(self.screen, COLOR_BLUE, (block_x + 15, table_y - block_h - 10, 30, 10))
            rope_attach_x, rope_attach_y = block_x + block_w, table_y - 10

        else:
            if block_name_str == "Valise":
                # === DESSIN DÉTAILLÉ DE LA VALISE (Cuir Vintage) ===
                
                # Dimensions de la valise
                case_w, case_h = 110, 75
                case_x = block_x
                case_y = table_y - case_h
                
                # Couleurs dérivées de la couleur principale du cuir (col_block)
                # On crée une teinte plus foncée pour les contours et les sangles
                dark_leather = (max(0, col_block[0] - 40), max(0, col_block[1] - 40), max(0, col_block[2] - 40))
                brass_color = (220, 180, 60) # Couleur laiton/doré pour les ferrures

                # 1. Le corps principal de la valise (Cuir)
                main_rect = pygame.Rect(case_x, case_y, case_w, case_h)
                # Fond avec coins arrondis
                pygame.draw.rect(self.screen, col_block, main_rect, border_radius=8)
                # Contour épais plus foncé
                pygame.draw.rect(self.screen, dark_leather, main_rect, 3, border_radius=8)

                # 2. Sangles verticales (Cuir plus foncé)
                strap_width = 14
                strap_x1 = case_x + 20
                strap_x2 = case_x + case_w - 20 - strap_width
                
                pygame.draw.rect(self.screen, dark_leather, (strap_x1, case_y, strap_width, case_h))
                pygame.draw.rect(self.screen, dark_leather, (strap_x2, case_y, strap_width, case_h))
                
                # Boucles de sangle (Laiton)
                buckle_y = case_y + case_h // 2
                pygame.draw.rect(self.screen, brass_color, (strap_x1 - 2, buckle_y, strap_width + 4, 12))
                pygame.draw.rect(self.screen, brass_color, (strap_x2 - 2, buckle_y, strap_width + 4, 12))
                # Ardillon de la boucle (petit trait noir)
                pygame.draw.line(self.screen, COLOR_BLACK, (strap_x1 + strap_width//2, buckle_y), (strap_x1 + strap_width//2, buckle_y+12), 2)
                pygame.draw.line(self.screen, COLOR_BLACK, (strap_x2 + strap_width//2, buckle_y), (strap_x2 + strap_width//2, buckle_y+12), 2)

                # 3. Coins renforcés (Laiton)
                corner_radius = 10
                # Haut-gauche
                pygame.draw.circle(self.screen, brass_color, (case_x + corner_radius, case_y + corner_radius), corner_radius)
                # Haut-droite
                pygame.draw.circle(self.screen, brass_color, (case_x + case_w - corner_radius, case_y + corner_radius), corner_radius)
                # Bas-gauche
                pygame.draw.circle(self.screen, brass_color, (case_x + corner_radius, case_y + case_h - corner_radius), corner_radius)
                # Bas-droite
                pygame.draw.circle(self.screen, brass_color, (case_x + case_w - corner_radius, case_y + case_h - corner_radius), corner_radius)
                # On redessine le contour par dessus pour nettoyer les bords des cercles
                pygame.draw.rect(self.screen, dark_leather, main_rect, 3, border_radius=8)

                # 4. La Poignée (Sur le dessus)
                handle_w = 40
                handle_h_arc = 20
                handle_x = case_x + (case_w - handle_w) // 2
                handle_y_base = case_y + 5
                
                # Attaches de la poignée (Laiton)
                pygame.draw.circle(self.screen, brass_color, (handle_x + 5, handle_y_base), 7)
                pygame.draw.circle(self.screen, brass_color, (handle_x + handle_w - 5, handle_y_base), 7)
                
                # L'arche de la poignée (Cuir foncé)
                handle_rect = pygame.Rect(handle_x, handle_y_base - handle_h_arc, handle_w, handle_h_arc*2)
                pygame.draw.arc(self.screen, dark_leather, handle_rect, 0, 3.14159, 6)

                # Poids additionnel (Petit tag bleu discret sur le côté)
                pygame.draw.rect(self.screen, COLOR_BLUE, (case_x + case_w - 15, case_y + 20, 15, 25))
                
                # Point d'attache : à l'attache droite de la poignée
                rope_attach_x = handle_x + handle_w - 5
                rope_attach_y = handle_y_base

            else:
                # === DESSIN RECTANGLE STANDARD (Fallback) ===
                block_w, block_h = 100, 60
                pygame.draw.rect(self.screen, col_block, (block_x, table_y - block_h, block_w, block_h))
                pygame.draw.rect(self.screen, COLOR_BLACK, (block_x, table_y - block_h, block_w, block_h), 2)
                # Poids additionnel
                pygame.draw.rect(self.screen, COLOR_BLUE, (block_x + 10, table_y - block_h + 10, block_w - 20, block_h - 20))
                rope_attach_x, rope_attach_y = block_x + block_w, table_y - 30


        # --- 3. Corde, Dynamomètre et Main (Horizontalité Stricte) ---
        
        # Le point de référence Y est rope_attach_y (calculé selon la forme du bloc).
        # Tout (Corde, Dyno, Main) doit être aligné sur cet axe Y.

        # A. La Corde (Du bloc au dynamomètre)
        rope_length = 150
        dyno_start_x = rope_attach_x + rope_length
        pygame.draw.line(self.screen, COLOR_STRING, (rope_attach_x, rope_attach_y), (dyno_start_x, rope_attach_y), 3)

        # B. Le Dynamomètre
        dyno_w, dyno_h = 100, 40
        dyno_rect_y = rope_attach_y - (dyno_h // 2)
        
        # 1. Crochet gauche
        pygame.draw.circle(self.screen, (50, 50, 50), (int(dyno_start_x), int(rope_attach_y)), 4)
        
        # 2. Corps du dynamomètre
        pygame.draw.rect(self.screen, (150, 150, 160), (dyno_start_x, dyno_rect_y, dyno_w, dyno_h), border_radius=5)
        pygame.draw.rect(self.screen, COLOR_BLACK, (dyno_start_x, dyno_rect_y, dyno_w, dyno_h), 2, border_radius=5)
        
        # 3. Écran digital et valeur
        screen_margin = 6
        screen_rect = pygame.Rect(dyno_start_x + screen_margin, dyno_rect_y + screen_margin, 
                                  dyno_w - 20 - screen_margin, dyno_h - 2*screen_margin)
        pygame.draw.rect(self.screen, (220, 240, 255), screen_rect, border_radius=3)
        pygame.draw.rect(self.screen, (50, 50, 100), screen_rect, 1, border_radius=3)
        display_force = max(0.0, self.state.applied_force)
        text_surf = self.font.render(f"{display_force:.1f} N", True, COLOR_BLACK)
        text_rect = text_surf.get_rect(center=screen_rect.center)
        self.screen.blit(text_surf, text_rect)

        # 4. Tige et Poignée droite (La main va s'agripper dessus)
        handle_start_x = dyno_start_x + dyno_w
        handle_grip_x = handle_start_x + 40 # L'endroit où les doigts se ferment
        
        # Tige métal horizontale
        pygame.draw.line(self.screen, (100, 100, 100), (handle_start_x, rope_attach_y), (handle_grip_x + 5, rope_attach_y), 5)
        
        # Poignée verticale (Anneau en D sur lequel le poing se ferme)
        # On la dessine avant les doigts pour qu'elle soit "dans" la main
        handle_bar_color = (80, 80, 80)
        pygame.draw.line(self.screen, handle_bar_color, (handle_grip_x, rope_attach_y - 20), (handle_grip_x, rope_attach_y + 20), 6)


        # C. La Main Améliorée et le Bras
        # Définition des couleurs de peau
        skin_color_base = (235, 190, 150)    # Couleur chair de base
        skin_color_dark = (215, 170, 130)    # Pour les jointures/ombres
        skin_outline = (160, 110, 80)        # Contour pour la définition

        center_y = rope_attach_y
        grip_x = handle_grip_x

        # 1. Paume/Base du poing (Derrière la poignée)
        # Un rectangle arrondi pour la masse principale de la main
        fist_rect = pygame.Rect(grip_x - 18, center_y - 28, 30, 56)
        pygame.draw.rect(self.screen, skin_color_base, fist_rect, border_radius=12)

        # 2. Les Doigts (Jointures) qui s'enroulent devant la poignée
        # On dessine 4 cercles superposés pour simuler les doigts fermés
        for i in range(4):
            # Position Y décalée pour chaque doigt (Index au petit doigt)
            finger_y = center_y - 18 + (i * 12)
            # Cercle de base du doigt
            pygame.draw.circle(self.screen, skin_color_base, (int(grip_x + 4), int(finger_y)), 9)
            # Cercle un peu plus foncé pour la jointure (le dessus du doigt)
            pygame.draw.circle(self.screen, skin_color_dark, (int(grip_x + 2), int(finger_y - 3)), 6)
            # Contour fin pour séparer les doigts visuellement
            pygame.draw.circle(self.screen, skin_outline, (int(grip_x + 4), int(finger_y)), 9, 1)

        # 3. Le Pouce (S'enroule par dessous pour verrouiller la prise)
        # Une ligne épaisse inclinée pour le pouce
        thumb_start = (grip_x - 15, center_y + 15)
        thumb_end = (grip_x + 8, center_y + 22)
        pygame.draw.line(self.screen, skin_color_base, thumb_start, thumb_end, 14)
        # L'extrémité du pouce (jointure)
        pygame.draw.circle(self.screen, skin_color_dark, (int(thumb_end[0]), int(thumb_end[1])), 7)
        
        # 4. Poignet et Manche de vêtement
        wrist_start_x = grip_x + 12
        
        # Transition du poignet (forme trapézoïdale)
        pygame.draw.polygon(self.screen, skin_color_base, [
            (wrist_start_x, center_y - 22),
            (wrist_start_x + 25, center_y - 18),
            (wrist_start_x + 25, center_y + 18),
            (wrist_start_x, center_y + 22)
        ])

        # La Manche (Couleur tissu, ex: bleu foncé)
        sleeve_color = (40, 50, 100)
        sleeve_start_x = wrist_start_x + 20
        # Le poignet de la manche (le bord)
        pygame.draw.rect(self.screen, (60, 70, 120), (sleeve_start_x, center_y - 20, 10, 40))
        # Le reste du bras couvert par la manche, sortant de l'écran
        pygame.draw.rect(self.screen, sleeve_color, (sleeve_start_x + 10, center_y - 22, 300, 44))

        # --- 3-BIS. Le Mur (Masque le bout du bras) ---
        
        # On définit une zone "mur" à l'extrême droite de l'écran
        wall_width = 970
        wall_x = WIDTH - wall_width # Ex: 1000 - 80 = 920
        
        # 1. Le Mur lui-même (Gris clair "Labo")
        wall_color = (220, 220, 225)
        # On dessine un grand rectangle de haut en bas
        pygame.draw.rect(self.screen, wall_color, (wall_x, 800, wall_width, HEIGHT))
        # Une ligne verticale pour marquer l'angle du mur
        pygame.draw.line(self.screen, (180, 180, 190), (wall_x, 800), (wall_x, HEIGHT), 3)

        # --- 3. UI / HUD ---

        cursor_y = 30 # Position verticale de départ
        # --- SECTION BLOC ---
        # Titre souligné
        title_block = self.bold_font.render("--- BLOC ---", True, (50, 50, 150)) # Bleu foncé
        self.screen.blit(title_block, (INFO_X, cursor_y))
        cursor_y += 30
        
        # Infos Bloc
        txt_type_block = self.font.render(f"Type : {block_name_str}", True, COLOR_BLACK)
        txt_mass = self.font.render(f"Masse : {(self.state.block.mass_kg * 1000):.0f} g", True, COLOR_BLACK)
        
        self.screen.blit(txt_type_block, (INFO_X, cursor_y))
        cursor_y += 25
        self.screen.blit(txt_mass, (INFO_X, cursor_y))
        cursor_y += 35 # Espace avant la section suivante

        # --- SECTION SURFACE ---
        title_surf = self.bold_font.render("--- SURFACE ---", True, (100, 100, 100)) # Gris foncé
        self.screen.blit(title_surf, (INFO_X, cursor_y))
        cursor_y += 30
        
        # Infos Surface
        txt_type_surf = self.font.render(f"Type : {surface_name_str}", True, COLOR_BLACK)
        self.screen.blit(txt_type_surf, (INFO_X, cursor_y))
        cursor_y += 40 # Espace plus grand

        # --- SECTION SIMULATION (Données dynamiques) ---
        # Ligne de séparation fine
        pygame.draw.line(self.screen, (200, 200, 200), (INFO_X, cursor_y), (INFO_X + 200, cursor_y), 1)
        cursor_y += 15

        force_text = f"Traction : {self.state.applied_force:.2f} N"
        #fric_text = f"Frottement : {current_friction:.2f} N"
        
        self.screen.blit(self.bold_font.render(force_text, True, COLOR_RED), (INFO_X, cursor_y))
        cursor_y += 30
        #self.screen.blit(self.bold_font.render(fric_text, True, COLOR_RED), (INFO_X, cursor_y))
        cursor_y += 35
        
        state_text = "EN MOUVEMENT" if self.state.is_moving else "STATIQUE"
        color_state = (0, 180, 0) if self.state.is_moving else (220, 100, 0)
        self.screen.blit(self.bold_font.render(f"ÉTAT : {state_text}", True, color_state), (INFO_X, cursor_y))


        # --- BOUTONS (Position mise à jour : Y=280 et Y=330) ---
        
        # 1. Bouton Auto-Scale
        btn_scale_rect = pygame.Rect(INFO_X, 330, 120, 40)
        btn_color = (200, 200, 200) if btn_scale_rect.collidepoint(pygame.mouse.get_pos()) else (230, 230, 230)
        pygame.draw.rect(self.screen, btn_color, btn_scale_rect)
        pygame.draw.rect(self.screen, COLOR_BLACK, btn_scale_rect, 2)
        self.screen.blit(self.font.render("Auto-Scale", True, COLOR_BLACK), (btn_scale_rect.x + 15, btn_scale_rect.y + 10))
        
        # Couleur interactive
        btn_color = (200, 200, 200) if btn_scale_rect.collidepoint(pygame.mouse.get_pos()) else (230, 230, 230)
        
        # Dessin du rectangle
        pygame.draw.rect(self.screen, btn_color, btn_scale_rect)
        pygame.draw.rect(self.screen, COLOR_BLACK, btn_scale_rect, 2)
        
        # Centrage du texte "Auto-Scale"
        txt_scale = self.font.render("Auto-Scale", True, COLOR_BLACK)
        txt_scale_rect = txt_scale.get_rect(center=btn_scale_rect.center) # <--- La magie du centrage est ici
        self.screen.blit(txt_scale, txt_scale_rect)

        # 2. Bouton Infos Coefficients
        # On définit le rectangle (Largeur 160)
        btn_info_rect = pygame.Rect(INFO_X, 380, 190, 40)
        
        # Couleur interactive
        btn_info_color = (200, 200, 200) if btn_info_rect.collidepoint(pygame.mouse.get_pos()) else (230, 230, 230)
        
        # Dessin du rectangle
        pygame.draw.rect(self.screen, btn_info_color, btn_info_rect)
        pygame.draw.rect(self.screen, COLOR_BLACK, btn_info_rect, 2)
        
        # Texte dynamique : "Coefficients" au lieu de "Coeffs"
        str_info = "Cacher Coefficients" if self.show_coeff_info else "Voir Coefficients"
        txt_info = self.font.render(str_info, True, COLOR_BLACK)
        
        # Centrage du texte
        txt_info_rect = txt_info.get_rect(center=btn_info_rect.center) # <--- Centrage parfait
        self.screen.blit(txt_info, txt_info_rect)

        # --- Panneau d'informations (Largeur adaptée) ---
        if self.show_coeff_info:
            panel_x, panel_y = INFO_X, 430
            panel_w, panel_h = 230, 80 # Élargi à 200px pour l'esthétique
            
            pygame.draw.rect(self.screen, (245, 245, 220), (panel_x, panel_y, panel_w, panel_h), border_radius=5)
            pygame.draw.rect(self.screen, (100, 100, 0), (panel_x, panel_y, panel_w, panel_h), 2, border_radius=5)
            
            mu_s = self.state.block.mu_static
            mu_k = self.state.block.mu_kinetic
            
            # Utilisation de "Coefficient" en entier ou abréviation propre
            self.screen.blit(self.font.render(f"Coeff. Statique : {mu_s:.3f}", True, (50, 50, 50)), (panel_x + 10, panel_y + 10))
            self.screen.blit(self.font.render(f"Coeff. Cinétique: {mu_k:.3f}", True, (50, 50, 50)), (panel_x + 10, panel_y + 40))

        # Instructions (Centrées en bas ou décalées)
        inst_str = "[ESPACE] Start | [P] Pause | [R] Reset | [S] Couple Surface/Objet | [M] Masse"
        inst_text = self.font.render(inst_str, True, COLOR_BLACK)
        self.screen.blit(inst_text, (150, table_y+120))

        pygame.display.flip()

    def handle_input(self) -> None:
        """Gestion des événements clavier avec support de la Pause."""
        """gestion position souris"""
        mouse_pos = pygame.mouse.get_pos() 
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # --- GESTION SOURIS ---
            if event.type == pygame.MOUSEBUTTONDOWN:
                # 1. Bouton Auto-Scale
                # MODIFIÉ : Largeur passée à 160 pour harmoniser
                btn_scale_rect = pygame.Rect(INFO_X, 330, 160, 40)
                if btn_scale_rect.collidepoint(mouse_pos):
                    self.autoscale_graph()

                # 2. Bouton Infos Coefficients
                # MODIFIÉ : Largeur passée à 160
                btn_info_rect = pygame.Rect(INFO_X, 380, 160, 40)
                if btn_info_rect.collidepoint(mouse_pos):
                    self.show_coeff_info = not self.show_coeff_info
                    print(f"Bouton cliqué ! Affichage infos : {self.show_coeff_info}")

            # --- GESTION CLAVIER ---

            if event.type == pygame.KEYDOWN:
                # Plein Écran (Fullscreen) - Méthode Robuste
                # if event.key == pygame.K_f:
                #     self.is_fullscreen = not self.is_fullscreen
                    
                #     if self.is_fullscreen:
                #         # On passe en FULLSCREEN tout en gardant le SCALED
                #         self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED | pygame.FULLSCREEN | pygame.RESIZABLE, vsync=1)
                #     else:
                #         # On revient en mode FENÊTRÉ standard
                #         self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.SCALED | pygame.RESIZABLE, vsync=1)

                # Toggle Pause
                if event.key == pygame.K_p and self.running_sim:
                    self.is_paused = not self.is_paused

                if event.key == pygame.K_SPACE and not self.running_sim:
                    self.start()
                elif event.key == pygame.K_r:
                    global GRAPH_SCALE_X, GRAPH_SCALE_Y
                    GRAPH_SCALE_X = 20.0
                    GRAPH_SCALE_Y = 50.0
                    self.reset_simulation()
                    self.is_paused = False # Reset pause on restart
                
                # Inputs actifs seulement si reset et non en cours
                if self.state and not self.running_sim:
                    if event.key == pygame.K_s:
                        # Navigation via l'index
                        new_idx = (self.state.surface_index + 1) % len(self.material_names)
                        self.state.surface_index = new_idx
                        
                        new_name = self.material_names[new_idx]
                        mat_data = self.materials[new_name]
                        
                        # Update Block avec les données du JSON
                        self.state.block.name = new_name
                        self.state.block.mu_static = mat_data["static"]
                        self.state.block.mu_kinetic = mat_data["kinetic"]
                        # La couleur n'est pas stockée dans le block, elle est lue dans draw()
                        
                    elif event.key == pygame.K_m:
                        # Changement de masse
                        self.added_mass_g += 100
                        if self.added_mass_g > 1000:
                            self.added_mass_g = 0
                            
                        # MISE A JOUR DE L'OBJET BLOCK
                        new_total_mass = (self.base_mass_g + self.added_mass_g) / 1000.0
                        self.state.block.mass_kg = new_total_mass

    def autoscale_graph(self) -> None:
        """Adapte l'échelle du graphique pour inclure tous les points enregistrés."""
        if not self.data_points:
            return

        # 1. Trouver les maximums dans les données (X: Traction, Y: Frottement)
        max_traction = max(p[0] for p in self.data_points)
        max_friction = max(p[1] for p in self.data_points)

        # 2. Ajouter une marge de 10% pour l'esthétique
        target_max_x = max_traction * 1.1 if max_traction > 0 else 10.0
        target_max_y = max_friction * 1.1 if max_friction > 0 else 10.0

        # 3. Recalculer les échelles (Pixels par Newton)
        # GRAPH_WIDTH et GRAPH_HEIGHT sont définis dans les constantes (ex: 500 et 300)
        global GRAPH_SCALE_X, GRAPH_SCALE_Y

        GRAPH_SCALE_X = GRAPH_WIDTH / target_max_x
        GRAPH_SCALE_Y = GRAPH_HEIGHT / target_max_y # 300 est la hauteur du graph

        # 4. Régénérer les points graphiques (pixels)
        self.regenerate_graph_points()
        print(f"Auto-scale: Max X={target_max_x:.2f}N, Max Y={target_max_y:.2f}N")

    def regenerate_graph_points(self) -> None:
        """Recalcule les coordonnées pixels de tous les points selon l'échelle actuelle."""
        self.graph_points = []
        for traction, friction in self.data_points:
            gx = GRAPH_ORIGIN_X + (traction * GRAPH_SCALE_X)
            gy = GRAPH_ORIGIN_Y - (friction * GRAPH_SCALE_Y)
            self.graph_points.append((gx, gy))

    def run(self) -> None:
        """Boucle principale."""
        while True:
            dt = self.clock.tick(FPS) / 1000.0 # Delta time en secondes
            self.handle_input()
            current_friction = self.update_physics(dt)
            self.draw(current_friction)

if __name__ == "__main__":
    sim = Simulation()
    sim.run()