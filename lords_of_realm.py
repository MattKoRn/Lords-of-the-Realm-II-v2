#!/usr/bin/env python3
"""
Lords of the Realm II - Text Based Game with Neural Network AI
Based on the original 1996 game mechanics from the official manual

AUTHENTIC GAME MECHANICS:
- Labor allocation slider (Agriculture vs Industry)
- Field management (Cattle, Grain, Empty fields)
- Rations system with diet slider (Dairy/Beef/Grain)
- County management with multiple territories
- Merchant trading system
- Castle building with customization
- Seasonal turn-based gameplay
- Real-time battles with tactical control
- Diplomacy with AI nobles (Bishop, Baron, Knight, Countess)

NEURAL NETWORK FEATURES:
- AI learns optimal labor allocation strategies
- Evolves combat tactics through reinforcement learning
- Predicts resource needs and enemy behavior
- Auto-play mode with trainable neural network
- Network mutation and evolution across games

Controls:
- Arrow keys: Navigate menus and sliders
- Enter: Select/confirm
- Escape: Back/exit panel
- Space: End turn / Pause battle
- A: Toggle auto-play (neural network control)
- S: Save game
- L: Load game
- ?: Help
"""

import curses
import time
import json
import os
import math
import random
import threading
import pickle
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Number suffix system for unlimited scaling
NUMBER_SUFFIXES = [
    "", "K", "M", "B", "T", "Qa", "Qi", "Sx", "Sp", "Oc", "No", "Dc",
    "UDc", "DDc", "TDc", "QaDc", "QiDc", "SxDc", "SpDc", "OcDc", "NoDc", "Vg",
    "UVg", "DVg", "TVg", "QaVg", "QiVg", "SxVg", "SpVg", "OcVg", "NoVg", "Tg",
    "UTg", "DTg", "TTg", "QaTg", "QiTg", "SxTg", "SpTg", "OcTg", "NoTg", "Qd",
    "UQd", "DQd", "TQd", "QaQd", "QiQd", "SxQd", "SpQd", "OcQd", "NoQd", "Qt",
    "UQt", "DQt", "TQt", "QaQt", "QiQt", "SxQt", "SpQt", "OcQt", "NoQt", "Sxt",
    "USxt", "DSxt", "TSxt", "QaSxt", "QiSxt", "SxSxt", "SpSxt", "OcSxt", "NoSxt", "Oct",
    "UOct", "DOct", "TOct", "QaOct", "QiOct", "SxOct", "SpOct", "OcOct", "NoOct", "Non"
]

def format_number(num: float) -> str:
    """Format large numbers with suffixes"""
    if num < 1000:
        return f"{int(num)}" if num == int(num) else f"{num:.2f}"
    
    suffix_index = min(int(math.log10(num) / 3), len(NUMBER_SUFFIXES) - 1)
    if suffix_index >= len(NUMBER_SUFFIXES):
        suffix_index = len(NUMBER_SUFFIXES) - 1
    
    scaled = num / (1000 ** suffix_index)
    suffix = NUMBER_SUFFIXES[suffix_index]
    
    if scaled >= 100:
        return f"{scaled:.0f}{suffix}"
    elif scaled >= 10:
        return f"{scaled:.1f}{suffix}"
    else:
        return f"{scaled:.2f}{suffix}"

def parse_number(text: str) -> float:
    """Parse number with suffixes back to float"""
    text = text.strip()
    if not text:
        return 0.0
    
    suffix = ""
    num_part = text
    
    for s in reversed(NUMBER_SUFFIXES):
        if s and text.endswith(s):
            suffix = s
            num_part = text[:-len(s)] if s else text
            break
    
    try:
        value = float(num_part)
        if suffix:
            suffix_index = NUMBER_SUFFIXES.index(suffix)
            value *= (1000 ** suffix_index)
        return value
    except ValueError:
        return 0.0


class NeuralNetwork:
    """Simple neural network for game AI that learns and evolves"""
    
    def __init__(self, input_size: int = 20, hidden_size: int = 32, output_size: int = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with small random values
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.5
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.5
        self.bias2 = np.zeros((1, output_size))
        
        # Evolution tracking
        self.fitness = 0
        self.generation = 0
        self.games_played = 0
        self.wins = 0
        self.total_resources_gathered = 0
        
        # Learning rate
        self.learning_rate = 0.01
        
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the network"""
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        inputs = inputs.reshape(1, -1)
        
        # Hidden layer
        self.z1 = np.dot(inputs, self.weights1) + self.bias1
        self.a1 = self.relu(self.z1)
        
        # Output layer
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        output = self.relu(self.z2)
        
        return output.flatten()
    
    def train(self, inputs: np.ndarray, target: np.ndarray, reward: float):
        """Train the network with reward-based learning"""
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        if isinstance(target, list):
            target = np.array(target)
        
        inputs = inputs.reshape(1, -1)
        target = target.reshape(1, -1)
        
        # Forward pass
        z1 = np.dot(inputs, self.weights1) + self.bias1
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.weights2) + self.bias2
        output = self.relu(z2)
        
        # Calculate error with reward weighting
        error = (target - output) * reward
        self.fitness += reward
        
        # Backward pass
        d_z2 = error * self.relu_derivative(z2)
        d_w2 = np.dot(a1.T, d_z2) * self.learning_rate
        d_b2 = np.sum(d_z2, axis=0, keepdims=True) * self.learning_rate
        
        d_a1 = np.dot(d_z2, self.weights2.T)
        d_z1 = d_a1 * self.relu_derivative(z1)
        d_w1 = np.dot(inputs.T, d_z1) * self.learning_rate
        d_b1 = np.sum(d_z1, axis=0, keepdims=True) * self.learning_rate
        
        # Update weights
        self.weights2 += d_w2
        self.bias2 += d_b2
        self.weights1 += d_w1
        self.bias1 += d_b1
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.5):
        """Mutate weights for evolution"""
        mask1 = np.random.rand(*self.weights1.shape) < mutation_rate
        mask2 = np.random.rand(*self.weights2.shape) < mutation_rate
        
        self.weights1 += mask1 * np.random.randn(*self.weights1.shape) * mutation_strength
        self.weights2 += mask2 * np.random.randn(*self.weights2.shape) * mutation_strength
        
        self.generation += 1
    
    def get_state_vector(self, game: 'Game') -> np.ndarray:
        """Convert game state to input vector for neural network.
        Uses authentic LOTR2 mechanics: labor allocation, fields, rations, etc."""
        state = []
        
        # Resource ratios (normalized) - now includes wood and weapons
        total_resources = sum([
            game.resources.get('food', 1),
            game.resources.get('gold', 1),
            game.resources.get('stone', 1),
            game.resources.get('iron', 1),
            game.resources.get('wood', 1),
            game.resources.get('weapons', 1)
        ])
        
        state.append(game.resources.get('food', 0) / max(total_resources, 1))
        state.append(game.resources.get('gold', 0) / max(total_resources, 1))
        state.append(game.resources.get('stone', 0) / max(total_resources, 1))
        state.append(game.resources.get('iron', 0) / max(total_resources, 1))
        state.append(game.resources.get('wood', 0) / max(total_resources, 1))
        state.append(game.resources.get('weapons', 0) / max(total_resources, 1))
        
        # Population metrics
        state.append(min(game.population / max(game.max_population, 1), 1.0))
        state.append(game.happiness / 100.0)
        
        # Labor allocation (core LOTR2 mechanic)
        state.append(game.labor_agriculture / 100.0)
        state.append(game.labor_industry / 100.0)
        
        # Field assignments
        state.append(game.fields.get('cattle', 0) / max(game.total_fields, 1))
        state.append(game.fields.get('grain', 0) / max(game.total_fields, 1))
        state.append(game.fields.get('empty', 0) / max(game.total_fields, 1))
        
        # Rations system
        ration_values = {'none': 0, 'quarter': 0.2, 'half': 0.4, 'normal': 0.6, 'double': 0.8, 'triple': 1.0}
        state.append(ration_values.get(game.ration_level, 0.5))
        state.append(game.diet_slider / 100.0)
        
        # Health rating
        health_values = {'perfect': 1.0, 'good': 0.8, 'average': 0.6, 'sick': 0.3, 'diseased': 0.1}
        state.append(health_values.get(game.health_rating, 0.5))
        
        # Industry status
        active_count = sum(1 for ind in game.industries.values() if ind['active'])
        state.append(active_count / len(game.industries))
        
        # Military strength
        total_units = sum(game.army.values())
        max_army = 1000  # Default max army size
        state.append(min(total_units / max(max_army, 1), 1.0))
        
        # Diplomacy relations (average with AI nobles)
        avg_relation = sum(n['relation'] for n in game.ai_nobles.values()) / len(game.ai_nobles)
        state.append(avg_relation / 100.0)
        
        # Treasury/tax rate
        tax_values = {'none': 0, 'low': 0.25, 'normal': 0.5, 'high': 0.75, 'excessive': 1.0}
        state.append(tax_values.get(game.tax_rate, 0.5))
        state.append(min(game.treasury / 1000, 1.0))
        
        # Season (one-hot encoded as value)
        season_values = {'spring': 0.25, 'summer': 0.5, 'autumn': 0.75, 'winter': 1.0}
        state.append(season_values.get(game.season, 0.5))
        
        # Combat/siege state
        state.append(1.0 if game.in_combat else 0.0)
        state.append(1.0 if game.in_siege else 0.0)
        
        # Pad or truncate to input_size
        while len(state) < self.input_size:
            state.append(0.0)
        state = state[:self.input_size]
        
        return np.array(state)
    
    def decide_action(self, game: 'Game') -> int:
        """Decide next action based on game state.
        Actions correspond to authentic LOTR2 mechanics."""
        state = self.get_state_vector(game)
        output = self.forward(state)
        
        # Actions for LOTR2:
        # 0: Increase agriculture labor
        # 1: Decrease agriculture labor (increase industry)
        # 2: Activate an industry
        # 3: Deactivate an industry
        # 4: Adjust rations
        # 5: Adjust diet slider
        # 6: Train troops (conscript)
        # 7: Trade with merchant
        # 8: End turn
        # 9: Diplomacy action
        return int(np.argmax(output))
    
    def save(self, filepath: str):
        """Save neural network to file"""
        data = {
            'weights1': self.weights1.tolist(),
            'bias1': self.bias1.tolist(),
            'weights2': self.weights2.tolist(),
            'bias2': self.bias2.tolist(),
            'fitness': self.fitness,
            'generation': self.generation,
            'games_played': self.games_played,
            'wins': self.wins,
            'total_resources_gathered': self.total_resources_gathered,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'NeuralNetwork':
        """Load neural network from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        nn = cls(data['input_size'], data['hidden_size'], data['output_size'])
        nn.weights1 = np.array(data['weights1'])
        nn.bias1 = np.array(data['bias1'])
        nn.weights2 = np.array(data['weights2'])
        nn.bias2 = np.array(data['bias2'])
        nn.fitness = data['fitness']
        nn.generation = data['generation']
        nn.games_played = data['games_played']
        nn.wins = data['wins']
        nn.total_resources_gathered = data['total_resources_gathered']
        
        return nn


class Game:
    """Main game class implementing authentic Lords of the Realm II mechanics
    with Neural Network AI for intelligent automation and learning."""
    
    def __init__(self):
        # === AUTHENTIC LOTR2 MECHANICS ===
        
        # Labor allocation (0-100%, Agriculture vs Industry) - Core mechanic from manual
        self.labor_agriculture = 80  # % of workforce in agriculture
        self.labor_industry = 20     # % of workforce in industry
        
        # Field management system (each county has fields)
        self.fields = {
            'cattle': 3,   # Fields assigned to cattle farming
            'grain': 0,    # Fields assigned to grain farming  
            'empty': 2     # Unassigned fields
        }
        self.total_fields = 5
        
        # Rations system with diet slider (manual page 31-33)
        self.ration_level = 'normal'  # none, quarter, half, normal, double, triple
        self.diet_slider = 100  # 0=all grain, 100=all beef (dairy is automatic)
        self.dairy_produce = 50  # Automatic dairy production from cows
        self.beef_available = 20  # Cows available for slaughter
        self.grain_stored = 0     # Grain sacks stored
        
        # Resources (food represents combined food types)
        self.resources = {
            'food': 1000.0,   # Combined food (dairy/beef/grain)
            'gold': 500.0,
            'stone': 200.0,
            'iron': 100.0,
            'wood': 200.0,    # Added wood for castle building
            'weapons': 50     # Weapons for arming troops
        }
        
        # Health system (manual page 26)
        self.health_rating = 'good'  # perfect, good, average, sick, diseased
        
        # Population with serfs/peasants terminology from original
        self.population = 100
        self.max_population = 500
        self.happiness = 75.0
        
        # County/Territory management (multiple counties like original)
        self.counties = {
            'home_county': {
                'name': 'Home County',
                'population': 100,
                'happiness': 75,
                'health': 'good',
                'fields': {'cattle': 3, 'grain': 0, 'empty': 2},
                'industries': {
                    'lumber_mill': False,
                    'quarry': False, 
                    'iron_mine': False,
                    'blacksmith': False
                },
                'castle': None,
                'garrison': 0
            }
        }
        self.selected_county = 'home_county'
        
        # Industry buildings (toggle on/off like original - manual page 46-49)
        self.industries = {
            'lumber_mill': {'active': False, 'output': 5.0},
            'quarry': {'active': False, 'output': 2.0},
            'iron_mine': {'active': False, 'output': 1.0},
            'blacksmith': {'active': False, 'output': 0.5}
        }
        
        # Castle building system (manual page 50-54)
        self.castle = {
            'under_construction': False,
            'design': 'small',  # small, medium, large, fortress
            'progress': 0,      # 0-100%
            'materials_needed': {'wood': 100, 'stone': 100},
            'workers_assigned': 0,
            'fortification': 0,
            'has_moat': False,
            'tower_count': 0
        }
        
        # Army composition with authentic troop types (manual page 70)
        self.army = {
            'peasants': 50,     # Unarmed farmers (weak but numerous)
            'archers': 0,       # Ranged units
            'swordsmen': 0,     # Basic infantry
            'knights': 0,       # Elite cavalry
            'catapults': 0      # Siege weapons
        }
        self.army_location = 'home'  # Location of army on map
        self.army_wages = 0      # Weekly wages for professional troops
        
        # Mercenary system (manual page 68)
        self.mercenaries_available = []
        self.mercenary_hired = []
        
        # Combat state
        self.in_combat = False
        self.in_siege = False
        self.enemy_strength = 0
        self.player_strength = 0
        self.combat_log = []
        self.battle_paused = False
        
        # Diplomacy - AI nobles from original game (manual page 97)
        self.ai_nobles = {
            'bishop': {'name': 'The Bishop', 'color': 'purple', 'relation': 50, 'alive': True},
            'baron': {'name': 'The Baron', 'color': 'red', 'relation': 50, 'alive': True},
            'knight': {'name': 'The Knight', 'color': 'blue', 'relation': 50, 'alive': True},
            'countess': {'name': 'The Countess', 'color': 'green', 'relation': 50, 'alive': True}
        }
        self.alliances = []
        self.enemies = []
        self.messages_received = []
        
        # Merchant trading system (manual page 57)
        self.merchant_present = False
        self.merchant_inventory = {
            'cattle': {'price': 50, 'stock': 10},
            'grain': {'price': 30, 'stock': 20},
            'weapons': {'price': 80, 'stock': 15},
            'ale': {'price': 100, 'stock': 5}  # Ale boosts happiness
        }
        
        # Tax system (manual page 59)
        self.tax_rate = 'normal'  # none, low, normal, high, excessive
        self.treasury = 500.0
        
        # Conscription level (affects happiness - manual page 27)
        self.conscription_rate = 0  # % of population drafted
        
        # Season/Year tracking (turn-based by season)
        self.season = 'spring'  # spring, summer, autumn, winter
        self.year = 1268  # Starting year from original game
        self.turn_number = 1
        
        # Research/Technology (academy based)
        self.technologies = {
            'farming': 1,
            'mining': 1,
            'warfare': 1,
            'construction': 1,
            'trade': 1
        }
        
        # === NEURAL NETWORK AI SYSTEM ===
        self.neural_network = NeuralNetwork()
        self.auto_play = False
        self.nn_actions_taken = 0
        self.nn_learning_enabled = True
        
        # Time tracking
        self.start_time = time.time()
        self.last_save_time = time.time()
        self.last_nn_save_time = time.time()
        self.offline_start = None
        self.offline_end = None
        
        # UI navigation
        self.current_tab = 0
        self.current_sub_tab = 0
        self.tabs = [
            {'name': 'County', 'sub_tabs': ['Overview', 'Labor', 'Fields', 'Rations']},
            {'name': 'Industry', 'sub_tabs': ['Buildings', 'Castle', 'Garrison']},
            {'name': 'Military', 'sub_tabs': ['Army', 'Training', 'Battles']},
            {'name': 'Economy', 'sub_tabs': ['Treasury', 'Merchant', 'Trade']},
            {'name': 'Diplomacy', 'sub_tabs': ['Nobles', 'Alliances', 'Messages']},
            {'name': 'AI Neural Net', 'sub_tabs': ['Control', 'Learning', 'Stats']}
        ]
        
        # Messages and notifications
        self.messages = []
        self.offline_popup = None
        
        # Scaling factors
        self.world_scale = 1.0
        self.difficulty_multiplier = 1.0
        
        # Load existing save
        self.load_game()
    
    def calculate_production(self) -> Dict[str, float]:
        """Calculate resource production based on authentic LOTR2 mechanics:
        - Labor allocation between Agriculture and Industry
        - Field assignments (cattle/grain)
        - Industry buildings (lumber/quarry/iron/blacksmith)
        - Ration levels affecting worker efficiency
        """
        # Base production rates per worker
        base_rates = {
            'food': 0.5,      # From agriculture (cattle + grain)
            'gold': 0.3,      # From mining/taxes
            'stone': 0.2,     # From quarry
            'iron': 0.1,      # From iron mines
            'wood': 0.5,      # From lumber mills
            'weapons': 0.05   # From blacksmiths
        }
        
        # Calculate effective workforce
        total_workers = self.population * 0.7  # 70% of pop are workers
        
        # Agriculture production (based on labor allocation and fields)
        agri_workers = total_workers * (self.labor_agriculture / 100)
        
        # Cattle production (automatic from cattle fields)
        cattle_fields = self.fields.get('cattle', 0)
        cattle_factor = max(1, cattle_fields / self.total_fields)
        dairy_production = cattle_fields * 10 * cattle_factor
        beef_production = self.beef_available * 2 if self.diet_slider >= 50 else 0
        
        # Grain production (from grain fields)
        grain_fields = self.fields.get('grain', 0)
        grain_production = grain_fields * 15 * (agri_workers / max(total_workers, 1))
        
        # Total food = dairy + beef + grain
        food_rate = dairy_production + beef_production + grain_production
        
        # Industry production (based on labor allocation and active industries)
        industry_workers = total_workers * (self.labor_industry / 100)
        active_industries = sum(1 for ind in self.industries.values() if ind['active'])
        
        if active_industries > 0:
            workers_per_industry = industry_workers / active_industries
            
            wood_rate = self.industries['lumber_mill']['output'] * workers_per_industry if self.industries['lumber_mill']['active'] else 0
            stone_rate = self.industries['quarry']['output'] * workers_per_industry if self.industries['quarry']['active'] else 0
            iron_rate = self.industries['iron_mine']['output'] * workers_per_industry if self.industries['iron_mine']['active'] else 0
            weapon_rate = self.industries['blacksmith']['output'] * workers_per_industry if self.industries['blacksmith']['active'] else 0
        else:
            wood_rate = stone_rate = iron_rate = weapon_rate = 0
        
        # Technology bonuses
        food_rate *= (1 + self.technologies['farming'] * 0.1)
        wood_rate *= (1 + self.technologies['construction'] * 0.1)
        stone_rate *= (1 + self.technologies['construction'] * 0.1)
        iron_rate *= (1 + self.technologies['mining'] * 0.1)
        weapon_rate *= (1 + self.technologies['warfare'] * 0.1)
        
        # Health/rations effect on productivity
        health_multipliers = {'perfect': 1.2, 'good': 1.0, 'average': 0.8, 'sick': 0.5, 'diseased': 0.2}
        health_mult = health_multipliers.get(self.health_rating, 1.0)
        
        # Happiness effect
        happiness_mult = 0.5 + (self.happiness / 100) * 0.5
        
        # Apply multipliers
        food_rate *= health_mult * happiness_mult
        wood_rate *= health_mult * happiness_mult
        stone_rate *= health_mult * happiness_mult
        iron_rate *= health_mult * happiness_mult
        weapon_rate *= health_mult * happiness_mult
        
        # Gold from taxes (based on tax rate and population)
        tax_rates = {'none': 0, 'low': 0.1, 'normal': 0.2, 'high': 0.4, 'excessive': 0.6}
        gold_rate = self.population * tax_rates.get(self.tax_rate, 0.2) * 0.1
        
        # World scaling
        scale = self.world_scale
        
        return {
            'food': food_rate * scale,
            'gold': gold_rate * scale,
            'stone': stone_rate * scale,
            'iron': iron_rate * scale,
            'wood': wood_rate * scale,
            'weapons': weapon_rate * scale
        }
    
    def update(self, delta_time: float):
        """Update game state with authentic LOTR2 seasonal mechanics"""
        if self.in_combat:
            return
        
        # Resource production
        rates = self.calculate_production()
        for resource, rate in rates.items():
            if resource in self.resources:
                self.resources[resource] += rate * delta_time
        
        # Food consumption based on rations (manual page 31-33)
        ration_consumption = {
            'none': 0,
            'quarter': 0.25,
            'half': 0.5,
            'normal': 1.0,
            'double': 2.0,
            'triple': 3.0
        }
        consumption_rate = ration_consumption.get(self.ration_level, 1.0)
        food_needed = self.population * 0.1 * consumption_rate
        
        # Dairy is automatic and cannot be stored
        dairy_fed = min(self.dairy_produce, self.population)
        remaining_pop = max(0, self.population - dairy_fed)
        
        # Rest comes from stored food (beef/grain based on diet slider)
        if remaining_pop > 0:
            self.resources['food'] -= remaining_pop * 0.1 * consumption_rate * delta_time
        
        # Health effects from rations
        achieved_ration = self.ration_level
        if self.resources['food'] < food_needed * 0.25:
            achieved_ration = 'none'
        elif self.resources['food'] < food_needed * 0.5:
            achieved_ration = 'quarter'
        elif self.resources['food'] < food_needed:
            achieved_ration = 'half'
        
        # Update health based on rations
        if achieved_ration in ['normal', 'double', 'triple']:
            if self.health_rating in ['sick', 'diseased']:
                self.health_rating = 'average'
            elif self.health_rating == 'average':
                self.health_rating = 'good'
        elif achieved_ration in ['none', 'quarter']:
            if self.health_rating == 'good':
                self.health_rating = 'average'
            elif self.health_rating == 'average':
                self.health_rating = 'sick'
        
        # Population growth (immigration + births when happy)
        if self.happiness > 60 and self.resources['food'] > food_needed:
            growth_rate = 0.05 * (self.happiness / 100)
            immigration = 0.02 * (self.happiness / 50)
            self.population += (growth_rate + immigration) * delta_time
            self.population = min(self.population, self.max_population)
        elif self.happiness < 30:
            # Emigration when unhappy
            self.population -= 0.03 * delta_time
        
        # Happiness changes
        # Rations effect
        ration_happiness = {'none': -0.1, 'quarter': -0.05, 'half': -0.02, 'normal': 0, 'double': 0.02, 'triple': 0.05}
        self.happiness += ration_happiness.get(self.ration_level, 0) * delta_time
        
        # Tax effect
        tax_happiness = {'none': 0.05, 'low': 0.02, 'normal': 0, 'high': -0.03, 'excessive': -0.1}
        self.happiness += tax_happiness.get(self.tax_rate, 0) * delta_time
        
        # Conscription effect
        if self.conscription_rate > 20:
            self.happiness -= 0.05 * delta_time
        
        # Clamp happiness
        self.happiness = max(0, min(100, self.happiness))
        
        # Season progression (each turn = 1 season)
        # Seasons affect production slightly
        season_multipliers = {'spring': 1.0, 'summer': 1.1, 'autumn': 0.9, 'winter': 0.7}
        
        # Neural network auto-play
        if self.auto_play and self.nn_learning_enabled:
            self.run_neural_network_ai(delta_time)
        
        # Auto-save neural network periodically
        current_time = time.time()
        if current_time - self.last_nn_save_time >= 10.0:
            self.save_neural_network()
            self.last_nn_save_time = current_time
        # Auto-save every second
        current_time = time.time()
        if current_time - self.last_save_time >= 1.0:
            self.save_game()
            self.last_save_time = current_time
        
        # Auto-save neural network every 5 seconds
        if current_time - self.last_nn_save_time >= 5.0:
            self.save_neural_network()
            self.last_nn_save_time = current_time
    
    def upgrade_building(self, building: str) -> bool:
        """Upgrade a building"""
        if building not in self.buildings:
            return False
        
        current_level = self.buildings[building]['level']
        max_level = self.buildings[building]['max_level']
        
        if current_level >= max_level:
            return False
        
        # Cost calculation with scaling
        base_costs = {
            'farm': {'food': 100, 'gold': 50},
            'mine': {'food': 150, 'gold': 100},
            'quarry': {'food': 200, 'stone': 100},
            'iron_works': {'food': 250, 'stone': 150, 'iron': 50},
            'barracks': {'food': 300, 'gold': 200, 'stone': 100},
            'castle': {'food': 500, 'gold': 500, 'stone': 500, 'iron': 200},
            'market': {'food': 200, 'gold': 300},
            'academy': {'food': 400, 'gold': 400, 'stone': 200}
        }
        
        costs = {}
        for resource, base_cost in base_costs.get(building, {}).items():
            costs[resource] = base_cost * (1.5 ** current_level) * self.world_scale
        
        # Check if player can afford
        for resource, cost in costs.items():
            if self.resources.get(resource, 0) < cost:
                return False
        
        # Deduct costs
        for resource, cost in costs.items():
            self.resources[resource] -= cost
        
        # Upgrade
        self.buildings[building]['level'] += 1
        
        # Update max population based on castle level
        if building == 'castle':
            self.max_population = int(500 * (1.5 ** self.buildings['castle']['level']))
        
        # Update max army size based on barracks level
        if building == 'barracks':
            self.max_army_size = int(1000 * (1.3 ** self.buildings['barracks']['level']))
        
        self.messages.append(f"{building.capitalize()} upgraded to level {current_level + 1}")
        return True
    
    def train_unit(self, unit_type: str, amount: int = 1) -> bool:
        """Train military units"""
        if unit_type not in self.army:
            return False
        
        current_total = sum(self.army.values())
        if current_total + amount > self.max_army_size:
            return False
        
        # Training costs
        unit_costs = {
            'peasants': {'food': 10, 'gold': 5},
            'archers': {'food': 50, 'gold': 30, 'iron': 10},
            'swordsmen': {'food': 75, 'gold': 50, 'iron': 25},
            'knights': {'food': 150, 'gold': 150, 'iron': 75},
            'catapults': {'food': 200, 'gold': 200, 'stone': 100, 'iron': 50}
        }
        
        costs = unit_costs.get(unit_type, {})
        total_costs = {resource: cost * amount for resource, cost in costs.items()}
        
        # Check affordability
        for resource, cost in total_costs.items():
            if self.resources.get(resource, 0) < cost:
                return False
        
        # Deduct costs
        for resource, cost in total_costs.items():
            self.resources[resource] -= cost
        
        # Add units
        self.army[unit_type] += amount
        
        # Train over time (instant for now)
        self.messages.append(f"Trained {amount} {unit_type}")
        return True
    
    def initiate_combat(self, enemy_province: str = None):
        """Initiate combat with an enemy"""
        if self.in_combat:
            return
        
        # Calculate player strength
        unit_strengths = {
            'peasants': 1,
            'archers': 3,
            'swordsmen': 5,
            'knights': 10,
            'catapults': 15
        }
        
        self.player_strength = sum(
            count * unit_strengths.get(unit, 1) 
            for unit, count in self.army.items()
        )
        
        # Castle bonus
        castle_level = self.buildings.get('castle', {}).get('level', 1)
        self.player_strength *= (1 + castle_level * 0.1)
        
        # Generate enemy strength based on world scale
        base_enemy_strength = 100 * (self.world_scale ** 1.5)
        enemy_variation = random.uniform(0.5, 2.0)
        self.enemy_strength = base_enemy_strength * enemy_variation * self.difficulty_multiplier
        
        self.in_combat = True
        self.combat_log = [f"Combat initiated! Enemy strength: {format_number(self.enemy_strength)}"]
        
        # AI decision making during combat
        if self.auto_play:
            self.nn_combat_decision()
    
    def resolve_combat(self) -> bool:
        """Resolve combat and return victory status"""
        if not self.in_combat:
            return False
        
        # Combat resolution with some randomness
        player_roll = random.uniform(0.8, 1.2)
        enemy_roll = random.uniform(0.8, 1.2)
        
        effective_player = self.player_strength * player_roll
        effective_enemy = self.enemy_strength * enemy_roll
        
        self.combat_log.append(f"Player: {format_number(effective_player)} vs Enemy: {format_number(effective_enemy)}")
        
        if effective_player >= effective_enemy:
            # Victory
            victory_margin = effective_player / max(effective_enemy, 1)
            
            # Casualties
            casualty_rate = max(0, (effective_enemy / max(effective_player, 1)) * 0.3)
            for unit in self.army:
                casualties = int(self.army[unit] * casualty_rate * random.uniform(0.5, 1.5))
                self.army[unit] = max(0, self.army[unit] - casualties)
            
            # Rewards
            reward_multiplier = min(victory_margin, 3.0)
            gold_reward = self.enemy_strength * 0.5 * reward_multiplier
            territory_gain = min(1, int(reward_multiplier))
            
            self.resources['gold'] += gold_reward
            self.territory += territory_gain
            self.total_provinces += 1
            
            self.combat_log.append(f"Victory! Gained {format_number(gold_reward)} gold, {territory_gain} territory")
            
            # AI learning
            if self.auto_play:
                self.neural_network.games_played += 1
                self.neural_network.wins += 1
                self.neural_network.total_resources_gathered += gold_reward
                target = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]  # Reinforce attack action
                self.neural_network.train(
                    self.neural_network.get_state_vector(self),
                    target,
                    reward=victory_margin
                )
            
            self.in_combat = False
            return True
        else:
            # Defeat
            loss_severity = effective_enemy / max(effective_player, 1)
            
            # Heavy casualties
            casualty_rate = min(0.9, 0.3 * loss_severity)
            for unit in self.army:
                casualties = int(self.army[unit] * casualty_rate * random.uniform(0.8, 1.5))
                self.army[unit] = max(0, self.army[unit] - casualties)
            
            self.combat_log.append(f"Defeat! Lost {casualty_rate*100:.1f}% of army")
            
            # AI learning from defeat
            if self.auto_play:
                self.neural_network.games_played += 1
                target = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]  # Reinforce defend action
                self.neural_network.train(
                    self.neural_network.get_state_vector(self),
                    target,
                    reward=-loss_severity
                )
            
            self.in_combat = False
            return False
    
    def nn_combat_decision(self):
        """Let neural network make combat decisions"""
        state = self.neural_network.get_state_vector(self)
        action = self.neural_network.decide_action(self)
        
        # Combat-specific actions
        if action == 6:  # Attack aggressively
            self.player_strength *= 1.2
            self.combat_log.append("AI chose aggressive attack!")
        elif action == 7:  # Defend
            self.player_strength *= 1.3
            self.combat_log.append("AI chose defensive stance!")
        elif action == 9:  # Special tactic
            self.player_strength *= random.uniform(0.9, 1.5)
            self.combat_log.append("AI attempted special tactic!")
        
        self.nn_actions_taken += 1
    
    def calculate_offline_progress(self, offline_duration: float) -> Dict[str, Any]:
        """Calculate progress made while offline"""
        rates = self.calculate_production()
        
        offline_earnings = {}
        for resource, rate in rates.items():
            offline_earnings[resource] = rate * offline_duration * 0.5  # 50% efficiency offline
        
        # Cap offline progress at 24 hours worth
        max_offline = 86400  # 24 hours in seconds
        if offline_duration > max_offline:
            scale_factor = max_offline / offline_duration
            for resource in offline_earnings:
                offline_earnings[resource] *= scale_factor
        
        return {
            'duration': offline_duration,
            'earnings': offline_earnings,
            'population_growth': min(offline_duration * 0.01, self.max_population * 0.1)
        }
    
    def apply_offline_progress(self):
        """Apply offline progress when loading game"""
        if self.offline_start and self.offline_end:
            duration = self.offline_end - self.offline_start
            if duration > 60:  # Minimum 1 minute offline
                progress = self.calculate_offline_progress(duration)
                
                for resource, amount in progress['earnings'].items():
                    self.resources[resource] += amount
                
                self.population = min(self.max_population, 
                                    self.population + progress['population_growth'])
                
                self.offline_popup = {
                    'duration': duration,
                    'earnings': progress['earnings'],
                    'dismissed': False
                }
                
                self.messages.append(f"Welcome back! You were offline for {duration/3600:.2f} hours")
                self.messages.append(f"Offline earnings: {', '.join(f'{format_number(v)} {k}' for k, v in progress['earnings'].items())}")
        
        self.offline_start = None
        self.offline_end = None
    
    def save_game(self):
        """Save game state silently"""
        try:
            save_data = {
                'resources': self.resources,
                'production_rates': self.production_rates,
                'population': self.population,
                'max_population': self.max_population,
                'happiness': self.happiness,
                'buildings': self.buildings,
                'army': self.army,
                'max_army_size': self.max_army_size,
                'territory': self.territory,
                'max_territory': self.max_territory,
                'provinces': self.provinces,
                'total_provinces': self.total_provinces,
                'technologies': self.technologies,
                'world_scale': self.world_scale,
                'difficulty_multiplier': self.difficulty_multiplier,
                'start_time': self.start_time,
                'last_save_time': time.time(),
                'offline_start': self.offline_start,
                'messages': self.messages[-50:]  # Keep last 50 messages
            }
            
            with open('lords_save.json', 'w') as f:
                json.dump(save_data, f, indent=2)
        except Exception as e:
            pass  # Silent save failure
    
    def load_game(self):
        """Load game state"""
        try:
            if os.path.exists('lords_save.json'):
                with open('lords_save.json', 'r') as f:
                    save_data = json.load(f)
                
                self.resources = save_data.get('resources', self.resources)
                self.population = save_data.get('population', self.population)
                self.max_population = save_data.get('max_population', self.max_population)
                self.happiness = save_data.get('happiness', self.happiness)
                self.buildings = save_data.get('buildings', self.buildings)
                self.army = save_data.get('army', self.army)
                self.max_army_size = save_data.get('max_army_size', self.max_army_size)
                self.territory = save_data.get('territory', self.territory)
                self.max_territory = save_data.get('max_territory', self.max_territory)
                self.provinces = save_data.get('provinces', self.provinces)
                self.total_provinces = save_data.get('total_provinces', self.total_provinces)
                self.technologies = save_data.get('technologies', self.technologies)
                self.world_scale = save_data.get('world_scale', self.world_scale)
                self.difficulty_multiplier = save_data.get('difficulty_multiplier', self.difficulty_multiplier)
                self.start_time = save_data.get('start_time', self.start_time)
                self.messages = save_data.get('messages', [])
                
                # Handle offline progress
                saved_time = save_data.get('last_save_time')
                if saved_time:
                    self.offline_start = saved_time
                    self.offline_end = time.time()
                
                # Load neural network
                self.load_neural_network()
                
        except Exception as e:
            pass  # Start fresh on error
    
    def run_neural_network_ai(self, delta_time: float):
        """Run neural network AI to automate game decisions.
        The NN learns optimal strategies through reinforcement learning."""
        
        # Get current game state as input vector
        state = self.neural_network.get_state_vector(self)
        
        # Get AI decision from neural network
        action = self.neural_network.decide_action(self)
        
        # Execute action based on NN output
        reward = 0.0
        
        # Actions: 0-3: Adjust labor, 4: Toggle industry, 5: Change rations, 
        #          6: Train troops, 7: Trade, 8: End turn, 9: Explore/Attack
        
        if action == 0:  # Increase agriculture labor
            old_labor = self.labor_agriculture
            self.labor_agriculture = min(100, self.labor_agriculture + 5)
            self.labor_industry = 100 - self.labor_agriculture
            reward += 0.1 if self.resources['food'] < 500 else -0.1
            
        elif action == 1:  # Decrease agriculture labor (increase industry)
            old_labor = self.labor_agriculture
            self.labor_agriculture = max(0, self.labor_agriculture - 5)
            self.labor_industry = 100 - self.labor_agriculture
            reward += 0.1 if (self.industries['lumber_mill']['active'] or 
                             self.industries['quarry']['active']) else -0.1
            
        elif action == 2:  # Activate an industry
            inactive = [k for k, v in self.industries.items() if not v['active']]
            if inactive:
                chosen = random.choice(inactive)
                self.industries[chosen]['active'] = True
                reward += 0.2
                
        elif action == 3:  # Deactivate an industry (save workers)
            active = [k for k, v in self.industries.items() if v['active']]
            if len(active) > 1:
                chosen = random.choice(active)
                self.industries[chosen]['active'] = False
                reward += 0.1
                
        elif action == 4:  # Adjust rations based on food supply
            if self.resources['food'] > self.population * 2:
                ration_order = ['none', 'quarter', 'half', 'normal', 'double', 'triple']
                idx = ration_order.index(self.ration_level)
                if idx < len(ration_order) - 1:
                    self.ration_level = ration_order[idx + 1]
                    reward += 0.1
            elif self.resources['food'] < self.population * 0.5:
                ration_order = ['none', 'quarter', 'half', 'normal', 'double', 'triple']
                idx = ration_order.index(self.ration_level)
                if idx > 0:
                    self.ration_level = ration_order[idx - 1]
                    reward += 0.1
                    
        elif action == 5:  # Adjust diet slider
            if self.beef_available > 10:
                self.diet_slider = min(100, self.diet_slider + 10)
            else:
                self.diet_slider = max(0, self.diet_slider - 10)
            reward += 0.05
            
        elif action == 6:  # Train troops (conscript peasants)
            if self.population > 50 and self.conscription_rate < 30:
                self.conscription_rate = min(50, self.conscription_rate + 5)
                self.army['peasants'] += int(self.population * 0.05)
                reward += 0.15
                
        elif action == 7:  # Trade with merchant (if available)
            if self.merchant_present:
                # Buy food if low, sell weapons if high
                if self.resources['food'] < 200 and self.treasury > 100:
                    self.resources['food'] += 50
                    self.treasury -= 50
                    reward += 0.2
                elif self.resources.get('weapons', 0) > 100:
                    self.resources['weapons'] -= 10
                    self.treasury += 80
                    reward += 0.15
                    
        elif action == 8:  # End turn (advance season)
            self.end_turn()
            reward += 0.1
            
        elif action == 9:  # Explore/Attack (diplomacy action)
            # Random diplomatic action
            noble = random.choice(list(self.ai_nobles.keys()))
            if self.ai_nobles[noble]['alive']:
                self.ai_nobles[noble]['relation'] += random.randint(-10, 10)
                reward += 0.05
        
        # Train neural network with this experience
        target = np.zeros(self.neural_network.output_size)
        target[action] = 1.0
        self.neural_network.train(state, target, reward)
        
        self.nn_actions_taken += 1
        self.neural_network.total_resources_gathered += sum(self.resources.values()) * 0.001
    
    def save_neural_network(self):
        """Save neural network silently"""
        try:
            self.neural_network.save('lords_nn.pkl')
        except Exception as e:
            pass
    
    def load_neural_network(self):
        """Load neural network if exists"""
        try:
            if os.path.exists('lords_nn.pkl'):
                self.neural_network = NeuralNetwork.load('lords_nn.pkl')
        except Exception as e:
            pass


class UI:
    """Curses-based user interface"""
    
    def __init__(self, stdscr, game: Game):
        self.stdscr = stdscr
        self.game = game
        self.running = True
        self.message_scroll = 0
        
        # Setup curses
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.timeout(100)  # 100ms refresh
        
        # Colors
        if curses.has_colors():
            curses.start_color()
            curses.use_default_colors()
            curses.init_pair(1, curses.COLOR_WHITE, -1)      # Normal
            curses.init_pair(2, curses.COLOR_GREEN, -1)       # Success
            curses.init_pair(3, curses.COLOR_RED, -1)         # Danger
            curses.init_pair(4, curses.COLOR_YELLOW, -1)      # Warning
            curses.init_pair(5, curses.COLOR_CYAN, -1)        # Info
            curses.init_pair(6, curses.COLOR_MAGENTA, -1)     # Special
            curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Selected
        
        self.last_update = time.time()
    
    def draw_header(self):
        """Draw header with tabs"""
        height, width = self.stdscr.getmaxyx()
        
        # Title
        title = " LORDS OF THE REALM II - TEXT EDITION "
        self.stdscr.attron(curses.A_BOLD | curses.color_pair(6))
        self.stdscr.addstr(0, (width - len(title)) // 2, title[:width-1])
        self.stdscr.attroff(curses.A_BOLD | curses.color_pair(6))
        
        # Tabs
        tab_y = 2
        tab_x = 1
        for i, tab in enumerate(self.game.tabs):
            tab_text = f" {tab['name']} "
            if i == self.game.current_tab:
                self.stdscr.attron(curses.color_pair(7))
                self.stdscr.addstr(tab_y, tab_x, tab_text[:width-tab_x-1])
                self.stdscr.attroff(curses.color_pair(7))
            else:
                self.stdscr.attron(curses.A_DIM)
                self.stdscr.addstr(tab_y, tab_x, tab_text[:width-tab_x-1])
                self.stdscr.attroff(curses.A_DIM)
            tab_x += len(tab_text) + 1
        
        # Sub-tabs
        sub_tab_y = 3
        sub_tab_x = 1
        current_tabs = self.game.tabs[self.game.current_tab]['sub_tabs']
        for i, sub_tab in enumerate(current_tabs):
            sub_text = f" [{sub_tab}] "
            if i == self.game.current_sub_tab:
                self.stdscr.attron(curses.A_BOLD | curses.color_pair(5))
                self.stdscr.addstr(sub_tab_y, sub_tab_x, sub_text[:width-sub_tab_x-1])
                self.stdscr.attroff(curses.A_BOLD | curses.color_pair(5))
            else:
                self.stdscr.addstr(sub_tab_y, sub_tab_x, sub_text[:width-sub_tab_x-1])
            sub_tab_x += len(sub_text)
        
        # Time and resources summary
        uptime = time.time() - self.game.start_time
        time_str = f"Time: {uptime/3600:.2f}h | Scale: {format_number(self.game.world_scale)}x"
        self.stdscr.addstr(1, width - len(time_str) - 1, time_str[:width-2])
    
    def draw_footer(self):
        """Draw footer with controls"""
        height, width = self.stdscr.getmaxyx()
        
        controls = " TAB:←→ | SUB:↑↓ | ACTION:Enter | Q:Quit | S:Save | A:Auto-play "
        self.stdscr.attron(curses.A_REVERSE)
        self.stdscr.addstr(height - 1, 0, controls[:width-1].ljust(width-1))
        self.stdscr.attroff(curses.A_REVERSE)
    
    def draw_overview_status(self, start_y: int):
        """Draw overview status tab"""
        height, width = self.stdscr.getmaxyx()
        
        info = [
            f"Population: {format_number(self.game.population)} / {format_number(self.game.max_population)}",
            f"Happiness: {self.game.happiness:.1f}%",
            f"Territory: {format_number(self.game.territory)} / {format_number(self.game.max_territory)}",
            f"Provinces Controlled: {self.game.total_provinces}",
            f"World Scale: {format_number(self.game.world_scale)}x",
            f"Difficulty: {format_number(self.game.difficulty_multiplier)}x",
            "",
            "Army Size:",
        ]
        
        total_units = sum(self.game.army.values())
        info.append(f"  Total: {format_number(total_units)} / {format_number(self.game.max_army_size)}")
        
        for unit, count in self.game.army.items():
            if count > 0:
                info.append(f"  {unit.capitalize()}: {format_number(count)}")
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_overview_resources(self, start_y: int):
        """Draw overview resources tab"""
        height, width = self.stdscr.getmaxyx()
        
        info = ["Resource Holdings:", ""]
        
        for resource, amount in self.game.resources.items():
            rate = self.game.production_rates.get(resource, 0)
            info.append(f"  {resource.capitalize():12}: {format_number(amount):>15} (+{format_number(rate)}/s)")
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_overview_population(self, start_y: int):
        """Draw overview population tab"""
        height, width = self.stdscr.getmaxyx()
        
        pop_growth = 0
        if self.game.resources['food'] > 100:
            pop_growth = 0.1 * (self.game.happiness / 100)
        
        info = [
            "Population Details:",
            "",
            f"  Current: {format_number(self.game.population)}",
            f"  Maximum: {format_number(self.game.max_population)}",
            f"  Growth Rate: {pop_growth:.3f}/s",
            f"  Happiness: {self.game.happiness:.1f}%",
            "",
            "Happiness Factors:",
        ]
        
        if self.game.resources['food'] > self.game.population * 10:
            info.append("  ✓ Abundant food supply")
        else:
            info.append("  ✗ Food shortage!")
        
        if self.game.happiness > 80:
            info.append("  ✓ High morale")
        elif self.game.happiness < 40:
            info.append("  ✗ Low morale")
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_province_buildings(self, start_y: int):
        """Draw province buildings tab"""
        height, width = self.stdscr.getmaxyx()
        
        info = ["Buildings (Press Enter to upgrade):", ""]
        
        buildings_list = list(self.game.buildings.keys())
        for i, building in enumerate(buildings_list):
            data = self.game.buildings[building]
            level = data['level']
            max_level = data['max_level']
            
            # Calculate upgrade cost
            base_costs = {
                'farm': {'food': 100, 'gold': 50},
                'mine': {'food': 150, 'gold': 100},
                'quarry': {'food': 200, 'stone': 100},
                'iron_works': {'food': 250, 'stone': 150, 'iron': 50},
                'barracks': {'food': 300, 'gold': 200, 'stone': 100},
                'castle': {'food': 500, 'gold': 500, 'stone': 500, 'iron': 200},
                'market': {'food': 200, 'gold': 300},
                'academy': {'food': 400, 'gold': 400, 'stone': 200}
            }
            
            costs = []
            for res, base in base_costs.get(building, {}).items():
                cost = base * (1.5 ** level) * self.game.world_scale
                costs.append(f"{format_number(cost)} {res}")
            
            cost_str = ", ".join(costs) if costs else "Free"
            status = "MAX" if level >= max_level else f"Lvl {level}"
            
            line = f"  {building.capitalize():15} [{status:>8}] - Cost: {cost_str}"
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
        
        instructions = ["", "Navigate with ↑↓, Press Enter to upgrade selected building"]
        for i, line in enumerate(instructions):
            y = start_y + len(buildings_list) + i
            if y < height - 2:
                self.stdscr.addstr(y, 2, line[:width-3])
    
    def draw_province_upgrades(self, start_y: int):
        """Draw province upgrades tab"""
        height, width = self.stdscr.getmaxyx()
        
        info = ["Technologies & Upgrades:", ""]
        
        for tech, level in self.game.technologies.items():
            cost = 1000 * (2 ** level) * self.game.world_scale
            info.append(f"  {tech.capitalize():15} [Level {level}] - Research: {format_number(cost)} gold")
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_province_management(self, start_y: int):
        """Draw province management tab"""
        height, width = self.stdscr.getmaxyx()
        
        info = [
            "Province Management:",
            "",
            f"  Home Province: Fortification Level {self.game.provinces['home']['fortification']}",
            f"  Total Provinces: {self.game.total_provinces}",
            f"  Territory: {format_number(self.game.territory)}",
            "",
            "Actions:",
            "  - Press 'C' to initiate combat",
            "  - Press 'E' to explore new territory",
            "  - Press 'F' to fortify defenses"
        ]
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_military_army(self, start_y: int):
        """Draw military army tab"""
        height, width = self.stdscr.getmaxyx()
        
        unit_strengths = {
            'peasants': 1,
            'archers': 3,
            'swordsmen': 5,
            'knights': 10,
            'catapults': 15
        }
        
        info = ["Army Composition:", ""]
        
        total_strength = 0
        for unit, count in self.game.army.items():
            strength = count * unit_strengths.get(unit, 1)
            total_strength += strength
            info.append(f"  {unit.capitalize():12}: {format_number(count):>10} (Strength: {format_number(strength)})")
        
        info.extend([
            "",
            f"  Total Army Strength: {format_number(total_strength)}",
            f"  Max Army Size: {format_number(self.game.max_army_size)}",
            "",
            "Training Costs:",
            "  Peasants: 10 food, 5 gold",
            "  Archers: 50 food, 30 gold, 10 iron",
            "  Swordsmen: 75 food, 50 gold, 25 iron",
            "  Knights: 150 food, 150 gold, 75 iron",
            "  Catapults: 200 food, 200 gold, 100 stone, 50 iron"
        ])
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_military_training(self, start_y: int):
        """Draw military training tab"""
        height, width = self.stdscr.getmaxyx()
        
        info = [
            "Unit Training (Press number key to train 10):",
            "",
            "  1. Train 10 Peasants",
            "  2. Train 10 Archers",
            "  3. Train 10 Swordsmen",
            "  4. Train 10 Knights",
            "  5. Train 10 Catapults",
            "",
            "Hold Shift + number to train 100"
        ]
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_military_combat(self, start_y: int):
        """Draw military combat tab"""
        height, width = self.stdscr.getmaxyx()
        
        if self.game.in_combat:
            info = [
                "⚔️  COMBAT IN PROGRESS  ⚔️",
                "",
                f"  Player Strength: {format_number(self.game.player_strength)}",
                f"  Enemy Strength: {format_number(self.game.enemy_strength)}",
                "",
                "Combat Log:",
            ]
            
            for log_entry in self.game.combat_log[-10:]:
                info.append(f"  {log_entry}")
            
            # Auto-resolve after a delay
            if len(self.game.combat_log) > 5:
                self.game.resolve_combat()
        else:
            info = [
                "Combat System:",
                "",
                "  Press 'C' to initiate combat with enemy forces",
                "  Enemy strength scales with world difficulty",
                "",
                "Combat Tips:",
                "  - Maintain a balanced army composition",
                "  - Upgrade your castle for defensive bonuses",
                "  - Knights are strong but expensive",
                "  - Catapults excel against fortified positions",
                "",
                f"  Current World Scale: {format_number(self.game.world_scale)}x",
                f"  Difficulty Multiplier: {format_number(self.game.difficulty_multiplier)}x"
            ]
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_research_technologies(self, start_y: int):
        """Draw research technologies tab"""
        height, width = self.stdscr.getmaxyx()
        
        tech_effects = {
            'farming': '+20% food production per level',
            'mining': '+20% gold production per level',
            'warfare': '+20% iron production & unit strength per level',
            'construction': '+20% stone production & building efficiency',
            'trade': '+15% all resource production per level'
        }
        
        info = ["Technology Research (Press number to research):", ""]
        
        for i, (tech, level) in enumerate(self.game.technologies.items()):
            cost = 1000 * (2 ** level) * self.game.world_scale
            effect = tech_effects.get(tech, '')
            info.append(f"  {i+1}. {tech.capitalize():12} [Lvl {level}] - {format_number(cost)} gold")
            info.append(f"     Effect: {effect}")
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                color = curses.color_pair(5) if i % 2 == 0 else curses.color_pair(1)
                self.stdscr.attron(color)
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
                self.stdscr.attroff(color)
    
    def draw_research_progress(self, start_y: int):
        """Draw research progress tab"""
        height, width = self.stdscr.getmaxyx()
        
        total_levels = sum(self.game.technologies.values())
        avg_level = total_levels / len(self.game.technologies)
        
        info = [
            "Research Progress:",
            "",
            f"  Total Technology Levels: {total_levels}",
            f"  Average Level: {avg_level:.1f}",
            "",
            "Research Benefits:",
            "  - Increased resource production",
            "  - Stronger military units",
            "  - Better building efficiency",
            "  - Enhanced trade income",
            "",
            "Tip: Focus on one technology tree first for maximum impact"
        ]
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_diplomacy_provinces(self, start_y: int):
        """Draw diplomacy provinces tab"""
        height, width = self.stdscr.getmaxyx()
        
        info = [
            "Province Control:",
            "",
        ]
        
        for prov_name, prov_data in list(self.game.provinces.items())[:10]:
            owner = "YOU" if prov_data['owner'] == 'player' else "ENEMY"
            fort = prov_data.get('fortification', 1)
            info.append(f"  {prov_name}: {owner} (Fort: {fort})")
        
        info.extend([
            "",
            f"  Total Provinces Controlled: {self.game.total_provinces}",
            "",
            "Conquer more provinces by winning battles!"
        ])
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_diplomacy_trade(self, start_y: int):
        """Draw diplomacy trade tab"""
        height, width = self.stdscr.getmaxyx()
        
        market_level = self.game.buildings.get('market', {}).get('level', 1)
        trade_efficiency = 1 + market_level * 0.1
        
        info = [
            "Trade System:",
            "",
            f"  Market Level: {market_level}",
            f"  Trade Efficiency: {trade_efficiency:.1f}x",
            "",
            "Available Trades (Press number):",
            "  1. Sell 1000 Food → Gold",
            "  2. Sell 1000 Gold → Food",
            "  3. Sell 500 Stone → Gold",
            "  4. Sell 500 Iron → Gold",
            "",
            "Trade rates improve with Market level"
        ]
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_diplomacy_alliances(self, start_y: int):
        """Draw diplomacy alliances tab"""
        height, width = self.stdscr.getmaxyx()
        
        info = [
            "Alliances & Diplomacy:",
            "",
            "  No active alliances",
            "",
            "Future Features:",
            "  - Form alliances with AI lords",
            "  - Trade agreements",
            "  - Military pacts",
            "  - Declare war on enemies",
            "",
            "Focus on building your empire first!"
        ]
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_ai_neural_net(self, start_y: int):
        """Draw AI neural network tab"""
        height, width = self.stdscr.getmaxyx()
        
        nn = self.game.neural_network
        
        status = "ACTIVE" if self.game.auto_play else "INACTIVE"
        status_color = curses.color_pair(2) if self.game.auto_play else curses.color_pair(3)
        
        info = [
            "Neural Network AI Controller",
            "",
        ]
        
        self.stdscr.attron(status_color)
        info_line = f"  Status: {status}"
        self.stdscr.addstr(start_y + 2, 2, info_line[:width-3])
        self.stdscr.attroff(status_color)
        
        info = [
            "",
            f"  Generation: {nn.generation}",
            f"  Fitness Score: {nn.fitness:.2f}",
            f"  Games Played: {nn.games_played}",
            f"  Wins: {nn.wins}",
            f"  Win Rate: {(nn.wins/max(nn.games_played,1))*100:.1f}%",
            f"  Actions Taken: {nn.nn_actions_taken}",
            f"  Total Resources Gathered: {format_number(nn.total_resources_gathered)}",
            "",
            "Controls:",
            "  Press 'A' to toggle auto-play",
            "  Press 'M' to mutate network",
            "  Press 'R' to reset network",
            "",
            "The AI learns from combat outcomes and resource gathering"
        ]
        
        for i, line in enumerate(info):
            y = start_y + 3 + i
            if y < height - 2:
                self.stdscr.addstr(y, 2, line[:width-3])
    
    def draw_ai_stats(self, start_y: int):
        """Draw AI statistics tab"""
        height, width = self.stdscr.getmaxyx()
        
        nn = self.game.neural_network
        
        info = [
            "AI Performance Statistics:",
            "",
            f"  Network Architecture:",
            f"    Input Layer: {nn.input_size} neurons",
            f"    Hidden Layer: {nn.hidden_size} neurons",
            f"    Output Layer: {nn.output_size} neurons",
            "",
            f"  Learning Metrics:",
            f"    Total Fitness: {nn.fitness:.2f}",
            f"    Generations: {nn.generation}",
            f"    Learning Rate: {nn.learning_rate}",
            "",
            "  Action Distribution:",
            "    0-3: Resource gathering",
            "    4: Train troops",
            "    5: Build structures",
            "    6: Attack",
            "    7: Defend",
            "    8: Trade",
            "    9: Explore"
        ]
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_ai_settings(self, start_y: int):
        """Draw AI settings tab"""
        height, width = self.stdscr.getmaxyx()
        
        info = [
            "AI Settings:",
            "",
            f"  Auto-Play: {'ON' if self.game.auto_play else 'OFF'}",
            f"  Mutation Rate: {0.1:.1%}",
            f"  Learning Rate: {self.game.neural_network.learning_rate}",
            "",
            "Advanced Options:",
            "  - Adjust learning rate for faster/slower learning",
            "  - Increase mutation rate for more exploration",
            "  - Monitor fitness to track improvement",
            "",
            "The AI saves automatically every 5 seconds",
            "Network persists between game sessions"
        ]
        
        for i, line in enumerate(info):
            if start_y + i < height - 2:
                self.stdscr.addstr(start_y + i, 2, line[:width-3])
    
    def draw_messages(self, start_y: int):
        """Draw message log"""
        height, width = self.stdscr.getmaxyx()
        
        box_height = 6
        box_y = height - box_height - 1
        
        self.stdscr.attron(curses.A_REVERSE)
        self.stdscr.addstr(box_y, 0, " Messages ".center(width)[:width-1])
        self.stdscr.attroff(curses.A_REVERSE)
        
        messages = self.game.messages[-10:]
        for i, msg in enumerate(messages):
            y = box_y + 1 + i
            if y < height - 2:
                # Color-code messages
                if "Victory" in msg or "upgraded" in msg:
                    color = curses.color_pair(2)
                elif "Defeat" in msg or "shortage" in msg:
                    color = curses.color_pair(3)
                elif "offline" in msg.lower():
                    color = curses.color_pair(4)
                else:
                    color = curses.color_pair(1)
                
                self.stdscr.attron(color)
                self.stdscr.addstr(y, 2, msg[:width-3])
                self.stdscr.attroff(color)
    
    def draw_offline_popup(self):
        """Draw offline progress popup"""
        if not self.game.offline_popup or self.game.offline_popup.get('dismissed', True):
            return
        
        height, width = self.stdscr.getmaxyx()
        
        popup_width = min(60, width - 4)
        popup_height = 10
        popup_y = (height - popup_height) // 2
        popup_x = (width - popup_width) // 2
        
        # Draw popup border
        for i in range(popup_height):
            self.stdscr.addstr(popup_y + i, popup_x, " " * popup_width)
        
        self.stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
        title = " OFFLINE PROGRESS "
        self.stdscr.addstr(popup_y, popup_x + (popup_width - len(title)) // 2, title)
        self.stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)
        
        duration = self.game.offline_popup['duration']
        earnings = self.game.offline_popup['earnings']
        
        lines = [
            f"You were offline for {duration/3600:.2f} hours",
            "",
            "Resources gained:",
        ]
        
        for resource, amount in earnings.items():
            if amount > 0:
                lines.append(f"  {resource.capitalize()}: +{format_number(amount)}")
        
        lines.extend([
            "",
            "Press ENTER or SPACE to dismiss"
        ])
        
        for i, line in enumerate(lines):
            y = popup_y + 2 + i
            if y < popup_y + popup_height - 1:
                self.stdscr.addstr(y, popup_x + 2, line[:popup_width-4])
    
    def handle_input(self):
        """Handle user input"""
        try:
            key = self.stdscr.getch()
        except:
            return
        
        if key == -1:
            return
        
        # Quit
        if key == ord('q') or key == ord('Q'):
            self.running = False
            return
        
        # Tab navigation
        if key == curses.KEY_RIGHT:
            self.game.current_tab = (self.game.current_tab + 1) % len(self.game.tabs)
            self.game.current_sub_tab = 0
        elif key == curses.KEY_LEFT:
            self.game.current_tab = (self.game.current_tab - 1) % len(self.game.tabs)
            self.game.current_sub_tab = 0
        
        # Sub-tab navigation
        if key == curses.KEY_DOWN:
            current_tabs = self.game.tabs[self.game.current_tab]['sub_tabs']
            self.game.current_sub_tab = (self.game.current_sub_tab + 1) % len(current_tabs)
        elif key == curses.KEY_UP:
            current_tabs = self.game.tabs[self.game.current_tab]['sub_tabs']
            self.game.current_sub_tab = (self.game.current_sub_tab - 1) % len(current_tabs)
        
        # Save
        if key == ord('s') or key == ord('S'):
            self.game.save_game()
            self.game.save_neural_network()
            self.game.messages.append("Game saved successfully!")
        
        # Toggle auto-play
        if key == ord('a') or key == ord('A'):
            self.game.auto_play = not self.game.auto_play
            status = "enabled" if self.game.auto_play else "disabled"
            self.game.messages.append(f"AI auto-play {status}")
        
        # Dismiss offline popup
        if key == 10 or key == 13 or key == ord(' '):  # Enter or Space
            if self.game.offline_popup and not self.game.offline_popup.get('dismissed', True):
                self.game.offline_popup['dismissed'] = True
        
        # Mutate neural network
        if key == ord('m') or key == ord('M'):
            if self.game.current_tab == 5:  # AI Control tab
                self.game.neural_network.mutate()
                self.game.messages.append("Neural network mutated!")
        
        # Reset neural network
        if key == ord('r') or key == ord('R'):
            if self.game.current_tab == 5:  # AI Control tab
                self.game.neural_network = NeuralNetwork()
                self.game.messages.append("Neural network reset!")
        
        # Initiate combat
        if key == ord('c') or key == ord('C'):
            if not self.game.in_combat:
                self.game.initiate_combat()
                self.game.messages.append("Combat initiated!")
        
        # Building upgrades (in buildings tab)
        if key == 10 or key == 13:  # Enter
            if self.game.current_tab == 1 and self.game.current_sub_tab == 0:  # Province -> Buildings
                buildings_list = list(self.game.buildings.keys())
                # Could add selection logic here
                pass
        
        # Technology research
        if ord('1') <= key <= ord('5'):
            if self.game.current_tab == 3 and self.game.current_sub_tab == 0:  # Research -> Technologies
                tech_index = key - ord('1')
                tech_list = list(self.game.technologies.keys())
                if tech_index < len(tech_list):
                    tech = tech_list[tech_index]
                    level = self.game.technologies[tech]
                    cost = 1000 * (2 ** level) * self.game.world_scale
                    
                    if self.game.resources['gold'] >= cost:
                        self.game.resources['gold'] -= cost
                        self.game.technologies[tech] += 1
                        self.game.messages.append(f"Researched {tech} to level {level + 1}!")
        
        # Unit training
        if ord('1') <= key <= ord('5'):
            if self.game.current_tab == 2 and self.game.current_sub_tab == 1:  # Military -> Training
                units = ['peasants', 'archers', 'swordsmen', 'knights', 'catapults']
                unit_index = key - ord('1')
                if unit_index < len(units):
                    unit = units[unit_index]
                    self.game.train_unit(unit, 10)
    
    def ai_update(self):
        """Let AI make decisions"""
        if not self.game.auto_play:
            return
        
        nn = self.game.neural_network
        state = nn.get_state_vector(self.game)
        action = nn.decide_action(self.game)
        
        # Execute AI action
        if action < 4:  # Gather resources (simulated)
            resources = ['food', 'gold', 'stone', 'iron']
            resource = resources[action]
            self.game.resources[resource] += self.game.production_rates.get(resource, 0) * 10
            nn.total_resources_gathered += self.game.production_rates.get(resource, 0) * 10
        
        elif action == 4:  # Train troops
            units = ['peasants', 'archers', 'swordsmen', 'knights', 'catapults']
            unit = random.choice(units)
            self.game.train_unit(unit, 5)
        
        elif action == 5:  # Build/upgrade
            buildings = list(self.game.buildings.keys())
            building = random.choice(buildings)
            self.game.upgrade_building(building)
        
        elif action == 6:  # Attack
            if not self.game.in_combat:
                self.game.initiate_combat()
        
        elif action == 7:  # Defend (skip turn)
            pass
        
        elif action == 8:  # Trade
            # Simple trade logic
            if self.game.resources['food'] > 1000:
                self.game.resources['food'] -= 1000
                self.game.resources['gold'] += 500
        
        elif action == 9:  # Explore
            self.game.territory += 1
        
        nn.nn_actions_taken += 1
    
    def render(self):
        """Render the entire screen"""
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()
        
        # Draw header with tabs
        self.draw_header()
        
        # Draw content based on current tab and sub-tab
        content_start = 5
        
        if self.game.current_tab == 0:  # Overview
            if self.game.current_sub_tab == 0:
                self.draw_overview_status(content_start)
            elif self.game.current_sub_tab == 1:
                self.draw_overview_resources(content_start)
            elif self.game.current_sub_tab == 2:
                self.draw_overview_population(content_start)
        
        elif self.game.current_tab == 1:  # Province
            if self.game.current_sub_tab == 0:
                self.draw_province_buildings(content_start)
            elif self.game.current_sub_tab == 1:
                self.draw_province_upgrades(content_start)
            elif self.game.current_sub_tab == 2:
                self.draw_province_management(content_start)
        
        elif self.game.current_tab == 2:  # Military
            if self.game.current_sub_tab == 0:
                self.draw_military_army(content_start)
            elif self.game.current_sub_tab == 1:
                self.draw_military_training(content_start)
            elif self.game.current_sub_tab == 2:
                self.draw_military_combat(content_start)
        
        elif self.game.current_tab == 3:  # Research
            if self.game.current_sub_tab == 0:
                self.draw_research_technologies(content_start)
            elif self.game.current_sub_tab == 1:
                self.draw_research_progress(content_start)
        
        elif self.game.current_tab == 4:  # Diplomacy
            if self.game.current_sub_tab == 0:
                self.draw_diplomacy_provinces(content_start)
            elif self.game.current_sub_tab == 1:
                self.draw_diplomacy_trade(content_start)
            elif self.game.current_sub_tab == 2:
                self.draw_diplomacy_alliances(content_start)
        
        elif self.game.current_tab == 5:  # AI Control
            if self.game.current_sub_tab == 0:
                self.draw_ai_neural_net(content_start)
            elif self.game.current_sub_tab == 1:
                self.draw_ai_stats(content_start)
            elif self.game.current_sub_tab == 2:
                self.draw_ai_settings(content_start)
        
        # Draw messages
        self.draw_messages(content_start)
        
        # Draw offline popup if needed
        self.draw_offline_popup()
        
        # Draw footer
        self.draw_footer()
        
        self.stdscr.refresh()
    
    def run(self):
        """Main game loop"""
        while self.running:
            # Calculate delta time
            current_time = time.time()
            delta_time = current_time - self.last_update
            self.last_update = current_time
            
            # Update game state
            self.game.update(delta_time)
            
            # AI decision making
            self.ai_update()
            
            # Handle input
            self.handle_input()
            
            # Render
            self.render()
        
        # Save on exit
        self.game.save_game()
        self.game.save_neural_network()


def main(stdscr):
    """Main entry point"""
    game = Game()
    ui = UI(stdscr, game)
    ui.run()


if __name__ == "__main__":
    print("Starting Lords of the Realm II - Text Edition...")
    print("Loading game and neural network...")
    
    # Ensure we have numpy
    try:
        import numpy as np
    except ImportError:
        print("Installing numpy...")
        os.system('pip install numpy')
        import numpy as np
    
    # Run the game
    curses.wrapper(main)
