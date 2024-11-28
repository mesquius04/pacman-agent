# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def get_home_boundary(self, game_state):
        """
        Returns a list of positions on the boundary closest to the agent's side.
        """
        walls = game_state.get_walls()
        height = walls.height
        mid_x = walls.width // 2 - 1 if self.red else walls.width // 2
        return [(mid_x, y) for y in range(height) if not walls[mid_x][y]]

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)


    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


import heapq

class OffensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        walls = game_state.get_walls()

        # Agent's position
        my_pos = tuple(map(int, successor.get_agent_state(self.index).get_position()))

        # Distance to nearest food
        if food_list:
            closest_food = self.a_star_search(my_pos, food_list, walls)
            features['distance_to_food'] = closest_food if closest_food is not None else 0

        # Returning home logic
        carrying_food = successor.get_agent_state(self.index).num_carrying
        features['carrying_food'] = carrying_food
        if carrying_food > 0:
            home_boundary = self.get_home_boundary(successor)
            dist_to_boundary = self.a_star_search(my_pos, home_boundary, walls)
            features['distance_to_home'] = dist_to_boundary if dist_to_boundary is not None else 0

            # Force boundary crossing if near it
            if my_pos not in home_boundary:
                dist_to_boundary = self.a_star_search(my_pos, home_boundary, walls)
                if dist_to_boundary == 1:  # Adjacent to boundary
                    features['force_cross'] = 1

        # Ghost avoidance
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [e for e in enemies if not e.is_pacman and e.get_position() is not None]
        if ghosts:
            ghost_distances = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            min_ghost_distance = min(ghost_distances)
            features['ghost_distance'] = min_ghost_distance
            if min_ghost_distance < 2:
                features['ghost_nearby'] = 1
        else:
            features['ghost_distance'] = 5

        # Stop and reverse penalties
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'distance_to_food': -10,       # The agent prefers to be closer to food (negative weight makes it avoid more distance).
            'carrying_food': 200,          # The agent gets a high positive weight when it is carrying food, encouraging it to prioritize food collection.
            'distance_to_home': -10,        # The agent prefers to be closer to its home (negative weight encourages returning to the home base).
            'force_cross': 700,            # Strong positive weight to encourage crossing the boundary if the agent is far from home.
            'successor_score': 100,        # Positive weight to prefer actions that increase the score (usually when it collects food).
            'ghost_distance': 15,          # Encourages the agent to stay away from ghosts by assigning a positive weight to actions that move farther from ghosts.
            'ghost_nearby': -750,          # Strong negative weight when ghosts are nearby, discouraging risky actions that might bring the agent near a ghost.
            'stop': -1000,                  # A strong negative weight for stopping (i.e., not moving), encouraging the agent to always try to move.
            'reverse': -2                  # A small negative weight for reversing, discouraging moving backwards unless necessary.
        }

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_pos = game_state.get_agent_state(self.index).get_position()
        carrying_food = game_state.get_agent_state(self.index).num_carrying
        home_boundary = self.get_home_boundary(game_state)  # Assume this returns home boundary positions

        # If carrying food, prioritize crossing the boundary
        if carrying_food > 0:
            for action in actions:
                successor = self.get_successor(game_state, action)
                successor_pos = successor.get_agent_state(self.index).get_position()
                if successor_pos in home_boundary:
                    return action

        # Default behavior: evaluate all actions and choose the best one
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        return random.choice(best_actions)

    def get_home_boundary(self, game_state):
        """
        Returns the positions on the boundary between the agent's side and the opponent's side.
        """
        walls = game_state.get_walls()
        width = walls.width
        height = walls.height
        mid_x = width // 2 - 1 if self.red else width // 2
        return [(mid_x, y) for y in range(height) if not walls[mid_x][y]]

    def a_star_search(self, start, goals, walls):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current in goals:
                return cost_so_far[current]

            for next_pos in self.get_neighbors(current, walls):
                new_cost = cost_so_far[current] + 1
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + self.heuristic(next_pos, goals)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current

        return None

    def get_neighbors(self, position, walls):
        x, y = position
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        neighbors = [(int(x + dx), int(y + dy)) for dx, dy in directions]
        return [n for n in neighbors if not walls[n[0]][n[1]]]

    def heuristic(self, pos, goals):
        return min(util.manhattan_distance(tuple(map(int, pos)), goal) for goal in goals)

class DefensiveReflexAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = tuple(map(int, my_state.get_position()))

        # Detect invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        # Detect invaders and evaluate their threat level
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if invaders:
            invader_distances = [self.get_maze_distance(my_pos, inv.get_position()) for inv in invaders]
            closest_invader = invaders[invader_distances.index(min(invader_distances))]
            closest_distance = min(invader_distances)
            
            # Factor in invader's proximity to base
            base_proximity = self.get_maze_distance(closest_invader.get_position(), self.get_team_base_position(game_state))
            threat_level = 1 / (closest_distance + base_proximity + 1)
            features['invader_threat'] = threat_level
            features['num_invaders'] = len(invaders)

        # Pursue invaders if there are any
        if len(invaders) > 0:
            invader_distances = [self.get_maze_distance(my_pos, invader.get_position()) for invader in invaders]
            features['invader_distance'] = min(invader_distances)

            # Prioritize getting closer to invaders
            features['invader_pursuit'] = 1 / (min(invader_distances) + 1)

        # Penalize stopping or reversing unnecessarily
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        capsule_positions = self.get_capsules_you_are_defending(successor)
        if capsule_positions:
            capsule_distances = [self.get_maze_distance(my_pos, capsule) for capsule in capsule_positions]
            features['capsule_protection'] = 1 / (min(capsule_distances) + 1)

        if not invaders:
            defensive_positions = self.get_defensive_positions(successor)
            if defensive_positions:
                defensive_distances = [self.get_maze_distance(my_pos, pos) for pos in defensive_positions]
                features['defensive_position'] = min(defensive_distances)


        # Reward patrolling key choke points if no invaders are visible
        if not invaders:
            patrol_points = self.get_dynamic_patrol_points(successor)
            if patrol_points:
                patrol_distances = [self.get_maze_distance(my_pos, point) for point in patrol_points]
                features['patrol_distance'] = min(patrol_distances)
        return features

    def get_dynamic_patrol_points(self, game_state):
        """
        Determine patrol points dynamically based on recent food or capsule events.
        """
        patrol_points = self.get_choke_points(game_state)
        food_positions = self.get_food(game_state).as_list()
        capsule_positions = self.get_capsules_you_are_defending(game_state)
        
        # Add recently eaten food and capsules to patrol targets
        patrol_points += food_positions + capsule_positions
        return patrol_points

    def get_weights(self, game_state, action):
        num_food_left = len(self.get_food(game_state).as_list())
        if num_food_left < 5:
            return {
                'num_invaders': -1500,
                'invader_distance': -80,
                'invader_pursuit': 300,
                'stop': -100,
                'reverse': -5,
                'patrol_distance': -2,
                'capsule_protection': 50,
                'distance_to_food': -10,
                'carrying_food': 50,
                'distance_to_home': -20,       
                'force_cross': 400,   
                'successor_score': 100, 
                'ghost_distance': 15,
                'ghost_nearby': -750,
                'stop': -1000,
                'reverse': -2 
            }
        else:
            return {
                'num_invaders': -1200,
                'invader_distance': -60,
                'invader_pursuit': 250,
                'stop': -200,
                'reverse': -2,
                'patrol_distance': -5,
                'capsule_protection': 20,
                'distance_to_food': -10,
                'carrying_food': 100,
                'distance_to_home': -15,
                'force_cross': 250,
                'successor_score': 50,
                'ghost_distance': 5,
                'ghost_nearby': -20,
                'reverse': -1
            }

    def get_defensive_positions(self, game_state):
        """
        Identify strategic defensive positions closer to the center of the map.
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2 - 1 if self.red else walls.width // 2
        defensive_positions = []
        for y in range(walls.height):
            if not walls[mid_x][y]:  # Asegurarse de que no sea una pared
                pos = (int(mid_x - 1), int(y))  # Convertir a enteros
                if self.is_valid_position(pos, walls):  # Validar posición explícitamente
                    defensive_positions.append(pos)
        return defensive_positions

    def is_valid_position(self, pos, walls):
        """
        Check if a position is within the grid and not a wall.
        """
        x, y = pos
        return 0 <= x < walls.width and 0 <= y < walls.height and not walls[x][y]


    def get_choke_points(self, game_state):
        """
        Identify key choke points on the map for patrolling.
        """
        walls = game_state.get_walls()
        mid_x = walls.width // 2 - 1 if self.red else walls.width // 2
        choke_points = []
        for y in range(walls.height):
            if not walls[mid_x][y] and not walls[mid_x - 1][y]:
                choke_points.append((mid_x, y))
        
        return choke_points 
    def get_team_base_position(self, game_state):
        """
        Dynamically calculates the approximate position of the team's base
        based on the layout.
        """
        walls = game_state.get_walls()
        max_width = walls.width
        max_height = walls.height

        if game_state.is_on_red_team(self.index):
            # Red team base is on the left side
            return (1, max_height // 2)
        else:
            # Blue team base is on the right side
            return (max_width - 2, max_height // 2)

