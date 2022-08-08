import enum
import math
import random
import uuid
from enum import Enum

import mesa
import numpy as np
from collections import defaultdict

import mesa.space
from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.visualization.ModularVisualization import VisualizationElement, ModularServer
from mesa.visualization.modules import ChartModule

from mesa.batchrunner import BatchRunner
import matplotlib.pyplot as plt

MAX_ITERATION = 100
PROBA_CHGT_ANGLE = 0.01


def move(x, y, speed, angle):
    return x + speed * math.cos(angle), y + speed * math.sin(angle)


def go_to(x, y, speed, dest_x, dest_y):
    if np.linalg.norm((x - dest_x, y - dest_y)) < speed:
        return (dest_x, dest_y), 2 * math.pi * random.random()
    else:
        angle = math.acos((dest_x - x)/np.linalg.norm((x - dest_x, y - dest_y)))
        if dest_y < y:
            angle = - angle
        return move(x, y, speed, angle), angle

def distanceTo(object1, object2):
    try:
        return np.linalg.norm((object1.x - object2.x, object1.y - object2.y))
    except AttributeError:
        # Special case for object 1 is object and object2 is list
        return np.linalg.norm((object1.x - object2[0], object1.y - object2[1]))


class MarkerPurpose(Enum):
    DANGER = enum.auto(),
    INDICATION = enum.auto()


class ContinuousCanvas(VisualizationElement):
    local_includes = [
        "./js/simple_continuous_canvas.js",
    ]

    def __init__(self, canvas_height=500,
                 canvas_width=500, instantiate=True):
        VisualizationElement.__init__(self)
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.identifier = "space-canvas"
        if (instantiate):
            new_element = ("new Simple_Continuous_Module({}, {},'{}')".
                           format(self.canvas_width, self.canvas_height, self.identifier))
            self.js_code = "elements.push(" + new_element + ");"

    def portrayal_method(self, obj):
        return obj.portrayal_method()

    def render(self, model):
        representation = defaultdict(list)
        for obj in model.schedule.agents:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        for obj in model.mines:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        for obj in model.markers:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        for obj in model.obstacles:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        for obj in model.quicksands:
            portrayal = self.portrayal_method(obj)
            if portrayal:
                portrayal["x"] = ((obj.x - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.y - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        return representation


class Obstacle:  # Environnement: obstacle infranchissable
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": "black",
                     "r": self.r}
        return portrayal


class Quicksand:  # Environnement: ralentissement
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": "olive",
                     "r": self.r}
        return portrayal


class Mine:  # Environnement: élément à ramasser
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 2,
                     "Color": "black",
                     "r": 2}
        return portrayal


class Marker:  # La classe pour les balises
    def __init__(self, x, y, purpose, direction=None):
        self.x = x
        self.y = y
        self.purpose = purpose
        if purpose == MarkerPurpose.INDICATION:
            if direction is not None:
                self.direction = direction
            else:
                raise ValueError("Direction should not be none for indication marker")

    def portrayal_method(self):
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 2,
                     "Color": "red" if self.purpose == MarkerPurpose.DANGER else "green",
                     "r": 2}
        return portrayal


class Robot(Agent):  # La classe des agents
    def __init__(self, unique_id: int, model: Model, x, y, speed, sight_distance, angle=0.0):
        super().__init__(unique_id, model)
        self.x = x
        self.y = y
        self.speed = speed
        self.sight_distance = sight_distance
        self.angle = angle
        self.counter = 0

        self.mineToRemove = None # Mine to remove
        self.markerToRemove = None # Marker to remove
        self.danger = False # Bool true if robot sees a marker DANGER
        self.inQuicksand = False # Bool true if robot is in quicksand
        self.counterQS = 0 # Counter for number of steps in quicksand
        self.dest_x = None
        self.dest_y = None

    def step(self):

        quicksands = [quicksand for quicksand in self.model.quicksands if distanceTo(quicksand, self) <quicksand.r] # Quicksands where the robot is

        if quicksands and not self.inQuicksand:
            self.speed /= 2
            self.inQuicksand = True
        
        elif not quicksands and self.inQuicksand:
            # If was in quicksand and is not anymore
            self.speed *= 2
            self.inQuicksand = False
            self.counterQS = 0
            if self.model.enableDangerMarkers:
                marker = Marker(self.x, self.y, MarkerPurpose.DANGER)
                markers = sorted([m for m in self.model.markers if m.purpose==MarkerPurpose.DANGER], key= lambda m: distanceTo(m,self))
                if markers:
                    if (marker.x, marker.y, marker.purpose) not in [(m.x,m.y,m.purpose) for m in self.model.markers if m.purpose==MarkerPurpose.DANGER] and distanceTo(self, markers[0]) > self.speed:
                        # Second condition is to avoid dropping markers for the same perimeter
                        self.model.markers.append(marker)
                else:
                    # Special case for init
                    self.model.markers.append(marker)
        elif self.inQuicksand:
            # Still in quicksand
            self.counterQS += 1

        hasRemovedMine = False

        if self.mineToRemove:
            if self.x==self.mineToRemove.x and self.y==self.mineToRemove.y:
                try:
                    self.model.mines.remove(self.mineToRemove) # If the robot is on a mine, removes it
                    hasRemovedMine = True
                    self.mineToRemove = None
                except ValueError as e:
                    #print("Mine already removed.")
                    self.mineToRemove = None
                except Exception as e:
                    print("Error while removing mine.", e)
            else:
                # If target mine is still not removed, keep this mine as target
                (self.dest_x, self.dest_y), self.angle = go_to(self.x, self.y, self.speed, self.mineToRemove.x, self.mineToRemove.y)
        
        elif self.markerToRemove:
            if self.x==self.markerToRemove.x and self.y==self.markerToRemove.y:
                try:
                    self.model.markers.remove(self.markerToRemove) # If the robot is on a mine, removes it
                    if random.random() < 0.5:
                        self.angle = self.markerToRemove.direction + np.pi/2
                    else:
                        self.angle = self.markerToRemove.direction - np.pi/2
                    self.markerToRemove = None
                except ValueError as e:
                    #print("Marker already removed.")
                    self.markerToRemove = None
                except Exception as e:
                    print("Error while removing marker.", e)
            else:
                # If target marker is still not removed, keep this marker as target
                (self.dest_x, self.dest_y), self.angle = go_to(self.x, self.y, self.speed, self.markerToRemove.x, self.markerToRemove.y)
        else:
            markers = [marker for marker in self.model.markers if distanceTo(marker, self) < self.sight_distance] # Marker that are in sight range
            # If the robot do not aim to remove a mine or a marker, explore according to plan
            if self.counter % 6 == 0: # Allows 6 steps before robots get affected by markers
                
                dangerMarkers = []
                if self.model.enableDangerMarkers:
                    dangerMarkers = [marker for marker in markers if marker.purpose == MarkerPurpose.DANGER]
                if dangerMarkers and not self.danger:
                    self.danger = True
                    self.angle += np.pi
                    self.dest_x, self.dest_y = move(self.x, self.y, self.speed, self.angle)

            if not self.danger:
                
                mines = [mine for mine in self.model.mines if distanceTo(mine, self) < self.sight_distance] # Mines that are in sight range
                if mines:
                    self.mineToRemove = random.choice(mines)
                    (self.dest_x, self.dest_y), self.angle = go_to(self.x, self.y, self.speed, self.mineToRemove.x, self.mineToRemove.y)
                else:
                    indicationMarkers = [marker for marker in markers if marker.purpose == MarkerPurpose.INDICATION]
                    if indicationMarkers and self.counter%6==0:
                        self.markerToRemove = random.choice(indicationMarkers)
                        (self.dest_x, self.dest_y), self.angle = go_to(self.x, self.y, self.speed, self.markerToRemove.x, self.markerToRemove.y)
                    else:
                        if random.random() < PROBA_CHGT_ANGLE:
                            self.angle = random.vonmisesvariate(0,0) # Generate random angle in [0; 2*pi[ with mean angle equal to 0
                        self.dest_x, self.dest_y = move(self.x, self.y, self.speed, self.angle) # Projected destination

        stay = True # Bool true if destination is not allowed
        while stay:
            neighbors = [agent for agent in self.model.schedule.agents 
            if ( distanceTo(agent, self) < self.sight_distance 
            and distanceTo(agent, [self.dest_x, self.dest_y]) < agent.speed) and agent != self] # Get neighbors in sight range and that might move to projected destination and avoid collision
            
            obstacles = [obstacle for obstacle in self.model.obstacles if distanceTo(obstacle, [self.dest_x, self.dest_y]) < obstacle.r] # Obstacles that are in sight range
            
            if not (neighbors or obstacles) and (self.dest_x < 600 and self.dest_x > 0 and self.dest_y > 0 and self.dest_y < 600):
                stay = False
            else:
                self.angle = random.vonmisesvariate(0,0)
                self.dest_x, self.dest_y = move(self.x, self.y, self.speed, self.angle) # Projected new destination
        
        if hasRemovedMine:
            # If the robot has removed a mine, drop a marker INDICATION
            marker = Marker(self.x, self.y, MarkerPurpose.INDICATION, self.angle)
            self.model.markers.append(marker)
        
        self.counter += 1
        
        self.x, self.y = self.dest_x, self.dest_y # Move straight forward 
        self.danger = False

    def portrayal_method(self):
        portrayal = {"Shape": "arrowHead", "s": 1, "Filled": "true", "Color": "Red", "Layer": 3, 'x': self.x,
                     'y': self.y, "angle": self.angle}
        return portrayal


class MinedZone(Model):
    collector = DataCollector(
        model_reporters={"Mines": lambda model: len(model.mines),
                         "Danger markers": lambda model: len([m for m in model.markers if
                                                          m.purpose == MarkerPurpose.DANGER]),
                         "Indication markers": lambda model: len([m for m in model.markers if
                                                          m.purpose == MarkerPurpose.INDICATION]),
                         "Demined mines": lambda model: model.nb_init_mines-len(model.mines),
                         },
        agent_reporters={"Steps in quicksand": lambda agent: agent.counterQS})

    def __init__(self, n_robots, n_obstacles, n_quicksand, n_mines, speed, enableDangerMarkers=True):
        Model.__init__(self)
        self.space = mesa.space.ContinuousSpace(600, 600, False)
        self.schedule = RandomActivation(self)
        self.mines = []
        self.markers = []
        self.obstacles = []
        self.quicksands = []
        
        self.counter = 0 # Step counter
        self.nb_init_mines = n_mines
        self.meanStepsQS = 0
        self.enableDangerMarkers = enableDangerMarkers

        for _ in range(n_obstacles):
            self.obstacles.append(Obstacle(random.random() * 500, random.random() * 500, 10 + 20 * random.random()))
        for _ in range(n_quicksand):
            self.quicksands.append(Quicksand(random.random() * 500, random.random() * 500, 10 + 20 * random.random()))
        for _ in range(n_robots):
            x, y = random.random() * 500, random.random() * 500
            while [o for o in self.obstacles if np.linalg.norm((o.x - x, o.y - y)) < o.r] or \
                    [o for o in self.quicksands if np.linalg.norm((o.x - x, o.y - y)) < o.r]:
                x, y = random.random() * 500, random.random() * 500
            self.schedule.add(
                Robot(int(uuid.uuid1()), self, x, y, speed,
                      2 * speed, random.random() * 2 * math.pi))
        for _ in range(n_mines):
            x, y = random.random() * 500, random.random() * 500
            while [o for o in self.obstacles if np.linalg.norm((o.x - x, o.y - y)) < o.r] or \
                    [o for o in self.quicksands if np.linalg.norm((o.x - x, o.y - y)) < o.r]:
                x, y = random.random() * 500, random.random() * 500
            self.mines.append(Mine(x, y))
        self.datacollector = self.collector

    def step(self):
        self.counter += 1
        self.datacollector.collect(self)
        self.schedule.step()
        if not self.mines:
            self.meanStepsQS = self.datacollector.get_agent_vars_dataframe().groupby("AgentID").mean().mean()
            #print("Mean Steps Spent in Quicksands : {:.2f}".format(self.meanStepsQS.values[0]))
            self.running = False




def run_single_server():
    chart = ChartModule([{"Label": "Mines",
                          "Color": "Orange"},
                         {"Label": "Danger markers",
                          "Color": "Red"},
                         {"Label": "Indication markers",
                          "Color": "Green"},
                         {"Label": "Demined mines",
                          "Color": "Blue"},
                         ],
                        data_collector_name='datacollector')
    server = ModularServer(MinedZone,
                           [ContinuousCanvas(),
                            chart],
                           "Deminer robots",
                           {"n_robots": mesa.visualization.
                            ModularVisualization.UserSettableParameter('slider', "Number of robots", 7, 3,
                                                                       15, 1),
                            "n_obstacles": mesa.visualization.
                            ModularVisualization.UserSettableParameter('slider', "Number of obstacles", 5, 2, 10, 1),
                            "n_quicksand": mesa.visualization.
                            ModularVisualization.UserSettableParameter('slider', "Number of quicksand", 5, 2, 10, 1),
                            "speed": mesa.visualization.
                            ModularVisualization.UserSettableParameter('slider', "Robot speed", 15, 5, 40, 5),
                            "n_mines": mesa.visualization.
                            ModularVisualization.UserSettableParameter('slider', "Number of mines", 15, 5, 30, 1),
                            })
    server.port = 8521
    server.launch()

def run_batch(n_iters=10):
    fixed_params = {
        "n_robots": 7,
        "n_obstacles": 5,
        "n_quicksand": 5,
        "speed": 15,
        "n_mines": 15
    }
    variable_params = None
    #variable_params={"enableDangerMarkers": [True, False]}
    batch_runner = BatchRunner(MinedZone, variable_params, fixed_params, iterations=n_iters,
        model_reporters={"Mean Time Spent in QS": lambda model: model.meanStepsQS,
                        "Steps Counter": lambda model: model.counter}
        )

    batch_runner.run_all()
    return batch_runner.get_model_vars_dataframe()
if __name__ == "__main__":
    batchRuns = False
    if not batchRuns:
        run_single_server()
    else:
        q2 = False
        q6 = False
        q7 = False
        df = run_batch()
        if q2 or q6:
            df_ = df["Steps Counter"]
            ax = df_.plot()
            for i,data in enumerate(df_):
                ax.text(i, data, "("+str(data)+")")
            plt.axhline(df_.mean(), color='r', linestyle='dashed', linewidth=1)
            ax.text(0, df_.mean(), "(Mean value : "+str(df_.mean())+")")
            plt.xlabel("Run")
            plt.ylabel("Value of last step (all mines removed)")
            plt.title("Simulations Demining")
            if q2:
                # Comment the part of code about Markers and variable params in BatchRunner
                plt.savefig("Q2_MeanTime_plot.png")
                plt.show()
            if q6:
                # Uncomment the part of code about Markers
                plt.savefig("Q6_MeanTime_plot.png")
                plt.show()

        if q7:
            # Uncomment the part of code about Markers and variable params 
            df = df.rename(columns={"enableDangerMarkers":"Markers DANGER"})
            df_ = df[["Mean Time Spent in QS", "Markers DANGER"]].astype(float)
            df_["Markers DANGER"] = df_["Markers DANGER"].astype(bool)
            df_.boxplot(by="Markers DANGER", column=["Mean Time Spent in QS"])
            plt.savefig("Q7_MeanTimeQS_BoxPlot.png")
            plt.show()

