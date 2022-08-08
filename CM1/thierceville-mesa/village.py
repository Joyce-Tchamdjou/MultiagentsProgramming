import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import uuid
import mesa
import numpy
import pandas
from mesa import space
from mesa.batchrunner import BatchRunner
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.visualization.ModularVisualization import ModularServer, VisualizationElement
from mesa.visualization.modules import ChartModule

class ContinuousCanvas(VisualizationElement):
    local_includes = [
        "./js/simple_continuous_canvas.js",
    ]

    def __init__(self, canvas_height=500,
                 canvas_width=500, instantiate=True):
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
                portrayal["x"] = ((obj.pos[0] - model.space.x_min) /
                                  (model.space.x_max - model.space.x_min))
                portrayal["y"] = ((obj.pos[1] - model.space.y_min) /
                                  (model.space.y_max - model.space.y_min))
            representation[portrayal["Layer"]].append(portrayal)
        return representation

def wander(x, y, speed, model):
    r = random.random() * math.pi * 2
    new_x = max(min(x + math.cos(r) * speed, model.space.x_max), model.space.x_min)
    new_y = max(min(y + math.sin(r) * speed, model.space.y_max), model.space.y_min)

    return new_x, new_y


class  Village(mesa.Model):
    def  __init__(self,  n_villagers, n_lycans, n_clerics, n_hunters):
        mesa.Model.__init__(self)
        self.space = mesa.space.ContinuousSpace(600, 600, False)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(model_reporters={"Humans":   lambda mod: len([agent for agent in mod.schedule.agents if agent.lycan == False]),\
                                                            "Lycans": lambda mod: len([agent for agent in mod.schedule.agents if agent.lycan == True]),\
                                                            "Werewolves": lambda mod: len([agent for agent in mod.schedule.agents if agent.lycan == True and agent.transformed == True]),\
                                                            "Population": lambda mod: len([agent for agent in mod.schedule.agents])})
        for  _  in  range(n_villagers):
            self.schedule.add(Villager(random.random()  *  600,  random.random()  *  600,  10, uuid.uuid1(), self, lycan=False))
        for  _  in  range(n_lycans):
            self.schedule.add(Villager(random.random()  *  600,  random.random()  *  600,  10, uuid.uuid1(), self, lycan=True))
        for  _  in  range(n_clerics):
            self.schedule.add(Cleric(random.random()  *  600,  random.random()  *  600,  10, uuid.uuid1(), self, lycan=False))
        for  _  in  range(n_hunters):
            self.schedule.add(Hunter(random.random()  *  600,  random.random()  *  600,  10, uuid.uuid1(), self, lycan=False))
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.schedule.steps >= 1000:
            self.running = False

class Villager(mesa.Agent):
    def __init__(self, x, y, speed, unique_id: int, model: Village, lycan, distance_attack=40, p_attack=0.6):
        super().__init__(unique_id, model)
        self.pos = (x, y)
        self.speed = speed
        self.model = model
        self.distance_attack = distance_attack
        self.p_attack = p_attack
        self.lycan = lycan
        self.transformed = False

    def portrayal_method(self):
        color = "blue"
        r = 3
        if self.lycan == True:
            color = "red"
            if self.transformed == True:
                r = 6
            
        
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": color,
                     "r": r}
        return portrayal

    def step(self):
        self.pos = wander(self.pos[0], self.pos[1], self.speed, self.model)
        # Transformation des lycans
        proba = random.random()
        if self.lycan == True and proba<0.1:
            self.transformed = True
        # Attaque des villageois à d=40
        if self.lycan == True:
            if self.transformed == True:
                victims = [agent for agent in self.model.schedule.agents if np.sqrt((agent.pos[0] - self.pos[0])**2 + (agent.pos[1] - self.pos[1])**2) <=self.distance_attack]
                for x in victims:
                    if x.lycan == False:
                        x.lycan = True

class Cleric(mesa.Agent):
    def __init__(self, x, y, speed, unique_id: int, model: Village, lycan, distance_heal = 30):
        super().__init__(unique_id, model)
        self.pos = (x, y)
        self.speed = speed
        self.model = model
        self.distance_heal = distance_heal
        self.lycan = lycan
        self.transformed = False

    def portrayal_method(self):
        color = "green"
        r = 3
        
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": color,
                     "r": r}
        return portrayal

    def step(self):
        self.pos = wander(self.pos[0], self.pos[1], self.speed, self.model)
        # Traitement des lycans à d=30
        patients = [agent for agent in self.model.schedule.agents if np.sqrt((agent.pos[0] - self.pos[0])**2 + (agent.pos[1] - self.pos[1])**2) <= self.distance_heal]
        for x in patients:
            if x.lycan == True: 
                if x.transformed == False:
                    x.lycan = False

class Hunter(mesa.Agent):
    def __init__(self, x, y, speed, unique_id: int, model: Village, lycan, distance_kill = 40):
        super().__init__(unique_id, model)
        self.pos = (x, y)
        self.speed = speed
        self.model = model
        self.distance_kill = distance_kill
        self.lycan = lycan
        self.transformed = False

    def portrayal_method(self):
        color = "black"
        r = 3
        
        portrayal = {"Shape": "circle",
                     "Filled": "true",
                     "Layer": 1,
                     "Color": color,
                     "r": r}
        return portrayal

    def step(self):
        self.pos = wander(self.pos[0], self.pos[1], self.speed, self.model)
        # Meurtre des lycans à d=40
        victimes = [agent for agent in self.model.schedule.agents if np.sqrt((agent.pos[0] - self.pos[0])**2 + (agent.pos[1] - self.pos[1])**2) <= self.distance_kill]

        for x in victimes:
            if x.lycan == True: 
                if x.transformed == True:
                    self.model.schedule.remove(x)

def run_single_server():
    chart = ChartModule([{"Label": "Humans", "Color": "Blue"},\
                        {"Label": "Lycans", "Color": "Red"},\
                        {"Label": "Werewolves", "Color": "Purple"},\
                        {"Label": "Population", "Color": "Yellow"}], data_collector_name="datacollector")

    n_villagers = mesa.visualization.ModularVisualization.UserSettableParameter('slider', "villager", 20, 0, 50, 1)
    n_lycans = mesa.visualization.ModularVisualization.UserSettableParameter('slider', "lycan", 5, 0, 50, 1)
    n_clerics = mesa.visualization.ModularVisualization.UserSettableParameter('slider', "cleric", 1, 0, 50, 1)
    n_hunters = mesa.visualization.ModularVisualization.UserSettableParameter('slider', "hunter", 2, 0, 50, 1)

    server  =  ModularServer(Village, [ContinuousCanvas(), chart],"Village",{"n_villagers":  n_villagers, "n_lycans": n_lycans, "n_clerics": n_clerics, "n_hunters": n_hunters})
    server.port = 8521
    server.launch()

def run_batch():
    params = {"n_villagers" : [50], "n_lycans" : [5], "n_clerics" : range(0, 6, 1), "n_hunters" : [1]}
    batch_run = BatchRunner(Village, params, model_reporters={"Humans":   lambda mod: len([agent for agent in mod.schedule.agents if agent.lycan == False]),\
                                                            "Lycans": lambda mod: len([agent for agent in mod.schedule.agents if agent.lycan == True]),\
                                                            "Werewolves": lambda mod: len([agent for agent in mod.schedule.agents if agent.lycan == True and agent.transformed == True]),\
                                                            "Population": lambda mod: len([agent for agent in mod.schedule.agents])})
    
    batch_run.run_all()
    batch_res = batch_run.get_model_vars_dataframe()
    batch_res[["Humans", "Lycans", "Werewolves", "Population"]].plot()
    plt.show()



if  __name__  ==  "__main__":
    #run_single_server()
    run_batch()
    