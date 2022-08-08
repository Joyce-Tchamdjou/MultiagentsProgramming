
import random
from re import T
import numpy as np

import uuid
import mesa
import pandas
from mesa import space
from mesa.batchrunner import BatchRunner
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from matplotlib import pyplot as plt
from scipy.stats import truncnorm

def get_truncated_noise(truncation=0.5):
    truncated_noise = truncnorm.rvs(-truncation, truncation, size=(1, 1))
    return truncated_noise+0.5

"""
DESCRIPTION :
-----------

- un marché est divisé entre 'nb_investisseurs' agents qui ont aussi de l'argent en liquide
- les agents ont une certaine tendance à acheter (vendre) même si tous les facteurs l'encouragent à acheter (vendre).
dans le cas contraire ils ne font aucune opération.
- un indice (entre 0 et 1/ 0 : vendre | 1 : acheter) donne une intuition financière de la tendance du marché et de si il faut acheter ou vendre (ou attendre)
- cet indice et combiné linéairement à un bruit gaussien qui ajoute une certaine stochasticité dans la prédiction
- l'agent prend en compte sa tendance, l'indice et le bruit pour faire une opération. Il s'assure qu'il a les moyens de le faire.
- le prix de l'action est calculé suite aux actions des agents de telle facon qu'il s'ajuste à l'offre et la demande.

TO DO :
-----

- améliorer l'index (IMP)
- s'assurer que chaque acheteur achete de qqun et chaque vendeur vend à qqun
- accepter des nouveaux entrants au marché
- comparer différents jeux de paramètres
"""

ARGENT_INVESTISSEUR = 100000 # combien de cash ils ont à part la part du marché initiale
TENDANCE_ACHAT = 0.7 # [0, 1]
TENDANCE_VENTE = 0.7 # [0, 1]
PROPORTION_ACHAT = [0.01, 0.1] # ACHETE AVEC UNE PROPORTION RANDOM DE SON PORTEFEUIL ENTRE CES DEUX VALEURS
PROPORTION_VENTE = [0.8, 1] # VEND AVEC UNE PROPORTION RANDOM DE SES ACTIONS ENTRE CES DEUX VALEURS
STOCHASTICITY = 0.2 # importance of exploration Vs exploitation

class  Marche(mesa.Model):
    """
    nb_investisseurs : nombre de traders

    nb_actions : en combien d'action est divisée l'entreprise en question

    prix_action : prix d'une action
    
    index : indicateur financier qui permet de prendre une décision d'achat ou de vente (entre 0 et 1)
    """
    def  __init__(self,  nb_investisseurs : int, nb_actions : int, prix_action : float):
        mesa.Model.__init__(self)

        self.nb_investisseurs = nb_investisseurs
        self.nb_actions = nb_actions
        self.prix_action = prix_action
        self.ema = prix_action # exponential moving average

        self.schedule = RandomActivation(self)
        self.dc = DataCollector(model_reporters={"Share Price" : lambda m: m.prix_action,\
                                                "Exponential Moving Average (EMA)" : lambda m: m.ema})

        
        # devide the market between all investers who become share-holders
        parts_marche = np.random.dirichlet(np.ones(nb_investisseurs),size=1)[0]
        for i  in  range(nb_investisseurs):
            part_marche = parts_marche[i]
            self.schedule.add(Investisseur(uuid.uuid1(), self, part_marche, ARGENT_INVESTISSEUR, TENDANCE_ACHAT, TENDANCE_VENTE))


        self.all_prices = [self.prix_action]
        self.index = 0.5


    def calcul_prix(self):

        if self.prix_action == 0:
            return 0 # if action went bankrupt it stays this way

        total_achat = 0
        total_vente = 0

        buyers = [agent for agent in self.schedule.agent_buffer() if agent.action == 1]
        sellers = [agent for agent in self.schedule.agent_buffer() if agent.action == -1]

        for agent in buyers:
            total_achat += agent.nb_actions_achetées * self.prix_action
            
        for agent in sellers:
            total_vente -= agent.nb_actions_vendues * self.prix_action

        # correction of price depending on supply and demand
        new_prix = self.prix_action + ( (total_achat - total_vente) / self.nb_actions )

        if new_prix < 0:
            return 0 # no negative price allowed

        return new_prix

    def calcul_index(self, facteur):
        df_prices = pandas.DataFrame({'All Prices' : self.all_prices})

        # prediction of price 
        ema = df_prices.ewm(facteur).mean()
        ema_last = ema['All Prices'].iloc[-1]
      
        # [0, 1]  < 0.5 sell | > 0.5 buy 
        idx = (((ema_last - self.prix_action) / self.prix_action) + 1) / 2
        return ema_last, idx



    def step(self):
        self.schedule.step() 
        self.dc.collect(self)

        # calcul prix : prix t, les actions les gens (offre et demande) => prix t+1
        self.prix_action = self.calcul_prix()
        print("PRIX Action = ", self.prix_action)
        # modify self.all_prices
        self.all_prices.append(self.prix_action)
        # calcul index : prix => tendance (montant, descendant, stable)
        self.ema, self.index = self.calcul_index(0.8)

        
        for agent in self.schedule.agent_buffer():
            agent.action = 0 # 0 : hold | 1 : acheter | -1 : vendre
            agent.nb_actions_achetées = 0
            agent.nb_actions_vendues = 0

        if self.schedule.steps >= 1000:
            self.running = False



class Investisseur(mesa.Agent):
    def __init__(self, unique_id: int, model: Marche, part_marche : float, portefeuil : float, proba_teandance_achat : float, proba_teandance_vente : float):
        super().__init__(unique_id, model)
        
        self.id = unique_id
        self.model = model
        self.portefeuil = portefeuil
        self.proba_tendance_achat = proba_teandance_achat
        self.proba_tendance_vente = proba_teandance_vente

        self.qte_action = part_marche*self.model.nb_actions # selon la part du marché qui lui est donnée
        
        # pour 1 tour
        self.action = 0 # 0 : hold | 1 : acheter | -1 : vendre
        self.nb_actions_achetées = 0
        self.nb_actions_vendues = 0


    def step(self, exploration_rate=STOCHASTICITY):
        # proba_achat + index + gaussienne = > decision

        idx = self.model.index
        rdm = get_truncated_noise()[0][0]

        # result = exploration_rate*rdm + (1 - exploration_rate)*idx # < 0.5 vendre | > 0.5 acheter
        result = rdm

        # acheter
        if result > 0.5:
            decision = random.uniform(0, 1)
            if decision <= self.proba_tendance_achat:
                self.action = 1 # acheter
                # verifier que argent dans portefeuille > 0
                if self.portefeuil > 0:
                    # decider qte
                    prop_buy = random.uniform(PROPORTION_ACHAT[0], PROPORTION_ACHAT[1])
                    # nb_action achetees
                    self.nb_actions_achetées = (prop_buy*self.portefeuil)/self.model.prix_action
                    self.qte_action += self.nb_actions_achetées
                    # modifier portfeuil (-= nb_action_achetées*prix_action) 
                    self.portefeuil -= self.nb_actions_achetées*self.model.prix_action
                else:
                    self.action = 0 # renoncer à l'achat

            else: 
                self.action = 0 # renoncer à l'achat

        # vendre
        else:
            decision = random.uniform(0, 1)
            if decision <= self.proba_tendance_vente:
                self.action = -1 # vendre
                # verifier qute qte action >0
                if self.qte_action > 0:
                    # decider qte
                    prop_sell = random.uniform(PROPORTION_VENTE[0], PROPORTION_VENTE[1])
                    # nb_action vendues
                    self.nb_actions_vendues = (prop_sell*(self.qte_action*self.model.prix_action))/self.model.prix_action
                    self.qte_action -= self.nb_actions_vendues
                    # modifier portfeuil (+= nb_action_vendues*prix_action) 
                    self.portefeuil += self.nb_actions_vendues*self.model.prix_action
                else:
                    self.action = 0 # renoncer à la vente
            else:
                self.action = 0 # renoncer à la vente

        print("ACTION of ", self.id, " : ", self.action)


                    
def run_single_server():
    # chart
    chart = ChartModule([{"Label": "Share Price", "Color": "Green"}, {"Label": "Exponential Moving Average (EMA)", "Color": "Red"} ],
                        data_collector_name='dc')

    # sliders          
    nb_investisseurs = mesa.visualization.ModularVisualization.UserSettableParameter('slider', "Number of Traders", 100, 10, 500, 10)
    nb_actions = mesa.visualization.ModularVisualization.UserSettableParameter('slider', "Number of Shares", 10000, 1000, 100000, 1000)
    prix_action = mesa.visualization.ModularVisualization.UserSettableParameter('slider', "Share Price", 50, 0, 500, 50)

    server  =  ModularServer(Marche, [chart], "Market",{"nb_investisseurs":  nb_investisseurs, "nb_actions" : nb_actions, "prix_action" : prix_action})
    server.port = 8500
    server.launch()

# def run_batch():
#     input_values = {"n_Investisseurs" : [50], "n_loupgarou" : [5], "n_cleric" : range(0, 6, 1), "n_hunter" : [1]}

#     batch_run = BatchRunner(Village, input_values, model_reporters={"Investisseurs Count" : lambda m: len([u for u in m.schedule.agent_buffer() if u.is_attackable and not u.loup_garou]),\
#                                                                     "Wolfs Count" : lambda m: len([u for u in m.schedule.agent_buffer() if u.is_attackable and u.loup_garou]),\
#                                                                     "WereWolfs Count" : lambda m: len([u for u in m.schedule.agent_buffer() if u.is_attackable and u.is_transformed]),\
#                                                                     "All Agents" : lambda m: m.schedule.get_agent_count() })

#     batch_run.run_all()
#     batch_df = batch_run.get_model_vars_dataframe()
#     batch_df[["All Agents", "Investisseurs Count", "WereWolfs Count", "Wolfs Count"]].plot()
#     plt.show()


if  __name__  ==  "__main__":
    # CHOOSE ONE OF THESE OPTIONS
    run_single_server()
    # run_batch()