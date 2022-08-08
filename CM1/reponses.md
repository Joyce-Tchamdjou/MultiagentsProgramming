##Compte rendu du TP IA310

Je n'ai jamais utilisé les list comprehension et les fonctions *lambda*, j'ai regardé de la documentation en ligne pour pouvoir les utiliser.  


**Question 1** : Pour modifier les agents, j'ai créé une liste avec tous les agents pour pouvoir accéder à ceux qui m'intéressent. Cela semble compatible avec la notion d'agent vue en cours car cette modification n'empêche en rien le fait que l'agent soit autonome, cela relève plutôt de l'interaction entre les agents.


**Question 2** : Ce système semble converger vers la stabilisation du nombre d'humains et de lycanthropes, après diminution et augmentation respectivement, dû à l'intervention de l'apothicaire et du chasseur. Cette stabilisation d'observe au bout d'une soixantaine de cycles environ.
La quantité d'agents de chaque espèce influe sur le résultat final, car plus il y'a d'apothicaires, plus il y'a de chance de guérison pour les lycans et donc leur nombre diminue; plus il y'a de chasseurs, plus il y'a de chances de tuer des loups transformés et donc moins d'attaques.

**Question 3** : Voir figure "Courbes_evol". Les courbes sont assez en accord avec les conjectures émises précédemment.

**Question 4** :  Voir figures "Courbe_moins_lycans_plus_pop", "Courbes_plus_cleric", "Courbes_plus_hunter", "Courbe_plus_lycans". Avec un nombre moins élevé de lycanthropes et plus d'humains sains, le progression est la même, elle est juste moins rapide que dans le cas de base. De plus, augmenter  le nombre d'apothicaires ou de chasseurs, freinent la progression des nombre de lycans et de loups transformés ensuite puisqu'il dépend fortement du premier. Enfin un trop grand nombre de lycanthropes conduit inévitablement vers un village où il y'a de moins en moins d'humains sains.

**Question 5** : Les paramètres qui auraient une influence sur le résultat de la simulation sont le nombre de d'apothicaires, le nombre de chasseurs et le nombre de lycans. Pour le dernier, plus il y'en a, plus la population va évoluer vers une prédominance de lycantropes.

**Question 6** : Un nombre d'apothicaires plus grand peut permettre de limiter le nombre de lycanthropes car on pourra en soigner un plus grand nombre et donc converger un état stationnaire sur le nombre de lycanthropes et de loups transformés.  

**Question 7** : Voir figure "batch_1". On observe que le nombre d'apothicaires influe plutôt dans le sens contraire de ce qui avait été supposé plus haut en ceci que les humains sains se font complètement dépasser par les lycanthropes. Ainsi,il faudrait soit beaucoup plus d'apothicaires que le seuil max fixé soit faire intervenir d'autres agents pour résoudre la situation.