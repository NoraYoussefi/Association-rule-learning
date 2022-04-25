# -*- coding: utf-8 -*-

#changement de dossier
import os
os.chdir("D:/DataMining/Databases_for_mining/dataset_for_soft_dev_and_comparison/assoc/mlxtend")

#importation des données
import pandas
D = pandas.read_table("market_basket.txt",delimiter="\t",header=0)

#10 premières lignes
print(D.head(10))

#vérification des dimensions
print(D.shape)

#tableau croisé 0/1
TC = pandas.crosstab(D.ID,D.Product)
print(TC.iloc[:20,:3])

#dimensions
print(TC.shape)

#liste des noms de produits
print(TC.columns)

#importation de la fonction apriori
from mlxtend.frequent_patterns import apriori

#itemsets frequents
freq_itemsets = apriori(TC,min_support=0.025,max_len=4,use_colnames=True)

#type -> pandas DataFrame
type(freq_itemsets)

#liste des colonnes
print(freq_itemsets.columns)

#nombre d'itemsets
print(freq_itemsets.shape)

#affichage des 15 premiers itemsets
print(freq_itemsets.head(15))

#type du champ 'itemsets'
print(type(freq_itemsets.itemsets))

#accès indexé au premier élément
print(freq_itemsets.itemsets[0])

#fonction de test d'inclusion
def is_inclus(x,items):
    return items.issubset(x)

#recherche des index des itemsets correspondant à une condition
import numpy
id = numpy.where(freq_itemsets.itemsets.apply(is_inclus,items={'Aspirin'}))
print(id)

#affichage des itemsets corresp.
print(freq_itemsets.loc[id])

#passer par une fonction lambda si on est préssé
numpy.where(freq_itemsets.itemsets.apply(lambda x,ensemble:ensemble.issubset(x),ensemble={'Aspirin'}))

#itemsets contenant Aspirin - passer par les méthodes natives de Series
print(freq_itemsets[freq_itemsets['itemsets'].ge({'Aspirin'})])

#itemsets avec Aspirin
print(freq_itemsets[freq_itemsets['itemsets'].eq({'Aspirin'})])

#itemsets contenant Aspirin et Eggs
print(freq_itemsets[freq_itemsets['itemsets'].ge({'Aspirin','Eggs'})])

#itemsets contenant Aspirin et Eggs
print(freq_itemsets[freq_itemsets['itemsets'].ge({'Eggs','Aspirin'})])

#fonction de calcul des règles
from mlxtend.frequent_patterns import association_rules

#génération des règles à partir des itemsets fréquents
regles = association_rules(freq_itemsets,metric="confidence",min_threshold=0.75)

#type de l'objet renvoyé
print(type(regles))

#dimension
print(regles.shape)

#liste des colonnes
print(regles.columns)

#5 "premières" règles
print(regles.iloc[:5,:])

#règles en restrieignant l'affichage à qqs colonnes
myRegles = regles.loc[:,['antecedents','consequents','lift']]
print(myRegles.shape)

#pour afficher toutes les colonnes
pandas.set_option('display.max_columns',5)
pandas.set_option('precision',3)

#affichage des 5 premières règles
print(myRegles[:5])

#affichage des règles avec un LIFT supérieur ou égal à 7
print(myRegles[myRegles['lift'].ge(7.0)])

#trier les règles dans l'ordre du lift décroissants - 10 meilleurs règles
print(myRegles.sort_values(by='lift',ascending=False)[:10])

#filtrer les règles menant à 2pct_milk
print(myRegles[myRegles['consequents'].eq({'2pct_Milk'})])

#filtrer les règles contenant 'Aspirin' dans l'antécédent
print(myRegles[myRegles['antecedents'].ge({'Aspirin'})])
