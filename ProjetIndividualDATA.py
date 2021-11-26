#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 19:25:39 2021

@author: Valtair
"""

import pandas as pd 
import numpy as np
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns 
import random
from datetime import datetime as dt
from pylab import array
from pylab import plot

data=pd.read_excel('/Users/Valtair/Desktop/Msc Finance/Python/spyder-py3/Individual project python.xlsx',sheet_name='Euro')
Date=data['Date']
print(Date)



#Même chose pour avoir la somme des arrivées 
#print(PSGA)
#Somme = 0
#for x in range(22):
#   Somme += PSGA[x]
#print(Somme)



# ARRIVE
data['PSGA']=data['PSG.A']
PSG_A=data['PSGA'].dropna()
print('the details on 21 years of each arrival for PSG :', PSG_A)
data['BARCA_A']=data['BARCA.A']
BARCA_A=data['BARCA_A'].dropna()
print('the details on 21 years of each arrival Barca :', BARCA_A)
data['REAL_A']=data['REAL.A']
REAL_A=data['REAL_A'].dropna()
print('the details on 21 years of each arrival for Real :', REAL_A)
data['BAYERN_A']=data['BAYERN.A']
BAYERN_A=data['BAYERN_A'].dropna()
print('the details on 21 years of each arrival for Bayern :', BAYERN_A)
data['MANU_A']=data['MANU.A']
MANU_A=data['MANU_A'].dropna()
print('the details on 21 years of each arrival for Man U :', MANU_A)
data['MANC_A']=data['MANC.A']
MANC_A=data['MANC_A'].dropna()
print('the details on 21 years of each arrival for Man C :', MANC_A)
data['LIVER_A']=data['LIVER.A']
LIVER_A=data['LIVER_A'].dropna()
print('the details on 21 years of each arrival for Liverpool :', LIVER_A)
data['JUV_A']=data['JUV.A']
JUV_A=data['JUV_A'].dropna()
print('the details on 21 years of each arrival for Juventus :', JUV_A)

SommeA_PSG = PSG_A.sum()
print('The sum of PSG arrival is', SommeA_PSG)
SommeA_BARCA = BARCA_A.sum()
print('The sum of Barcolona arrival is', SommeA_BARCA)
SommeA_REAL = REAL_A.sum()
print('The sum of Real de Madrid arrival is', SommeA_REAL)
SommeA_BAYERN = BAYERN_A.sum()
print('The sum of Bayern de Munich arrival is', SommeA_BAYERN)
SommeA_MANU = MANU_A.sum()
print('The sum of Man U arrival is', SommeA_MANU)
SommeA_MANC = MANC_A.sum()
print('The sum of Man C arrival is', SommeA_MANC)
SommeA_LIVER = LIVER_A.sum()
print('The sum of Liverpool arrival is', SommeA_LIVER)
SommeA_JUV = JUV_A.sum()
print('The sum of Juventus arrival is', SommeA_JUV)

mPSG_A=PSG_A.mean()
print('The mean of PSG for the total of arrival is', mPSG_A)
mBARCA_A=BARCA_A.mean()
print('The mean of Barcelona for the total of arrival is', mBARCA_A)
mREAL_A=REAL_A.mean()
print('The mean of Real de Madrid for the total of arrival is', mREAL_A)
mBAYERN_A=BAYERN_A.mean()
print('The mean of Bayern de Munich for the total of arrival is', mBAYERN_A)
mMANU_A=MANU_A.mean()
print('The mean of Man U for the total of arrival is', mMANU_A)
mMANC_A=MANC_A.mean()
print('The mean of Man C for the total of arrival is', mMANC_A)
mLIVER_A=LIVER_A.mean()
print('The mean of Liverpool for the total of arrival is', mLIVER_A)
mJUV_A=JUV_A.mean()
print('The mean of Juventus for the total of arrival is', mJUV_A)



# DEPART


data['PSG_D']=data['PSG.D']
PSG_D=data['PSG_D'].dropna()
print('the details on 21 years of each departure for PSG :', PSG_D)
data['BARCA_D']=data['BARCA.D']
BARCA_D=data['BARCA_D'].dropna()
print('the details on 21 years of each departure for Barca :', BARCA_D)
data['REAL_D']=data['REAL.D']
REAL_D=data['REAL_D'].dropna()
print('the details on 21 years of each departure for Real :', REAL_D)
data['BAYERN_D']=data['BAYERN.D']
BAYERN_D=data['BAYERN_D'].dropna()
print('the details on 21 years of each departure for Bayern :', BAYERN_D)
data['MANU_D']=data['MANU.D']
MANU_D=data['MANU_D'].dropna()
print('the details on 21 years of each departure for Man U :', MANU_D)
data['MANC_D']=data['MANC.D']
MANC_D=data['MANC_D'].dropna()
print('the details on 21 years of each departure for Man C :', MANC_D)
data['LIVER_D']=data['LIVER.D']
LIVER_D=data['LIVER_D'].dropna()
print('the details on 21 years of each departure for Liverpool :', LIVER_D)
data['JUV_D']=data['JUV.D']
JUV_D=data['JUV_D'].dropna()
print('the details on 21 years of each departure for Juventus :', JUV_D)

SommeD_PSG = PSG_D.sum()
print('The sum of PSG departure is', SommeD_PSG)
SommeD_BARCA = BARCA_D.sum()
print('The sum of Barcolona departure is', SommeD_BARCA)
SommeD_REAL = REAL_D.sum()
print('The sum of Real de Madrid departure is', SommeD_REAL)
SommeD_BAYERN = BAYERN_D.sum()
print('The sum of Bayern de Munich departure is', SommeD_BAYERN)
SommeD_MANU = MANU_D.sum()
print('The sum of Man U departure is', SommeD_MANU)
SommeD_MANC = MANC_D.sum()
print('The sum of Man C departure is', SommeD_MANC)
SommeD_LIVER = LIVER_D.sum()
print('The sum of Liverpool departure is', SommeD_LIVER)
SommeD_JUV = JUV_D.sum()
print('The sum of Juventus departure is', SommeD_JUV)

mPSG_D=PSG_D.mean()
print('The mean of PSG for the total of departure is', mPSG_D)
mBARCA_D=BARCA_D.mean()
print('The mean of Barcelona for the total of departure is', mBARCA_D)
mREAL_D=REAL_D.mean()
print('The mean of Real de Madrid for the total of departure is', mREAL_D)
mBAYERN_D=BAYERN_D.mean()
print('The mean of Bayern de Munich for the total of departure is', mBAYERN_D)
mMANU_D=MANU_D.mean()
print('The mean of Man U for the total of departure is', mMANU_D)
mMANC_D=MANC_D.mean()
print('The mean of Man C for the total of departure is', mMANC_D)
mLIVER_D=LIVER_D.mean()
print('The mean of Liverpool for the total of departure is', mLIVER_D)
mJUV_D=JUV_D.mean()
print('The mean of Juventus for the total of departure is', mJUV_D)



# LOSS OR PROFIT

data['L&PPSG']=-data['PSG.A']+data['PSG.D']
LPPSG=data['L&PPSG'].dropna() 
print(LPPSG)
TTL_LP_PSG = LPPSG.sum()
print(TTL_LP_PSG)
mLP_PSG = LPPSG.mean()
print(mLP_PSG)

data['L&PBARCA']=-data['BARCA.A']+data['BARCA.D']
LPBARCA=data['L&PBARCA'].dropna() 
print(LPBARCA)
TTL_LP_BARCA = LPBARCA.sum()
print(TTL_LP_BARCA)
mLP_BARCA = LPBARCA.mean()
print(mLP_BARCA)

data['L&PREAL']=-data['REAL.A']+data['REAL.D']
LPREAL=data['L&PREAL'].dropna() 
print(LPREAL)
TTL_LP_REAL = LPREAL.sum()
print(TTL_LP_REAL)
mLP_REAL = LPREAL.mean()
print(mLP_REAL)

data['L&PBAYERN']=-data['BAYERN.A']+data['BAYERN.D']
LPBAYERN=data['L&PBAYERN'].dropna() 
print(LPBAYERN)
TTL_LP_BAYERN = LPBAYERN.sum()
print(TTL_LP_BAYERN)
mLP_BAYERN = LPBAYERN.mean()
print(mLP_BAYERN)

data['L&PMANU']=-data['MANU.A']+data['MANU.D']
LPMANU=data['L&PMANU'].dropna() 
print(LPMANU)
TTL_LP_MANU = LPMANU.sum()
print(TTL_LP_MANU)
mLP_MANU = LPMANU.mean()
print(mLP_MANU)

data['L&PMANC']=-data['MANC.A']+data['MANC.D']
LPMANC=data['L&PMANC'].dropna() 
print(LPMANC)
TTL_LP_MANC = LPMANC.sum()
print(TTL_LP_MANC)
mLP_MANC = LPMANC.mean()
print(mLP_MANC)

data['L&PLIVER']=-data['LIVER.A']+data['LIVER.D']
LPLIVER=data['L&PLIVER'].dropna() 
print(LPLIVER)
TTL_LP_LIVER = LPLIVER.sum()
print(TTL_LP_LIVER)
mLP_LIVER = LPLIVER.mean()
print(mLP_LIVER)

data['L&PJUV']=-data['JUV.A']+data['JUV.D']
LPJUV=data['L&PJUV'].dropna() 
print(LPJUV)
TTL_LP_JUV = LPJUV.sum()
print(TTL_LP_JUV)
mLP_JUV = LPJUV.mean()
print(mLP_JUV)


TTL = TTL_LP_PSG + TTL_LP_BARCA + TTL_LP_BAYERN + TTL_LP_JUV + TTL_LP_LIVER + TTL_LP_MANC + TTL_LP_MANU + TTL_LP_REAL
print(TTL)

x_PSG = (TTL_LP_PSG / TTL)*100
print('Deficit after the mercato for PSG :', x_PSG)
x_BARCA = (TTL_LP_BARCA / TTL)*100
print('Deficit after the mercato for BARCA:',x_BARCA)
x_REAL = (TTL_LP_REAL / TTL)*100
print('Deficit after the mercato for REAL :',x_REAL)
x_BAYERN = (TTL_LP_BAYERN / TTL)*100
print('Deficit after the mercato for BAYERN :',x_BAYERN)
x_MANU = (TTL_LP_MANU / TTL)*100
print('Deficit after the mercato for MAN U :',x_MANU)
x_MANC = (TTL_LP_MANC / TTL)*100
print('Deficit after the mercato for MAN C :',x_MANC)
x_LIVER = (TTL_LP_LIVER / TTL)*100
print('Deficit after the mercato for LIVERPOOL:',x_LIVER)
x_JUV = (TTL_LP_JUV / TTL)*100
print('Deficit after the mercato for JUVENTUS:',x_JUV)


####GRAPHIQUE#####

#################################
#1er tableau avec tous les détails 
#file = "/Users/Valtair/Desktop/Msc Finance/Python/spyder-py3/Individual project python.xlsx"
#sheet = 'Euro'
#x1 = pd.ExcelFile(file)
#Foot_x1 = x1.parse(sheet, index_col="Date")


#################################
#Tableau qui montre la différence entre les arrivées et les départs 
#labels = ['P', 'B', 'R', 'BAY', 'MU', 'MC','L','J']
#Arrival = [SommeA_PSG, SommeA_BARCA, SommeA_REAL, SommeA_BAYERN, SommeA_MANU, SommeA_MANC, SommeA_LIVER, SommeA_JUV]
#Departure = [SommeD_PSG, SommeD_BARCA, SommeD_REAL, SommeD_BAYERN, SommeD_MANU, SommeD_MANC, SommeD_LIVER, SommeD_JUV]

#x = np.arange(len(labels))  
#width = 0.35  

#fig, x2 = plt.subplots()
#rects1 = x2.bar(x - width/2, Arrival, width, label='Arrivals')
#rects2 = x2.bar(x + width/2, Departure, width, label='Departures')


#x2.set_ylabel('Euro in billion')
#x2.set_title('The real aspect of the Mercato from 2000')
#x2.set_xticks(x, labels)
#x2.legend()

#x2.bar_label(rects1, padding=1)
#x2.bar_label(rects2, padding=1)

#fig.tight_layout()


##################################
#GRAPHIQUE A UTILISER - LES BULLES

browser_market_share = {
    'browsers': ['PSG 1.02B', 'Barca 0.89B', 'Réal 0.89B', 'Bayern 0.65B', 'Man U 1.26B', 'Man C 1.6B','Liver 0.58B','Juv 0.61B'],
    'market_share': [x_PSG, x_BARCA, x_REAL, x_BAYERN, x_MANU, x_MANC, x_LIVER, x_JUV],
    'color': ['#5A69AF', '#702200', '#F9C784', '#FC944A', '#ff4d00', '#59dce3', '#7C5AAF', '#87AF5A' ]
}


class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - \
            bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):
       
        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors):
        
        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')


bubble_chart = BubbleChart(area=browser_market_share['market_share'],
                           bubble_spacing=0.1)

bubble_chart.collapse()

fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
bubble_chart.plot(
    ax, browser_market_share['browsers'], browser_market_share['color'])
ax.axis("off")
ax.relim()
ax.autoscale_view()
ax.set_title('Clubs deficit after the Mercato from 2000 to 2021')

plt.show()


#plt.savefig("Clubs.png",dpi=200)





##################################
#x = Date
#y = [PSG_A]
#y = array([-100000000, -50000000, -10000000, -5000000,0, 5000000, 10000000, 15000000, 20000000])



#plt.plot(x,y)


#PSGA = np.arrange(PSG_A)
#columns = ['Arrival']
#index = [Date]

#df = pd.DataFrame(data=PSGA,index=index,columns=columns)



#file = "/Users/Valtair/Desktop/Msc Finance/Python/spyder-py3/Individual project python.xlsx"
#sheet = 'Euro'

#x1 = pd.ExcelFile(file)
#Foot_x1 = x1.parse(sheet, index_col="Date")





