"""

Title: Machine Learning Final Projet
Author: Jack VanSlooten
Date: 8 December 2019

"""
import pandas as pd

csvfile = pd.read_csv('/Users/jvanslooten/Desktop/Machine Learning/MachineLearningFinalProject/CarSampleDataMLFinal '
                      'copy.csv')

i = 1
while i <= 40:
    print(csvfile.sample())
    i = i + 1
