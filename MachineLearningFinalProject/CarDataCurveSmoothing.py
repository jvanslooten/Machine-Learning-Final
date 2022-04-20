import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg

# Set up dataset
data = pd.read_csv('/Users/jvanslooten/Desktop/Machine Learning/MachineLearningFinalProject/WeightMPGLinearRegression '
                   'copy.csv')
x = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values.reshape(-1, 1)

#Smooth the curve
kr = KernelReg(y,x,'c')
plt.plot(x, y, '+')
y_pred, y_std = kr.fit(x)

plt.plot(x, y_pred)
plt.show()

