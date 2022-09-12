import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import curve_fit
import os 
from scipy import stats
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from statistics import mean
import pandas as pd
from scipy.stats import spearmanr,pearsonr
import sklearn
from scipy.stats import gaussian_kde
import argparse
import time
from tqdm import tqdm
import matplotlib.font_manager as fm
import matplotlib









def read_files(path_to_file):
  df = pd.read_csv(path_to_file)
  values = df.values.tolist()
  mos =[]
  predicted = []
  for i in values:
    mos.append(i[0])
    predicted.append(i[1])
  return mos, predicted


def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat

  


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--mos_pred',  type=str, help='path tp mos vs predicted scores file')

  args = parser.parse_args()

  if not os.path.exists('./figures'):
    os.makedirs('./figures')

  predicted_file = args.mos_pred

  y_ss, y_p = read_files(predicted_file)
 


  beta_init = [np.max(y_ss), np.min(y_ss), np.mean(y_p), 0.5]
  popt, _ = curve_fit(logistic_func, y_p, y_ss, p0=beta_init, maxfev=int(1e8))
  
  y_pred_logistic = logistic_func(y_p, *popt)
  xy = np.vstack([y_ss,y_p])
  z = gaussian_kde(xy)(xy)
  min_mos = min(y_ss)
  max_mos = max(y_ss)
  min_pred = min(y_p)
  max_pred = max(y_p)  
  pas = (max_mos - min_mos)/5
  pas1 = (max_pred - min_pred)/5
  m = min(y_p) - pas1
  l = len(y_p)
  u = max(y_p) +pas1
  x = np.linspace(m-0.2,u+0.2,num=l)
  ms = y_ss
  kf = ms - y_pred_logistic

  
  sig = np.std(kf)

  

  
  print('SROCC = ',spearmanr(y_ss,y_p).correlation)
  print('======================================================')

  print('PLCC = ', stats.pearsonr(y_ss,y_pred_logistic)[0])
  print('======================================================')

  try:
    KRCC = stats.kendalltau(y_ss, y_p)[0]
  except:
    KRCC = stats.kendalltau(y_ss, y_p, method='asymptotic')[0]
  print('KROCC = ' , KRCC)
  print('======================================================')

  
  
  print('RMSE = ' , np.sqrt(mean_squared_error(y_ss,y_pred_logistic)))
  print('======================================================')


  fig = plt.figure()
  
  ax = fig.add_subplot(1, 1, 1)
  font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
  font2 = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
  fondt = fm.FontProperties(family='serif',
                                   weight='normal',
                                   style='normal', size=11)
  fondtitle = fm.FontProperties(family='serif',
                                   weight='normal',
                                   style='normal', size=8)




  
  ax.set_ylim([min_mos-pas,max_mos+pas])
  ax.set_xlim([min_pred-pas1,max_pred+pas1])



  plt.scatter(y_p,y_ss, s=10, marker='o', c=z)

  plt.plot(x, logistic_func(x, *popt), c='red',label=r'fitted $f(x)$',linewidth=1)
  plt.plot(x, logistic_func(x, *popt)+ 2*sig,'--' , c='red',label=r'$f(x) \pm  2  \sigma$',linewidth=1)
  plt.plot(x, logistic_func(x, *popt)- 2*sig,'--' , c='red',linewidth=1)
  plt.xlabel("Predicted Score",fontdict=font)
  plt.ylabel("MOS",fontdict=font)
  plt.legend(prop=fondt)
  plt.title('MOS vs predicted score', fontdict=font2)
  
  plt.grid(which='both')
  plt.grid(which='minor', alpha=0.2)
  plt.grid(which='major', alpha=0.5)
  
  plt.savefig('./figures/mos_sroc =' + str(spearmanr(y_ss,y_p).correlation)+'.png')
  plt.show()
  
  
