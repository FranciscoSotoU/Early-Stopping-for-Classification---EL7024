import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy_indexed as npi

@np.vectorize
def binarizar(x):
    return 1 if x >= 0.5 else 0

def mutual_info(conf_mat):
    '''Calcula la informacion mutua de una matriz de confusion
    --------------------------------
    Args:
      conf_mat: np.array
          Matriz de confusion
    
    Returns:
      float : Informacion mutua
    '''
    p_conj = (conf_mat/np.sum(conf_mat))
    p_x = np.sum(p_conj, axis=1)
    p_y = np.sum(p_conj, axis=0)
    p_x_y = p_conj
    suma = 0
    for i in range(2):
        for j in range(2):
            s = p_x_y[i,j] * np.log2(p_x_y[i,j]/(p_x[i]*p_y[j]))
            suma += s
    return suma


def training_curve(loss_train, loss_val, epochs, title,path):
  '''
  Grafica la curva de entrenamiento y las guarda en el directorio especificado
  --------------------------------
  Args:
      loss_train (list): Lista de perdidas de entrenamiento
      loss_val (list): Lista de perdidas de validacion
      epochs (int): Numero de epocas
      title (str): Titulo del grafico
      path (str): Directorio donde se guardara el grafico
  Returns:
      None

  '''
  ep = np.arange(1, epochs+1)
  plt.plot(ep, loss_train, label = "Train")
  plt.plot(ep, loss_val, label = "Validation")
  plt.legend()
  plt.title(title)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  fig_path = os.path.join(path, title+".png")
  plt.savefig(fig_path)
  plt.close()

def get_scores(losses, outputs, targets):
  '''
  Calcula la perdida y el accuracy
  --------------------------------
  
  Args:
      losses (list): Lista de perdidas
      outputs (torch.Tensor): Tensor de salidas
      targets (torch.Tensor): Tensor de targets

  Returns:
      float: Perdida promedio
      float: Accuracy
  '''
  out = binarizar(outputs.cpu().numpy())
  t = targets.cpu().numpy()
  cant = targets.shape[0]

  return torch.sum(losses).item()/cant, accuracy_score(t, out)

def get_R(outputs, targets):
  '''
  Calcula la matriz de residuos
  --------------------------------
  Args:
      outputs (np.array): Array de salidas
      targets (np.array): Array de targets
  
  Returns:
      np.array: Matriz de residuos'''
  Rs = []
  for epoch in range(len(targets)):
      y = targets[epoch]
      y_hat = binarizar(outputs[epoch])
      Rs.append(np.logical_xor(y, y_hat))
  return np.array(Rs)

def get_MI(outputs, targets):
  '''
  Calcula la informacion mutua entre las salidas y los targets
  --------------------------------
  Args:
      outputs (np.array): Array de salidas
      targets (np.array): Array de targets
  
  Returns:
      np.array: Array de informacion mutua'''
  cms = []
  for epoch in range(len(targets)):
      cm = confusion_matrix(binarizar(targets[epoch]), binarizar(outputs[epoch]))
      cms.append(cm/np.sum(cm))

  mi = []
  for epoch in range(len(cms)):
      mi.append(mutual_info(cms[epoch]))
  
  return np.array(mi)

def get_MI_XR(residuo, data):
  '''
  Calcula la informacion mutua entre los residuos y los inputs
  --------------------------------
  Args:
      residuo (np.array): Array de residuos
      data (np.array): Array de inputs
  
  Returns:
      np.array: Array de informacion mutua'''
  mi_r_x = []
  for epoch in range(len(data)):
      mi_r_x.append(estimador(residuo[epoch],data[epoch] ))
  return np.array(mi_r_x)

def plot_MI(MIs,title,epochs,path):
  '''
  Grafica la informacion mutua y la guarda en el directorio especificado
  --------------------------------
  Args:
      MIs (list): Lista de informacion mutua
      title (str): Titulo del grafico
      epochs (int): Numero de epocas
      path (str): Directorio donde se guardara el grafico
      
  Returns:
      None
  '''
  ep = np.arange(1, epochs+1)
  plt.scatter(ep,MIs)
  plt.title(title)
  plt.xlabel("Epochs")
  plt.ylabel("MI[bits]")
  fig_path = os.path.join(path, title+".png")
  plt.savefig(fig_path)
  plt.close()

def estimador(X,Y):
  '''
  Calcula la informacion mutua entre dos variables
  --------------------------------
  Args:
      X (np.array): Array de la primera variable
      Y (np.array): Array de la segunda variable
  
  Returns:
      float: Informacion mutua'''

  x_comp, x_num =  npi.count(X)
  y_comp, y_num = npi.count(Y)
  XY = np.concatenate((X, Y), axis=1)
  xy_comp, xy_num = npi.count(XY)
  IM = 0

  for x in x_comp:
    for y in y_comp:

      xy = np.concatenate((x, y))
      #print('xy:',xy)
      index_xy = find_row_index(xy_comp,xy)
      #print('index_xy:',index_xy)
      pxy = xy_num[index_xy]/np.sum(xy_num)

      index_x = find_row_index(x_comp,x)
      px =  x_num[index_x]/np.sum(x_num)

      index_y = find_row_index(y_comp,y)
      py =  y_num[index_y]/np.sum(y_num)
      IM += pxy * np.log2(pxy/(px*py))
  return IM

def find_row_index(matrix, target_row):
    '''
    Busca una fila en una matriz
    --------------------------------
    Args:
        matrix (np.array): Matriz
        target_row (np.array): Fila a buscar
    
    Returns:
        int: Indice de la fila buscada
    '''
    for i, row in enumerate(matrix):
        if np.array_equal(row, target_row):
            return i
    return -1