import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy_indexed as npi
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import itertools
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


def training_curve(loss_train, loss_val, epochs,overfit, title,path):
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
    plt.axvline(x = overfit, color = 'b', label = 'overfitting')
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
  out = outputs.cpu().numpy()
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
            y_hat = outputs[epoch].reshape(-1)
            Rs.append(np.abs(y_hat-y ))
    return np.array(Rs)

def H(X):
    """
    This function calculates the entropy for binary variables
    """
    N = len(X)
    p = np.sum(X)/N
    if (p == 0) or (p == 1):
        return 0
    return p * np.log2((1-p)/p) - np.log2(1-p)

def get_Hs(outputs, target):
    '''
    Calcula la entropia de las salidas y los targets
    --------------------------------
    Args:
        outputs (np.array): Array de salidas
        targets (np.array): Array de targets
    
    Returns:
        np.array: Array de entropias'''
    Hs = []
    for epoch in range(len(target)):
        Hs.append(H(outputs[epoch]))
    return np.array(Hs)
def plot_Hs(Hs,title,epochs,overfit,path):
    '''
  Grafica la entropia y la guarda en el directorio especificado
  --------------------------------
  Args:
  
      Hs (list): Lista de entropias
      title (str): Titulo del grafico
      epochs (int): Numero de epocas
      path (str): Directorio donde se guardara el grafico'''
    ep = np.arange(1, epochs+1)
    plt.scatter(ep,Hs)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("H")
    plt.axvline(x = overfit, color = 'b', label = 'overfitting')
    fig_path = os.path.join(path, title+".png")
    plt.savefig(fig_path)
    plt.close()
   
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

def get_MI_XR(residuo, data,epochs):
    '''
  Calcula la informacion mutua entre los residuos y los inputs
  --------------------------------
  Args:
      residuo (np.array): Array de residuos
      data (np.array): Array de inputs
  
  Returns:
      np.array: Array de informacion mutua
      '''
    
    mi_r_x = []
    for epoch in tqdm(range(epochs)):
        mi_r_x.append(estimador2(data[epoch],residuo[epoch].ravel()))
    return np.array(mi_r_x)

def plot_MI(MIs,title,epochs,overfit,path):
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
    plt.axvline(x = overfit, color = 'b', label = 'overfitting')
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
      index_xy = find_row_index(xy_comp,xy)
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

def estimador2(x,y):
    #return mutual_info_classif(x,y,n_neighbors=3)
    return mutual_info_regression(x,y,n_neighbors=5)


def estimador(X,Y, n= np.arange(0,13)):
  """
  X corresponde al primer vector, y Y al segundo. [0,2,7,8,9,10,11,12]
  """
  X=X[:,n]
  x_comp, x_num =  npi.count(X)
  y_comp, y_num = npi.count(Y)
  XY = np.concatenate((X, Y), axis=1)
  xy_comp, xy_num = npi.count(XY)
  IM = 0

  for x in x_comp:
    for y in y_comp:

      xy = np.concatenate((x, y))
      index_xy = find_row_index(xy_comp,xy)
      pxy = xy_num[index_xy]/np.sum(xy_num)

      index_x = find_row_index(x_comp,x)
      px =  x_num[index_x]/np.sum(x_num)

      index_y = find_row_index(y_comp,y)
      py =  y_num[index_y]/np.sum(y_num)
      IM += pxy * np.log2(pxy/(px*py))
  return IM

def plot_accuracy(accuracies_train,accuacies_val,epochs,overfit,title,path):
    '''
  Grafica la accuracy y la guarda en el directorio especificado
  --------------------------------
  Args:
      accuracies (list): Lista de accuracies
      title (str): Titulo del grafico
      epochs (int): Numero de epocas
      path (str): Directorio donde se guardara el grafico
      
  Returns:
      None
  '''
    ep = np.arange(1, epochs+1)
    plt.scatter(ep,accuracies_train)
    plt.scatter(ep,accuacies_val)
    plt.legend(['Train','Validation'])
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.axvline(x = overfit, color = 'b', label = 'overfitting')
    fig_path = os.path.join(path, title+".png")
    plt.savefig(fig_path)
    plt.close()

def get_MI_XRE(data,residuo,indices,epochs):
    '''
  Calcula la informacion mutua entre los residuos y los inputs
  '''

    mi_r_x_e = []
    for epoch in tqdm(range(epochs)):
        mi_r_x_e.append(estimador(data[epoch],residuo[epoch], indices))
    return np.array(mi_r_x_e)

def get_MI_mean(residuo):
    n = 5
    MI_mean = []
    for i in range(np.array(residuo).shape[0]):
            MI_mean.append(np.mean(np.array(residuo[i])))
    MI_mean = list(itertools.chain.from_iterable([i]*n for i in [np.mean(MI_mean[i:i+n]) for i in range(0,len(MI_mean),n)])) 

    return np.array(MI_mean)

def plot_all_MIS(residuos,overfit,title,path):
    '''
  Grafica la informacion mutua y la guarda en el directorio especificado
    --------------------------------

    Args:
        MIs (list): Lista de informacion mutua
        title (str): Titulo del grafico
        epochs (int): Numero de epocas
        path (str): Directorio donde se guardara el grafico
    '''
    n = 10
    MI_mean = get_MI_mean(residuos)
    mi_r_x = residuos
    plt.figure(figsize=(10,6))
    for i in range(np.array(mi_r_x).shape[1]):
        list1 = mi_r_x[:,i]
        list1 = list(itertools.chain.from_iterable([i]*n for i in [np.mean(list1[i:i+n]) for i in range(0,len(list1),n)]))
        plt.scatter(range(len(mi_r_x)),list1, label = 'x'+str(i))
    plt.axvline(x = overfit, color = 'b', label = 'overfitting')
    plt.plot(range(len(mi_r_x)),MI_mean, label = "Promedio de la MI por componente", c= 'yellow')
    plt.ylabel('MI[bits]')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.legend()
    fig_path = os.path.join(path, "all.png")
    plt.savefig(fig_path)
    plt.title(f'Promedio de MI(R,x) cada {n} Epochs')