import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split


class DataProvider():
    def __init__(self,args):
        self.args = args
        self._create_dataset()
        self._create_dataloader()
    def _create_dataset(self):
        '''
        Crea el dataset de entrenamiento y validacion
        --------------------------------
        Args:
            None
        Returns:
            None'''
        data = self.CreateTruthTable(p = self.args.p, n = self.args.n_samples, seed = self.args.seed)

        train, val = train_test_split(data, test_size=0.2, random_state=42,shuffle=True)

        x_train = torch.tensor(train[:,:2])
        y_train = torch.tensor(train[:,2]).unsqueeze(1)

        x_val = torch.tensor(val[:,:2])
        y_val = torch.tensor(val[:,2]).unsqueeze(1)

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        self.train_data = train_dataset
        self.val_data = val_dataset
    
    def _create_dataloader(self):
        '''
        Crea los dataloaders de entrenamiento y validacion
        --------------------------------
        Args:
            None
        Returns:
            None'''
        self.train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_data, batch_size=self.args.batch_size, shuffle=False)

    def CreateTruthTable(self,p = 0.5, n = 1000, seed = 42):
        '''
        Crea una tabla de verdad de la compuerta logica AND con ruido bernoulli
        --------------------------------
        Args:
            p (float): Probabilidad de ruido
            n (int): Numero de muestras
            seed (int): Semilla para la generacion de numeros aleatorios
        Returns:
            np.array: Tabla de verdad con ruido
        '''
        np.random.seed(seed)
        a = np.random.randint(2, size=n)
        b = np.random.randint(2, size=n)
        ruido = np.random.binomial(1,p,size=n)
        c = np.logical_and(a, b)
        c = np.logical_xor(c, ruido)
        truth_table = np.array([a, b, c]).T
        return np.array(truth_table)

