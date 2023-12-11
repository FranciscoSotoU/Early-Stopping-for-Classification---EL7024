import torch
import numpy as np
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from torch.utils.data import DataLoader
class DataProviderBase():
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
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_data, batch_size=self.args.batch_size, shuffle=False)

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


class DataProviderHD():
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
        df=pd.read_csv(self.args.data_path)
        df = df.drop(df.columns[0], axis=1)
        encoder = LabelEncoder()
        for c in df.columns[0:]:
            if(df[c].dtype == 'object'):
                df[c] = encoder.fit_transform(df[c])
            else:
                df[c] = df[c]
        df = df.fillna(0)
        y = df["target"]
        df_scaled = df.copy()
        df_scaled.pop("target")
        scaler = StandardScaler()
        df_scaled[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']] = pd.DataFrame(scaler.fit_transform(df[['age', 'trestbps', 'chol', 'thalach', 'oldpeak']]))
        columnas_a_binarizar = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        df_scaled[columnas_a_binarizar] = df_scaled[columnas_a_binarizar].apply(lambda x: (x > x.mean()).astype(int))
        x = df_scaled
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.40)

        x_train_tensor = torch.tensor(x_train.values)
        y_train_tensor = torch.tensor(y_train.values).view(-1,1)
        x_val_tensor = torch.tensor(x_test.values)
        y_val_tensor = torch.tensor(y_test.values).view(-1,1)

        self.train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        self.val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    def _create_dataloader(self):
        '''
        Crea los dataloaders de entrenamiento y validacion
        --------------------------------
        Args:
            None
        Returns:
            None'''
        self.train_dataloader = DataLoader(self.train_dataset, batch_size = self.args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size = self.args.batch_size)

