import os
import time
import torch
import numpy as np
from data.data_provider import DataProviderBase, DataProviderHD
from models.mlp_model import MLP
import torch.nn as nn
from hd_experiment.utils import *
from sklearn.metrics import accuracy_score

class BaseExperimentHD():
    
    def __init__(self,args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self._get_data()
        self.model = self._get_model()
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr)

    def _get_data(self):
        '''
        Crea el dataprovider
        --------------------------------
        Args:
            None
        Returns:
            DataProvider: DataProvider'''
        return DataProviderHD(self.args)

    def _get_model(self):
        '''
        Crea el modelo
        --------------------------------
        Args:
            None
        Returns:
            MLP: Modelo'''
        return MLP(input_dim = 13, hidden_dim = self.args.hidden_dim, output_dim =2 ).to(self.device)

    def run(self):
        '''
        Realizacion del experimento
        --------------------------------
        Args:
            None
        Returns:
            None'''
        experiment_path = os.path.join(str(self.args.exp_path), self.args.exp_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        loss_train = []
        loss_val = []
        train_accuracies = []
        val_accuracies = []
        all_train_outputs = []
        all_train_targets = []
        all_val_outputs = []
        all_val_targets = []
        all_train_data = []

        for _ in range(self.args.epochs):
            start_time = time.time()
            train_losses, train_outputs, train_targets, train_data = self.train_epoch(self.model, 
                                                                          self.data.train_dataloader, 
                                                                          self.loss_fn,
                                                                          self.optimizer, 
                                                                          self.device)
            loss, acc = get_scores(train_losses, train_outputs, train_targets)
            train_accuracies.append(acc)
            all_train_data.append(train_data)
            all_train_outputs.append(train_outputs)
            all_train_targets.append(train_targets)
            loss_train.append(loss)
            print("Train Loss: {:.3f}\tTrain Acc: {:.3f}".format(loss, acc))

            val_losses, val_outputs, val_targets= self.eval_epoch(self.model,
                                                                    self.data.val_dataloader,
                                                                    self.loss_fn,
                                                                    self.device)
            loss, acc = get_scores(val_losses, val_outputs, val_targets)
            val_accuracies.append(acc)
            all_val_outputs.append(val_outputs)
            all_val_targets.append(val_targets)
            loss_val.append(loss)
            print("Val Loss: {:.3f}\tVal Acc: {:.3f}".format(loss, acc))
            
            end_time = time.time()
            print("Tiempo: {:.2f}\n".format(end_time-start_time))

        all_train_outputs = torch.stack(all_train_outputs).cpu().numpy()
        all_train_targets = torch.vstack(all_train_targets).cpu().numpy()
        
        n = 10
        residuos = get_R(all_train_outputs, all_train_targets)
        entropy = get_Hs(residuos, all_train_outputs)
        overfit = np.argmin(loss_val)
        residuos_new = np.array(residuos).reshape(self.args.epochs,181,1)
        mutual_info_XR = get_MI_XR(residuos, all_train_data,self.args.epochs)
    
        indices_mas_altos = np.argpartition(np.mean(mutual_info_XR[:n],axis=0), -3)[-3:]
        mutual_info_XRE = get_MI_XRE(all_train_data,residuos_new , indices_mas_altos,self.args.epochs)

  

        plot_accuracy(train_accuracies, val_accuracies, self.args.epochs,overfit, "Accuracy por Epoca", experiment_path)
        training_curve(loss_train, loss_val, self.args.epochs,overfit, "Curva de Entrenamiento", experiment_path)
        plot_Hs(entropy, "Entropia del Residuo", self.args.epochs,overfit, experiment_path)
        plot_MI(mutual_info_XRE, "Informacion Mutua del Residuo-Input", self.args.epochs,overfit, experiment_path)
        plot_all_MIS(mutual_info_XR,overfit,'MI de la Entrada con Respecto al Residuo', experiment_path)


    def train_epoch(self,model, dataloader, loss_fn, optimizer, device):
        '''
        Realiza un epoch de entrenamiento
        --------------------------------
        Args:
            model (MLP): Modelo
            dataloader (DataLoader): DataLoader de entrenamiento
            loss_fn (nn.Module): Funcion de perdida
            optimizer (torch.optim): Optimizador
            device (torch.device): Dispositivo
        Returns:
            torch.Tensor: Lista de perdidas
            torch.Tensor: Salidas
            torch.Tensor: Targets
            torch.Tensor: Datos
        '''
        model.train()
        all_data = []
        all_outputs = []
        all_targets = []
    
        losses = []

        for x,y in dataloader:
            x = x.to(device)
            targets = y.to(device).view(-1)
            batch_size = x.shape[0]
            outputs = model(x.float())
            loss = loss_fn(outputs, targets.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            outputs = torch.argmax(outputs, dim=1)
            losses.append(loss.item()*batch_size)
            all_data.append(x.detach().cpu().numpy())
            all_outputs.append(outputs.detach())
            all_targets.append(targets.detach())
        all_data = np.concatenate(all_data)
        return torch.tensor(losses), torch.cat(all_outputs), torch.cat(all_targets),all_data
    
    def eval_epoch(self,model, dataloader, loss_fn, device):
        '''
        Realiza un epoch de validacion
        --------------------------------
        Args:
            model (MLP): Modelo
            dataloader (DataLoader): DataLoader de validacion
            loss_fn (nn.Module): Funcion de perdida
            optimizer (torch.optim): Optimizador
            device (torch.device): Dispositivo
        Returns:
            torch.Tensor: Lista de perdidas
            torch.Tensor: Salidas
            torch.Tensor: Targets
        '''

        model.eval()
        all_outputs = []
        all_targets = []
        losses = []
        with torch.no_grad():
            for x,y in dataloader:
                x = x.to(device)
                targets = y.to(device).view(-1)
                batch_size = x.shape[0]
                outputs = model(x.float())
                loss = loss_fn(outputs, targets.long())
                losses.append(loss.item()*batch_size)
                outputs = torch.argmax(outputs, dim=1)
                all_outputs.append(outputs.detach())
                all_targets.append(targets.detach())

        return torch.tensor(losses), torch.cat(all_outputs), torch.cat(all_targets)