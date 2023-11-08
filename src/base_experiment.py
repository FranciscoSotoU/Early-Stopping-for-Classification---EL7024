import os
import time
import torch
import numpy as np
from data.data_provider import DataProvider
from models.mlp_model import MLP
import torch.nn as nn
from utils import get_scores, training_curve, mutual_info, binarizar, get_R, get_MI, plot_MI, get_MI_XR

class BaseExperiment():
    
    def __init__(self,args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = self._get_data()
        self.model = self._get_model()
        self.loss_fn = nn.BCELoss().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.lr)

    def _get_data(self):
        '''
        Crea el dataprovider
        --------------------------------
        Args:
            None
        Returns:
            DataProvider: DataProvider'''
        return DataProvider(self.args)

    def _get_model(self):
        '''
        Crea el modelo
        --------------------------------
        Args:
            None
        Returns:
            MLP: Modelo'''
        return MLP(input_dim = 2, hidden_dim = self.args.hidden_dim, output_dim =1 ).to(self.device)

    def run(self):
        '''
        Realizacion del experimento
        --------------------------------
        Args:
            None
        Returns:
            None'''
        experiment_path = os.path.join(self.args.exp_path, self.args.exp_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        loss_train = []
        loss_val = []
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
            all_train_data.append(train_data)
            all_train_outputs.append(train_outputs)
            all_train_targets.append(train_targets)
            loss_train.append(loss)
            print("Train Loss: {:.3f}\tTrain Acc: {:.3f}".format(loss, acc))

            val_losses, val_outputs, val_targets= self.eval_epoch(self.model,
                                                                    self.data.val_dataloader,
                                                                    self.loss_fn,
                                                                    self.optimizer,
                                                                    self.device)
            loss, acc = get_scores(val_losses, val_outputs, val_targets)
            all_val_outputs.append(val_outputs)
            all_val_targets.append(val_targets)
            loss_val.append(loss)
            print("Val Loss: {:.3f}\tVal Acc: {:.3f}".format(loss, acc))
            
            end_time = time.time()
            print("Tiempo: {:.2f}\n".format(end_time-start_time))

        all_train_outputs = torch.stack(all_train_outputs).cpu().numpy()
        all_train_targets = torch.stack(all_train_targets).cpu().numpy()
        all_train_data = torch.stack(all_train_data).cpu().numpy()

        residuos = get_R(all_train_outputs, all_train_targets)

        mutual_infos = get_MI(all_train_outputs, all_train_targets)
        mutual_info_R = get_MI(residuos, all_train_outputs)
        mutual_info_XR = get_MI_XR(residuos, all_train_data)

        plot_MI(mutual_infos, "Informacion Mutua del True-Predicted", self.args.epochs, experiment_path)
        plot_MI(mutual_info_R, "Informacion Mutua del Residuo-Predicted", self.args.epochs, experiment_path)
        plot_MI(mutual_info_XR, "Informacion Mutua del Residuo-Input", self.args.epochs, experiment_path)
        training_curve(loss_train, loss_val, self.args.epochs, "Curva de entrenamiento", experiment_path)


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
            targets = y.to(device)
            batch_size = x.shape[0]
            outputs = model(x.float())
            loss = loss_fn(outputs, targets.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item()*batch_size)
            all_data.append(x.detach())
            all_outputs.append(outputs.detach())
            all_targets.append(targets.detach())

        return torch.tensor(losses), torch.cat(all_outputs), torch.cat(all_targets),torch.cat(all_data)
    
    def eval_epoch(self,model, dataloader, loss_fn, optimizer, device):
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
                targets = y.to(device)
                batch_size = x.shape[0]
                outputs = model(x.float())
                loss = loss_fn(outputs, targets.float())
                losses.append(loss.item()*batch_size)
                all_outputs.append(outputs.detach())
                all_targets.append(targets.detach())

        return torch.tensor(losses), torch.cat(all_outputs), torch.cat(all_targets)



