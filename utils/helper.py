# Helper functions file

# PyTorch geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, MixHopSyntheticDataset
from ogb.nodeproppred import PygNodePropPredDataset
from src.models import GCN, GAT, SAGE, APPNPNet
import torch.optim as optim
from torch_geometric.utils import homophily
from utils.helper_path import DATA_PATH, MODELS_PATH
import os
import random
import numpy as np
import torch
import torch.nn as nn
import yaml

def seed_everything(seed:int):
    """
    Sets a seed for reproducibility
    :param seed: seed
    :return: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def create_directory(path:str):
    """
    Checks if a dictionary exists and creates it if necessary
    :param path: path to check
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

def get_data(dataset_name:str, root:str=DATA_PATH, homophily:float=0.3):

    """
    Load and return a dataset based on the specified dataset_name.

    Args:
        dataset_name (str): The name of the dataset to be loaded.
        root (str): The root directory where the dataset is located. Default is DATA_PATH.
        homophily (float): A parameter used in the case of the "Mixhop" dataset. Default is 0.1.

    Returns:
        torch_geometric.data.Data: The loaded dataset.
    """
    # Validate dataset_name
    supported_datasets = ["Cora", "Citeseer", "Pubmed", "Amazon-Computers", "Amazon-Photos",
                          "Coauthor-Physics", "Coauthor-CS", "Mixhop", "OGBN-Arxiv", "OGBN-Products"]

    #check if the datasets folder exists otherwise creates the folder
    create_directory(root)

    if dataset_name not in supported_datasets:
        raise ValueError(f"The dataset '{dataset_name}' is not supported.")

    if dataset_name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(root=root, name=dataset_name, transform=T.NormalizeFeatures())

    elif dataset_name in ["Amazon-Computers", "Amazon-Photos"]:
        dataset = Amazon(root=os.path.join(root, dataset_name), name=dataset_name.split("-")[-1],
                                          transform=T.NormalizeFeatures())

    elif dataset_name in ["Coauthor-Physics", "Coauthor-CS"]:
        dataset = Coauthor(root=os.path.join(root, dataset_name), name=dataset_name.split("-")[-1],
                                        transform=T.NormalizeFeatures())

    elif dataset_name in ["Mixhop"]:
        dataset = MixHopSyntheticDataset(root=os.path.join(root, dataset_name), homophily=homophily, transform=T.NormalizeFeatures())

    elif dataset_name in ["OGBN-Arxiv", "OGBN-Products"]:
        dataset = PygNodePropPredDataset(name=dataset_name.lower(), root=root)

    #Retrieve the data
    data = dataset[0]

    return data

def load_yaml_config_file(path_to_yaml_file:str):
    """
    Loads a yaml file
    :param path_to_yaml_file:
    :return: the resulting dictionary
    """
    try:
        with open(path_to_yaml_file) as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print('We could not find the yaml file that you specified')

def get_homophily(edge_idx, y):
    """
    Returns the homophily for a given graph dataset
    :param edge_idx: edge index
    :param y: true labels for the classes
    :return:
    """
    return np.round(homophily(edge_idx, y), 4)

def get_model(model_name, num_features, num_classes, **kwargs):

    """
    Loads the model
    :param model_name: model_type
    :param num_features: number of features
    :param num_classes: number of classes
    :param kwargs: any additional parameters needed for loading the model
    :return: PyTorch object
    """

    if model_name == "GCN":
        model = GCN(num_features, num_classes, **kwargs)
    elif model_name == "GAT":
        model = GAT(num_features, num_classes, **kwargs)
    elif model_name == "SAGE":
        model = SAGE(num_features, num_classes, **kwargs)
    elif model_name == "APPNPNet":
        model = APPNPNet(num_features, num_classes, **kwargs)
    else:
        raise ValueError("Please indicate one of the available models. 'GCN', 'GAT', 'SAGE' or 'APPNPNET'")
    return model

class DataManager:

    def __init__(self, data, random_seed=None):

        self.x = data.x
        self.y = data.y
        self.edge_index = data.edge_index
        self.random_seed = random_seed if random_seed is not None else 7

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def get_train_test_split(self, percentage_train_data=0.75):

        node_idx = list(np.arange(len(self.x)))
        n_train = int(len(self.x) * percentage_train_data)
        n_test = len(self.x) - n_train

        #We get the test and train idx
        test_idx = np.random.choice(node_idx, replace=False, size=n_test)
        train_idx = np.setdiff1d(node_idx, test_idx)

        return train_idx, test_idx

    def get_calibration_split(self, train_idx, equal_samples_per_cls=True,
                              number_nodes_per_cls=80, percentage_train_data=1/3, tuning_set=False, percentage_tuning_data=None):

        #ToDo: Clean this a bit up and write it more concisely

        node_classes = self.y.unique()
        n_train = int(len(train_idx) * percentage_train_data)

        if tuning_set:
            n_calibration = len(train_idx) - n_train * (1-percentage_tuning_data)
        else:
            n_calibration = int(len(train_idx) - n_train)

        if not equal_samples_per_cls:
            true_train_idx = np.random.choice(train_idx, replace=False, size=n_train)
            total_calibration_idx = np.setdiff1d(train_idx, true_train_idx)

            if not tuning_set:
                true_calibration_idx = total_calibration_idx
            else:
                true_calibration_idx = np.random.choice(total_calibration_idx, replace=False, size=n_calibration)
                true_tuning_idx = np.setdiff1d(true_calibration_idx, total_calibration_idx)

            # Quick check if things worked out as intended
            assert len(true_train_idx) == n_train and len(true_calibration_idx) == n_calibration

        else:
            true_train_idx = []
            true_calibration_idx = []
            if tuning_set:
                true_tuning_idx = []

            for cls in node_classes:
                #Retrieve idx for which the node has class label
                class_idx = torch.nonzero(self.y==cls, as_tuple=True)[0]
                # Example with default setting: 20 * 3 = 60 nodes in total for calibration and train
                total_number_samples = int(number_nodes_per_cls * (1/percentage_train_data))
                number_true_train_idx_samples = int(percentage_train_data * total_number_samples)

                if tuning_set:
                    number_true_calibration_samples = int((1 - percentage_train_data) * total_number_samples * (1-percentage_tuning_data))
                else:
                    number_true_calibration_samples = int((1 - percentage_train_data) * total_number_samples)

                total_number_calibration_samples = total_number_samples - number_true_train_idx_samples
                rdm_permutation_idx = torch.randperm(len(class_idx))

                shuffled_class_idx = class_idx[rdm_permutation_idx]
                true_train_idx.extend(shuffled_class_idx[:number_true_train_idx_samples].tolist())

                total_calibration_idx = shuffled_class_idx[number_true_train_idx_samples:number_true_train_idx_samples + total_number_calibration_samples].tolist()

                if tuning_set:
                    true_calibration_idx.extend(total_calibration_idx[:number_true_calibration_samples])
                    true_tuning_idx.extend(total_calibration_idx[number_true_calibration_samples:])

                else:
                    true_calibration_idx.extend(total_calibration_idx)

            # Quick check if things worked out as intended
            assert len(true_train_idx) == number_true_train_idx_samples * len(node_classes)
            assert len(true_calibration_idx) == number_true_calibration_samples * len(node_classes)

        if tuning_set:
            return true_train_idx, true_calibration_idx, true_tuning_idx

        else:
            return true_train_idx, true_calibration_idx

#Code adapted from:
# https://github.com/soroushzargar/DAPS/blob/main/torch-conformal/gnn_cp/models/model_manager.py

class Graph_Trainer:

    def __init__(self, model, **kwargs):
        self.model = model
        self.loss_module = nn.CrossEntropyLoss()

        optimizer_name = kwargs.get("optimizer_name", "adam")
        learning_rate = kwargs.get("learning_rate", 1e-3)
        weight_decay = kwargs.get("weight_decay", 1e-4)

        self.optimizer = self._get_optimizer(optimizer_name, self.model.parameters(), learning_rate, weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_optimizer(self, optimizer_name, parameters, learning_rate, weight_decay):
        if optimizer_name == "adam":
            optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

        elif optimizer_name == "sgd":
            optimizer = optim.SGD(parameters, lr=learning_rate, momentum=0.90, weight_decay=weight_decay)

        elif optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)

        return optimizer

    def fit(self, data, train_mask, valid_mask, **kwargs):
        x, edge_index, y = data.x, data.edge_index, data.y
        #Set the model to train mode
        self.model.train()

        warm_up = kwargs.get("warm_up", True)
        warm_up_epochs = kwargs.get("warm_up_epochs", 50)
        n_epochs = kwargs.get("n_epochs", 200)
        self.early_stopping = kwargs.get("early_stopping", False)
        self.patience = kwargs.get("patience", 10)
        verbose = kwargs.get("verbose", True)

        self.warm_up = 50

        #Set the model to training
        self.model.train().to(self.device)

        #Keep track of the best validation loss so far -> we minimize!
        best_validation_loss = 1000

        #Do a warmup in case the boolean flag was set
        if warm_up:
            for warm_up_epoch in range(warm_up_epochs):
                _, loss_valid, _ = self._train(x, edge_index, y, train_mask, valid_mask)

                #Keep track of the best validation loss so far
                if loss_valid < best_validation_loss:
                    best_validation_loss = loss_valid

        for epoch in range(1, n_epochs+1):
            #We train full batch
            #Log every 10 steps
            if verbose and (epoch % 10 == 0):
                print(f'Epoch {epoch:3d} - ', end='')

            loss_train, loss_valid, accuracy_train = self._train(x, edge_index, y, train_mask, valid_mask)

            # Early stopping based on validation loss, if the validation loss does not decrease in 10 consecutive epochs we break
            if self.early_stopping:
                if loss_valid < best_validation_loss:
                    best_validation_loss = loss_valid
                    #Save the best checkpoint so far
                    self.best_checkpoint = self.model.state_dict()
                    #Reset the patience score
                    self.patience = kwargs.get("patience", 10)
                else:
                    self.patience -= 1

            if verbose and (epoch % 10 == 0):
                print(f'train_loss: {float(loss_train):.4f}\ttrain_acc: {accuracy_train:.4f}\tvalid_loss: {loss_valid:.4f}')

            if self.patience == 0:
                print(f'Training is stopped as the model has not improved the validation loss in the past {kwargs.get("patience")} epochs!')
                # If the model has not improved the past patience steps, then we stop training
                break

    def save_model(self, directory, save_name):
        if self.early_stopping:
            self.model.load_state_dict(self.best_checkpoint)

        torch.save(self.model.state_dict(), os.path.join(directory, save_name))

    def _train(self, x, edge_index, y, train_mask, valid_mask):
        """
        Performs an iteration of the training
        """
        y_pred = self.model(x.to(self.device), edge_index.to(self.device)).cpu()
        y_pred_train, y_pred_valid = y_pred[train_mask], y_pred[valid_mask]
        y_true_train, y_true_valid = y[train_mask], y[valid_mask]

        # Calculate the loss
        loss_train = self.loss_module(y_pred_train, y_true_train)
        loss_valid = self.loss_module(y_pred_valid, y_true_valid)

        #Calculate the accuracies
        accuracy_train = self._get_accuracy(y_pred_train, y_true_train)

        # Zero gradients, perform backward pass and update the weights
        self.optimizer.zero_grad()
        loss_train.backward()
        self.optimizer.step()

        return loss_train, loss_valid, accuracy_train

    def _get_accuracy(self, y_pred, y_true):
        return torch.mean((torch.argmax(y_pred, dim=-1) == y_true).float(), dim=0)

    def test(self, data, test_mask):

        #Set the model to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        y_true = data.y[test_mask]
        y_pred = self.model(data.x.to(self.device), data.edge_index.to(self.device))[test_mask].cpu()
        loss = float(self.loss_module(y_pred, y_true)) #Release memory
        accuracy = self._get_accuracy(y_pred, y_true)

        print(f'test loss {loss:.4f}\ttest_acc {accuracy:.4f}')




#Code adapted from: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html
# class Graph_Trainer(L.LightningModule):
#
#     def __init__(self, model, train_mask, valid_mask, test_mask, optimizer_name, learning_rate, weight_decay):
#         super().__init__()
#
#         self.save_hyperparameters()
#         self.model = model
#         self.loss_module = nn.CrossEntropyLoss()
#         self.train_mask = train_mask
#         self.valid_mask = valid_mask
#         self.test_mask = test_mask
#         self.optimizer_name = optimizer_name
#         self.lr = learning_rate
#         self.weight_decay = weight_decay
#
#     def forward(self, data, mode="train"):
#         x, edge_index = data.x, data.edge_index
#         #Get the forward pass
#         x = self.model(x, edge_index)
#
#         #Now we will mask out the loss to get to only the node of the
#         if mode == "train":
#             mask = self.train_mask
#         elif mode == "validation":
#             mask = self.valid_mask
#         elif mode == "test":
#             mask = self.test_mask
#
#         loss = self.loss_module(x[mask], data.y[mask])
#         accuracy = (x[mask].argmax(dim=-1) == data.y[mask]).sum().float() / mask.sum()
#
#         return loss, accuracy
#
#     def configure_optimizer(self):
#
#         if self.optimizer_name == "adam":
#             optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#         elif self.optimizer_name == "sgd":
#             optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.90, weight_decay=self.weight_decay)
#
#         elif self.optimizer_name == 'rmsprop':
#             optimizer = optim.RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#
#         return optimizer
#
#     def training_step(self, batch, batch_idx):
#         loss, acc = self.forward(batch, mode="train")
#         self.log("train_loss", loss)
#         self.log("train_accuracy", acc)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         loss, acc = self.forward(batch, mode="validation")
#         self.log("validation_loss", loss)
#         self.log("validation_accuracy", acc)
#         return loss
#
#     def test_step(self, batch, batch_idx):
#         loss, acc = self.forward(batch, mode="test")
#         self.log("test_loss", loss)
#         self.log("test_accuracy", acc)
#         return loss

#Code adapted from: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/06-graph-neural-networks.html



















