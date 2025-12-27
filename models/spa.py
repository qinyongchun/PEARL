import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet,SimpleCosineIncrementalNet,MultiBranchCosineIncrementalNet,SimpleVitNet,SimpleVitNet_SPA
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy

# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8
class AdjustedSigmoid(nn.Module):
    def __init__(self, k=1, b=0):
        super(AdjustedSigmoid, self).__init__()
        self.k = k
        self.b = b

    def forward(self, x):
        # x = 1 - x
        sigmoid = 1 / (1 + torch.exp(-10 * (x - 0.5)))
        return self.k*sigmoid + self.b

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleVitNet_SPA(args, True)
        self. batch_size = args["batch_size"]
        self. init_lr = args["init_lr"]
        
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args['min_lr'] if args['min_lr'] is not None else 1e-8
        self.args = args
        self.activation = AdjustedSigmoid(k=0.199,b=0.8)
        # self.sigma = 0.1

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network

    def replace_fc(self, trainloader, model, args):       
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        
        Y = target2onehot(label_list, self.args["nb_classes"])
        Features_h = F.relu(embedding_list @ self.W_rand.cpu())
        self.Q = self.Q + Features_h.T @ Y
        self.G = self.G + Features_h.T @ Features_h
        ridge = self.optimise_ridge_parameter(Features_h, Y)
        Wo = torch.linalg.solve(self.G + ridge*torch.eye(self.G.size(dim=0)), self.Q).T # better nmerical stability than .invv
        self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0],:].to(self._device)
        
        return model

    def setup_RP(self):
        M = self.args['M']
        self._network.fc.weight = nn.Parameter(torch.Tensor(self._network.fc.out_features, M).to(self._device)).requires_grad_(False) # num classes in task x M
        self._network.RP_dim = M
        self.W_rand = torch.randn(self._network.fc.in_features, M).to(self._device)
        self._network.W_rand = self.W_rand

        self.Q = torch.zeros(M, self.args["nb_classes"])
        self.G = torch.zeros(M, M)

    def optimise_ridge_parameter(self, Features, Y):
        ridges = 10.0 ** np.arange(-8, 9)
        num_val_samples = int(Features.shape[0] * 0.8)
        losses = []
        Q_val = Features[0:num_val_samples, :].T @ Y[0:num_val_samples, :]
        G_val = Features[0:num_val_samples, :].T @ Features[0:num_val_samples, :]
        for ridge in ridges:
            Wo = torch.linalg.solve(G_val + ridge*torch.eye(G_val.size(dim=0)), Q_val).T #better nmerical stability than .inv
            Y_train_pred = Features[num_val_samples::,:] @ Wo.T
            losses.append(F.mse_loss(Y_train_pred, Y[num_val_samples::, :]))
        ridge = ridges[np.argmin(np.array(losses))]
        print('selected lambda =',ridge)
        return ridge
    
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train", )
        self.train_dataset=train_dataset
        self.data_manager=data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        self._network.to(self._device)
        
        # if self._cur_task == 0:
        #     # show total parameters and trainable parameters
        #     total_params = sum(p.numel() for p in self._network.parameters())
        #     print(f'{total_params:,} total parameters.')
        #     total_trainable_params = sum(
        #         p.numel() for p in self._network.parameters() if p.requires_grad)
        #     print(f'{total_trainable_params:,} training parameters.')
        #     if total_params != total_trainable_params:
        #         for name, param in self._network.named_parameters():
        #             if param.requires_grad:
        #                 print(name, param.numel())
        #     if self.args['optimizer'] == 'sgd':
        #         optimizer = optim.SGD(self._network.parameters(), momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        #     elif self.args['optimizer'] == 'adam':
        #         optimizer = optim.AdamW(self._network.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        #     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        #     self._init_train(train_loader, test_loader, optimizer, scheduler)
        # else:
        #     pass
        
        total_params = sum(p.numel() for p in self._network.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self._network.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        
        if self._cur_task == 0:
            para = self._network.parameters()
        else:
            for name, p in self._network.backbone.blocks.named_parameters():
                p.requires_grad = False 
            para = self._network.parameters()

            
        # if total_params != total_trainable_params:
        #     for name, param in self._network.named_parameters():
        #         if param.requires_grad:
        #             print(name, param.numel())
                    
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(para, momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.AdamW(para, lr=self.init_lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        
        
        if self._cur_task == 0 and self.args["use_RP"]:
            self.setup_RP()
        self.replace_fc(train_loader_for_protonet, self._network, None)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        if self._cur_task>0:
            # 初始化sigma
            self._network.sigma = self.args['sigma']
            if self.args['learnable_sigma']:
                print("Learnable sigma")
                self._network.sigma = nn.Parameter(torch.FloatTensor(1).fill_(self._network.sigma), requires_grad=True).cuda()

        if self._cur_task>=0:
            for _, epoch in enumerate(prog_bar):
                self._network.train()
                losses = 0.0
                correct, total = 0, 0
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)

                    if self._old_network!=None:
                        with torch.no_grad():
                            old_logits = self._old_network(inputs, train=False, old=True)["logits"]
                            # print('old sigma:',self._old_network.sigma)
                            now_logits = self._network(inputs, train=False)["logits"]
                            s = int(self.args['increment']*self._cur_task)
                            # T = 0.00002
                            T = self.args['temperature']
                            sim,___ = average_mae(old_logits[:,0:s]/T,now_logits[:,0:s]/T)
                            sim = torch.tensor(sim)
                            self._network.sigma = self._network.sigma*0.97 + self.activation(sim).detach()*0.03
                            # print('sigma:',self._network.sigma,'sim:',sim)
                            logging.info('sigma:{}-sim:{}'.format(self._network.sigma,sim))
                        logits = self._network(inputs,train=True)["logits"]
                        # print(logits)
                    else:
                        logits = self._network(inputs,train=True)["logits"]
                    logits[:, :self._known_classes] = float('-inf')
                    loss = F.cross_entropy(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    losses += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

                scheduler.step()
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, Sigma {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                    self._network.sigma.cpu().item() if isinstance(self._network.sigma, torch.Tensor) else self._network.sigma
                )
                prog_bar.set_description(info)
            with torch.no_grad():
                if self._old_network != None:
                    outputs = self._network(inputs,reset_fix=True)["logits"][:, :self._total_classes]
                else:
                    outputs = self._network(inputs,reset_fix=True)["logits"][:, :self._total_classes]

            logging.info(info)
        
def average_mae(A, B):
    # 计算每对对应向量的MAE
    mae = F.l1_loss(A, B, reduction='mean')
    
    # MAE本身在[0, ∞)区间，但为了和其它度量统一，可以考虑根据应用需要进行归一化
    return mae.item(),mae
