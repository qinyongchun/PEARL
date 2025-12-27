import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.optim import Optimizer
import math
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import TSPVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy,target2onehot

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8
class AdjustedSigmoid(nn.Module):
    def __init__(self, k=1, b=0):
        super(AdjustedSigmoid, self).__init__()
        self.k = k
        self.b = b

    def forward(self, x):
        x = 1 - x
        sigmoid = 1 / (1 + torch.exp(-10 * (x - 0.5)))
        return self.k*sigmoid + self.b

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
    
        self._network = TSPVitNet(args, True)

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.activation = AdjustedSigmoid(k=0.99,b=0.01)
        self.sigma = 0.1
        
        # total_params = sum(p.numel() for p in self._network.parameters())
        # logging.info(f'{total_params:,} total parameters.')
        # total_trainable_params = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad) + sum(p.numel() for p in self._network.prompt.parameters() if p.requires_grad)
        # logging.info(f'{total_trainable_params:,} fc and prompt training parameters.')


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
                embedding_list.append(embedding)
                label_list.append(label)
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        class_list = np.unique(self.train_dataset.labels)
        proto_list = []

        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        
        # Y = target2onehot(label_list, self.args["nb_classes"])
        # Features_h = F.relu(embedding_list @ self.W_rand)
        # self.Q = self.Q + Features_h.T @ Y
        # self.G = self.G + Features_h.T @ Features_h
        # ridge = self.optimise_ridge_parameter(Features_h, Y)
        # Wo = torch.linalg.solve(self.G + ridge*torch.eye(self.G.size(dim=0)), self.Q).T # better nmerical stability than .invv
        # self._network.fc.weight.data = Wo[0:self._network.fc.weight.shape[0],:].to(self._device)
        
        return model
    def get_proto(self, trainloader, model, args):       
        model = model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_,data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.extract_vector(data)
                embedding_list.append(embedding)
                label_list.append(label)
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)
        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            proto_list.append(proto)
        # proto_list = torch.tensor(proto_list)
                
        return torch.stack(proto_list,dim=0)

    def replace_fc_ranpac(self, trainloader, model, args):       
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
        print('selected lambda TSP =',ridge)
        return ridge

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc_ranpac(self._cur_task, self._total_classes)
        if self._cur_task==3:
            xxxx=1
        if self._cur_task > 0:
            try:
                if self._network.module.prompt is not None:
                    self._network.module.prompt.process_task_count()
            except:
                if self._network.prompt is not None:
                    self._network.prompt.process_task_count()

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        

        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.cuda()
        # proto = self.get_proto(self.train_loader_for_protonet, self._network, None)
        # print(proto.shape)
        # self._network.update_fc(self._total_classes,proto)

        self._train(self.train_loader, self.test_loader,self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader,train_loader_for_protonet):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self.data_weighting()
        
        # self.replace_fc(train_loader_for_protonet, self._network, None)
        self._init_train(train_loader, test_loader, optimizer, scheduler,train_loader_for_protonet)
        if self._cur_task == 0 and self.args["use_RP"]:
            self.setup_RP()
        self.replace_fc_ranpac(train_loader_for_protonet, self._network, None)
        # self.replace_fc(train_loader_for_protonet, self._network, None)

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        if len(self._multiple_gpus) > 1:
            params = list(self._network.module.prompt.parameters()) + list(self._network.module.parameters())
            # params = list(self._network.module.prompt.parameters())
        else:
            if self._cur_task > 0:
                params = list(self._network.prompt.parameters())
            else:
                # params = list(self._network.prompt.parameters()) + list(self._network.backbone.blocks.parameters())
                params = list(self._network.backbone.parameters())
                #输出可训练的参数名
                print('trainable parameters:',sum(p.numel() for p in self._network.parameters()))
                print('trainable parameters:',sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad))
                print('trainable parameters:',sum(p.numel() for p in self._network.prompt.parameters() if p.requires_grad))
            # params = list(self._network.parameters())+list(self._network.fc.parameters())
            # params = list(self._network.prompt.parameters())
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr,weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = CosineSchedule(optimizer, K=self.args["tuned_epoch"])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler,train_loader_for_protonet):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        # if self._cur_task==3:
        #     prog_bar = tqdm(range(5))
        self.sigma = 0.01
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            # if self._cur_task > 0:
            #     self._network.fc.requires_grad_(False)
            #     self.sigma = 0.2
            # else:
            #     self._network.fc.requires_grad_(True)
            
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                embedding_list = []
                label_list = []
                label_list.append(targets)
                if self._old_network!=None:
                    with torch.no_grad():
                        old_logits = self._old_network(inputs, train=False, old=True,sigma=self.sigma)["logits"]
                        now_logits = self._network(inputs, train=False,sigma=self.sigma)["logits"]
                        s = int(self.args['increment']*self._cur_task)
                        # old_logits = -F.log_softmax(old_logits[:,0:s], dim=1)
                        # now_logits = -F.log_softmax(now_logits[:,0:s], dim=1)
                        # kl = F.kl_div(now_logits, old_logits, reduction='batchmean')
                        # sigma = self.activation(kl)
                        T = 0.0001
                        T = 1
                        sim,___ = average_mae(old_logits[:,0:s]/T,now_logits[:,0:s]/T)
                        # sim,___ = average_kl_divergence(old_logits[:,0:s]/T,now_logits[:,0:s]/T)
                        # average_kl_divergence
                        sim = torch.tensor(sim)
                        # sim = torch.tensor(average_mse(old_logits[:,0:s],now_logits[:,0:s]))
                        # sim = torch.tensor(average_kl_divergence(old_logits[:,0:s],now_logits[:,0:s]))
                        # sim = torch.tensor(average_cosine_similarity(old_logits[:,0:s],now_logits[:,0:s]))
                        # print(sim)
                        # self.sigma = self.sigma*0.9 + self.activation(sim).detach()*0.1
                        self.sigma = self.activation(sim).detach()
                        # print(self.sigma)
                        # self.sigma = 0.01
                        # print(kl)
                        # print(self.sigma)
                # logits
                    logits, prompt_loss = self._network(inputs, train=True,sigma=self.sigma)
                else:
                    logits, prompt_loss = self._network(inputs, train=True)

                # with torch.no_grad():
                #     embedding = self._network.extract_vector(inputs)
                #     embedding_list.append(embedding)
                #     embedding_list = torch.cat(embedding_list, dim=0)
                #     label_list = torch.cat(label_list, dim=0)
                #     # class_list = np.unique(self.train_dataset.labels)
                #     class_list = np.array(label_list.unique().cpu())
                #     for class_index in class_list:
                #         # print('Replacing...',class_index)
                #         data_index = (label_list == class_index).nonzero().squeeze(-1)
                #         embedding = embedding_list[data_index]
                #         proto = embedding.mean(0)
                #         lamda = 0.01
                #         self._network.fc.weight.data[class_index] = (1-lamda)*self._network.fc.weight.data[class_index] + lamda* proto

                logits = logits["logits"][:, :self._total_classes]
                logits_old = logits[:, :self._known_classes].clone()
                logits[:, :self._known_classes] = float('-inf')
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                loss_supervised = (F.cross_entropy(logits, targets.long()) * dw_cls).mean()

                # ce loss
                if self._old_network!=None:
                    # loss = loss_supervised + sim_loss
                    __,sim_loss = average_mae(old_logits[:,0:s]/T,now_logits[:,0:s]/T)
                    loss = loss_supervised + sim_loss
                else:
                    loss = loss_supervised

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # self.replace_fc(train_loader_for_protonet, self._network, None)

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Sigma {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                    self.sigma
                )
            prog_bar.set_description(info)

        logging.info(info)

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        # if hasattr(self, "_class_means"):
        #     y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
        #     nme_accy = self._evaluate(y_pred, y_true)
        # else:
        #     nme_accy = None

        # if hasattr(self, "prototypes"):
        #     y_pred, y_true = self._eval_proto(self.test_loader,self.prototypes)
        
        #     proto_accy = self._evaluate(y_pred, y_true)
        # else:
        #     proto_accy = None

        return cnn_accy, None

    def _eval_cnn(self, loader):
        self._network.eval()
        
        y_pred, y_true = [], []
        with torch.no_grad():
            inputs = torch.randn(16, 3, 224, 224).to(self._device)
            if self._old_network != None:
                outputs = self._network(inputs,sigma=self.sigma,reset_fix=True)["logits"][:, :self._total_classes]
            else:
                outputs = self._network(inputs,sigma=0,reset_fix=True)["logits"][:, :self._total_classes]

        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                if self._old_network != None:
                    outputs = self._network(inputs, train=False, sigma=self.sigma)["logits"][:, :self._total_classes]
                else:
                    outputs = self._network(inputs, train=False)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            # print(predicts)
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)
    
    def _compute_accuracy2(self, model, loader):
        model.eval()
        correct, total = 0, 0
        y_pred, y_true = [], []
        with torch.no_grad():
            inputs = torch.randn(16, 3, 224, 224).to(self._device)
            if self._old_network != None:
                outputs = self._network(inputs,sigma=self.sigma,reset_fix=True)["logits"][:, :self._total_classes]
            else:
                outputs = self._network(inputs,sigma=0,reset_fix=True)["logits"][:, :self._total_classes]
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            # correct += (predicts.cpu() == targets).sum()
            # total += len(targets)

        return np.concatenate(y_pred), np.concatenate(y_true)

def average_cosine_similarity(A, B):
    # 计算每一对对应向量的余弦相似度
    cos_sim = F.cosine_similarity(A, B, dim=1)
    
    # 对所有相似度取平均值
    avg_cos_sim = torch.mean(cos_sim)
    
    # 归一化到0-1区间
    normalized_similarity = (avg_cos_sim + 1) / 2
    
    return normalized_similarity.item()
def average_kl_divergence(A, B):
    # 将张量转化为概率分布
    A_prob = F.softmax(A, dim=1)
    B_prob = F.softmax(B, dim=1)
    
    # 计算每对对应向量的KL散度
    kl_div = F.kl_div(A_prob.log(), B_prob, reduction='batchmean')
    
    # KL散度天然非负且无界，所以无须额外归一化
    return kl_div.item(),kl_div
def average_mse(A, B):
    # 计算每对对应向量的MSE
    mse = F.mse_loss(A, B, reduction='mean')
    
    # MSE本身在[0, ∞)区间，但为了和其它度量统一，可以考虑根据应用需要进行归一化
    return mse.item(),mse
def average_mae(A, B):
    # 计算每对对应向量的MAE
    mae = F.l1_loss(A, B, reduction='mean')
    
    # MAE本身在[0, ∞)区间，但为了和其它度量统一，可以考虑根据应用需要进行归一化
    return mae.item(),mae


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K-1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]