import argparse
import torch
from utils.utils_data import cifar_dataloader, generate_ooc_partial_matrix
from utils.utils_algo import AverageMeter, cnt_same_idx, generate_random_candidate_labels, cal_wood_loss, cal_ce_loss
import torch.nn.functional as F
from cifar_models.resnet import resnet18
import numpy as np
import random
import math
###################
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='batch_size of ordinary labels.', default=128, type=int)
parser.add_argument('--dataset', help='specify a dataset', default='cifar10', choices=['cifar10'], type=str, required=False)
parser.add_argument('--os_dataset', help='specify a dataset', default='SVHN', choices=['SVHN', 'cifar100', 'ImageNet32'], type=str, required=False)
parser.add_argument('--data_dir', help='data', default='../../datasets/', type=str, required=False)
parser.add_argument('--num_class', help='the number of class', default=10, type=int)
parser.add_argument('--epoch', help='number of epochs', type=int, default=300)
parser.add_argument('--seed', help='Random seed', default=7193, type=int, required=False)
parser.add_argument('--gpu', help='used gpu id', default='0', type=str, required=False)
######
parser.add_argument('--partial_rate', help='partial rate', default=0.1, type=float)
parser.add_argument('--cs_rate', help='rate', default=0.2, type=float)
parser.add_argument('--os_rate', help='rate', default=0.4, type=float)
#######
parser.add_argument('--lr', help='learning rate', default=1e-2, type=float)
parser.add_argument('--weight_decay', help='weight decay', default=1e-3, type=float)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum of SGD solver')
parser.add_argument('--lr_decay_epochs', type=str, default='80,200', help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--cosine', action='store_true', default=True, help='use cosine lr schedule')
######
parser.add_argument('--T', type=float, default=0.5, help='temperature')
parser.add_argument('--warmup', default=30, type=int, help='warm up')
parser.add_argument('--alpha', type=float, default=1, help='loss for cs_ooc')
parser.add_argument('--beta', type=float, default=0.1, help='loss for os_ooc')
parser.add_argument('--eta', type=float, default=0.9, help='ensemble outputs')

#####
args = parser.parse_args()
print(args)
################### 
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
###################
device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
def main():
    #####
    ooc_data,ooc_partial_matrix,true_labels,true_normal_index,true_cs_index,true_os_index,test_data,test_labels=generate_ooc_partial_matrix(args.data_dir,args.os_dataset,args.partial_rate,args.cs_rate,args.os_rate)

    loader = cifar_dataloader(args.batch_size,ooc_data,ooc_partial_matrix,true_labels,true_normal_index,true_cs_index,true_os_index,test_data,test_labels,4)

    warmup_loader = loader.run(mode='warmup')
    eval_loader = loader.run(mode='all_eval')
    test_loader = loader.run(mode='test')
    #####
    tempY = ooc_partial_matrix.sum(dim=1).unsqueeze(1).repeat(1, ooc_partial_matrix.shape[1])
    init_confidence = ooc_partial_matrix.float() / tempY
    init_confidence = init_confidence.to(device)
    ####
    model = resnet18(n_class=args.num_class)
    model = model.to(device)
    #####
    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.weight_decay)
    #####
    test_acc_list = []
    all_outputs=[]
    for epoch in range(args.epoch):
        adjust_learning_rate(optimizer, epoch)
        if epoch<args.warmup:
            warmup_one_epoch(epoch, model, optimizer, warmup_loader, init_confidence)
            selected_normal_index, selected_cs_index, selected_os_index=eval_one_epoch(epoch, model, eval_loader,all_outputs,true_normal_index,true_cs_index, true_os_index,ooc_partial_matrix)
        else:
            selected_normal_index, selected_cs_index, selected_os_index = eval_one_epoch(epoch, model, eval_loader,all_outputs,true_normal_index,true_cs_index, true_os_index,ooc_partial_matrix)
            train_loader = loader.run('all_train', selected_normal_index, selected_cs_index, selected_os_index)
            train_one_epoch(epoch, model, optimizer, train_loader)
        test_one_epoch(epoch, model, test_loader, test_acc_list)
    avg_test_acc = np.mean(test_acc_list[-10:])
    #####
    print("Average Test Accuracy over Last 10 Epochs:{:.4f}".format(avg_test_acc))
    return

def train_one_epoch(epoch,model,optimizer,train_loader):
    model.train()
    all_loss_am = AverageMeter('all_loss', ':2.2f')
    normal_loss_am = AverageMeter('normal_loss', ':2.2f')
    cs_loss_am = AverageMeter('cs_loss', ':2.2f')
    os_loss_am = AverageMeter('os_loss', ':2.2f')
    for i, (images1, images2, labels, true_labels, normal_mask, cs_mask, os_mask, index) in enumerate(train_loader):
        X1, X2, py, ty, normal_mask, cs_mask, os_mask, index = images1.to(device), images2.to(device), labels.long().to(device), true_labels.to(device), normal_mask.to(device), cs_mask.to(device), os_mask.to(device), index.to(device)
        batch_size = X1.size(0)
        outputs1 = model(X1)
        outputs2 = model(X2)
        all_outputs = torch.cat([outputs1, outputs2])
        with torch.no_grad():
            sm_outputs = (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1)) / 2
            candidate_outputs = sm_outputs * py
            candidate_outputs = candidate_outputs ** (1 / args.T)
            candidate_confidence = candidate_outputs/candidate_outputs.sum(dim=1, keepdim=True)
            all_normal_mask = torch.cat([normal_mask, normal_mask])
            #revised label disambiguation for closed-set ooc
            reversedY = ((1 + py) % 2)
            noncandidate_outputs = sm_outputs * reversedY
            noncandidate_outputs = noncandidate_outputs ** (1 / args.T)
            noncandidate_canfidence = noncandidate_outputs/noncandidate_outputs.sum(dim=1, keepdim=True)
            # if no noncandidates
            noncandidate_canfidence[torch.isnan(noncandidate_canfidence)] = 0

            # generate random candidate labels for open-set OOC
            os_targets = generate_random_candidate_labels(batch_size, args.num_class).to(device)

            all_cs_mask = torch.cat([cs_mask, cs_mask])
            all_os_mask = torch.cat([os_mask, os_mask])
            targets = candidate_confidence * normal_mask.unsqueeze(dim=1) + noncandidate_canfidence * cs_mask.unsqueeze(dim=1) + os_targets * os_mask.unsqueeze(dim=1)

        all_targets = torch.cat([targets.detach().clone(), targets.detach().clone()])
        all_loss = -torch.sum(F.log_softmax(all_outputs, dim=1) * all_targets, dim=1)
        # batchsize-wise loss: (loss1+alpha*loss2+beta*loss3)/batchsize
        normal_loss, cs_loss, os_loss = (all_loss*all_normal_mask).mean(), (all_loss*all_cs_mask).mean(), (all_loss*all_os_mask).mean()
        loss = normal_loss+args.alpha*cs_loss+args.beta*os_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss_am.update(loss.item())
        normal_loss_am.update(normal_loss.item())
        cs_loss_am.update(cs_loss.item())
        os_loss_am.update(os_loss.item())
    ######
    print('Train Epoch [{}]: lr:{:.8f}. all_loss:{:.4f}. normal_loss:{:.4f}. cs_loss:{:.4f}. os_loss:{:.4f}.'.format(epoch + 1, optimizer.param_groups[0]['lr'], all_loss_am.avg, normal_loss_am.avg, cs_loss_am.avg, os_loss_am.avg))
    return

def warmup_one_epoch(epoch,model,optimizer,eval_loader,confidence):
    model.train()
    loss_cls = AverageMeter('loss@cls', ':2.2f')
    for i, (image1, image2, labels, true_labels, index) in enumerate(eval_loader):
        image1, image2, Y, ty, index = image1.to(device),image2.to(device), labels.long().to(device), true_labels.to(device), index.to(device)
        outputs1 = model(image1)
        outputs2 = model(image2)
        all_outputs = torch.cat([outputs1, outputs2])
        all_confidence = torch.cat([confidence[index], confidence[index]])
        loss = -torch.mean(torch.sum(F.log_softmax(all_outputs, dim=1) * all_confidence, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_cls.update(loss.item())
        #update label confidence
        with torch.no_grad():
            sm_outputs1 = (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1)) / 2
            sm_outputs1 = sm_outputs1 * Y
            score1 = sm_outputs1 / sm_outputs1.sum(dim=1, keepdim=True)
            confidence[index] = score1
    print('Warmup Epoch [{}]: lr:{:.8f}. loss:{:.4f}.'.format(epoch + 1, optimizer.param_groups[0]['lr'], loss_cls.avg,))

def eval_one_epoch(epoch,model,eval_loader,all_outputs,true_normal_index,true_cs_index,true_os_index,partialY):
    model.eval()
    num_data = 50000+int(args.os_rate*50000)
    epoch_outputs = torch.zeros(num_data,args.num_class)
    with torch.no_grad():
        for i,(images,labels,true_labels,index) in enumerate(eval_loader):
            X,index = images.to(device),index.to(device)
            outputs = model(X)
            for b in range(X.size(0)):
                epoch_outputs[index[b]]=outputs[b].detach().clone().cpu()
    if epoch < args.warmup:
        all_outputs.append(epoch_outputs)
        ensemble_outputs = epoch_outputs.detach().clone()
    else:
        history_outputs = torch.from_numpy(np.vstack(all_outputs)).view(len(all_outputs), num_data, -1)
        ensemble_epoch = 20 if args.partial_rate == 0.5 else 5
        ensemble_outputs = history_outputs[-ensemble_epoch:].mean(dim=0).detach().clone()

    ensemble_outputs = args.eta * ensemble_outputs + (1 - args.eta) * epoch_outputs
    epoch_candidate_loss, epoch_noncandidate_loss = cal_wood_loss(ensemble_outputs, partialY)
    #OOC selection
    selected_normal_index = (epoch_noncandidate_loss-epoch_candidate_loss).sort(descending=True)[1][0:int((1-args.cs_rate) * 50000)]
    selected_cs_index = (epoch_noncandidate_loss-epoch_candidate_loss).sort(descending=False)[1][0:int(args.cs_rate * 50000)]
    selected_os_index = (epoch_noncandidate_loss+epoch_candidate_loss).sort(descending=True)[1][0:int(args.os_rate * 50000)]
    #evaluation
    selected_num_true_cs = cnt_same_idx(true_cs_index, selected_cs_index)
    selected_num_true_os = cnt_same_idx(true_os_index, selected_os_index)
    selected_num_true_normal = cnt_same_idx(true_normal_index, selected_normal_index)
    precision_cs = selected_num_true_cs/len(selected_cs_index)
    precision_os = selected_num_true_os/len(selected_os_index)
    precision_normal = selected_num_true_normal / len(selected_normal_index)

    print('Evaluation Epoch [%d]: normal(Precision:[%.4f]) cs_ooc(Precision:[%.4f]) os_ooc(Precision:[%.4f])'%(epoch + 1,precision_normal,precision_cs,precision_os))

    return selected_normal_index, selected_cs_index, selected_os_index

def test_one_epoch(epoch,model,test_loader,test_acc_list):
    model.eval()
    total, num_samples = 0, 0
    for i,(images, labels) in enumerate(test_loader):
        labels, images = labels.to(device), images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += (predicted == labels).sum().item()
        num_samples += labels.size(0)
    test_acc= total / num_samples
    test_acc_list.extend([test_acc])
    print('Test  Epoch [{}]: Test Acc: {:.2%}.'.format(epoch + 1,test_acc))

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epoch)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()


