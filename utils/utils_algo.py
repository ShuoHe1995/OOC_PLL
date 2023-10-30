import numpy as np
import torch
import torch.nn.functional as F
import random
import pickle


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
def accuracy_check(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs,_ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)
    return total / num_samples

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1, )).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

def generate_uniform_cv_candidate_labels(train_labels, partial_rate=0.1):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0
    p_1 = partial_rate
    transition_matrix =  np.eye(K)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0],dtype=bool))]=p_1
    #print(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    #print("Finish Generating Candidate Label Sets!\n")
    return partialY

def generate_ooc(data,os_data,partialY,true_labels,cs_rate=0.1,os_rate=0.2,partial_rate=0.4):
    num1=data.shape[0]
    num2=os_data.shape[0]
    num_cs=int(num1*cs_rate)
    num_os=int(num1*os_rate)
    random_index=torch.randperm(num1)
    index_cs=random_index[0:num_cs]
    index_normal=random_index[num_cs:num1]
    index_os=torch.randperm(num2)[0:num_os]
    #####
    index_of_no_noncandidate=[]
    for i,index in enumerate(index_cs):
        non_candidate_labels=torch.nonzero(partialY[index]==0).squeeze(dim=1)
        if non_candidate_labels.shape[0]==0: ### no non-candidate label
            index_of_no_noncandidate.append(i)
            continue
        else:
            ooc_label_index=random.randint(0,non_candidate_labels.shape[0]-1)
            partialY[index][true_labels[index]]=0
            partialY[index][non_candidate_labels[ooc_label_index]]=1
    # no non-candidate example is not closed-set ooc.
    temp = np.delete(index_cs.numpy(), index_of_no_noncandidate)
    index_cs = torch.from_numpy(temp)

    new_data = np.concatenate((data, os_data[index_os]), axis=0)
    os_partialY = generate_random_candidate_labels(num_os, 10, partial_rate, False)
    new_partialY = torch.cat([partialY, os_partialY])
    index_os = torch.arange(num1, num1+num_os)
    os_random_true_labels = generate_one_random_label(num_os, 10).max(dim=1)[1]
    #####
    return new_data, new_partialY, index_normal, index_cs, index_os, os_random_true_labels

def return_same_idx(a,b):
    uniset,cnt=torch.cat([a,b]).unique(return_counts=True)
    result=torch.nonzero(cnt==2).squeeze(dim=1)
    return uniset[result]

def cnt_same_idx(a,b):
    uniset, cnt = torch.cat([a,b]).unique(return_counts=True)
    result = torch.nonzero(cnt==2)
    return len(result)

def cal_ce_loss(outputs,Y):

    reversedY = ((1 + Y) % 2)
    logsm_outputs = F.log_softmax(outputs, dim=1)
    candidate_loss = (-logsm_outputs * Y).sum(dim=1) / Y.sum(dim=1)
    noncandidate_loss = (-logsm_outputs * reversedY).sum(dim=1) / reversedY.sum(dim=1)

    return candidate_loss, noncandidate_loss

def cal_wood_loss(outputs,Y):
    max_value=1000
    reversedY = ((1 + Y) % 2)
    logsm_outputs = F.log_softmax(outputs, dim=1)
    candidate_loss = -logsm_outputs * Y
    candidate_loss[candidate_loss==0] = max_value
    min_candidate_loss = candidate_loss.min(dim=1)[0]
    noncandidate_loss = -logsm_outputs * reversedY
    noncandidate_loss[noncandidate_loss==0]=max_value
    min_noncandidate_loss = noncandidate_loss.min(dim=1)[0]

    return min_candidate_loss, min_noncandidate_loss


def generate_random_candidate_labels(num_sample, num_class, a=0.5, normalize=True):
    prob = np.random.uniform(0, 1, size=(num_sample,num_class))
    random_targets=torch.zeros(num_sample,num_class)
    for i in range(num_sample):
        for j in range(num_class):
            if prob[i][j] < a:
                random_targets[i][j] = 1
        if random_targets[i].sum() == 0: # if no candidate label
            random_index = random.randint(0, num_class-1)
            random_targets[i][random_index] = 1
    if normalize:
        random_targets = random_targets/random_targets.sum(dim=1,keepdim=True)
    return random_targets

def generate_one_random_label(num_sample,num_class):
    random_targets = torch.zeros(num_sample, num_class)
    for i in range(num_sample):
        random_label=random.randint(0,num_class-1)
        random_targets[i][random_label] = 1
    return random_targets


