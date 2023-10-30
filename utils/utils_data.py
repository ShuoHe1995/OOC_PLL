import numpy as np
import torch
import os
import pickle
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from utils.utils_algo import generate_uniform_cv_candidate_labels, generate_ooc
import random

def generate_ooc_partial_matrix(data_dir,os_dataset_name,partial_rate,cs_rate,os_rate):
    original_train_cifar10 = dsets.CIFAR10(root=data_dir, train=True, transform=transforms.ToTensor(), download=False)
    original_test_cifar10 = dsets.CIFAR10(root=data_dir, train=False, transform=transforms.ToTensor(), download=False)
    train_data=original_train_cifar10.data
    train_labels=torch.Tensor(original_train_cifar10.targets).long()
    test_data=original_test_cifar10.data
    test_labels=original_test_cifar10.targets
    if os_dataset_name == 'cifar100':
        os_dataset = dsets.CIFAR100(root=data_dir, train=True, download=False, transform=transforms.ToTensor())
    elif os_dataset_name == 'SVHN':
        os_dataset = dsets.SVHN(root=data_dir + 'SVHN', split='train', transform=transforms.ToTensor(), download=False)
        os_dataset.data=os_dataset.data.transpose((0, 2, 3, 1))
    elif os_dataset_name == 'ImageNet32':
        os_dataset = ImageNet32(root=data_dir + 'ImageNet', transform=transforms.ToTensor())
    os_data = os_dataset.data
    ######
    partialY = generate_uniform_cv_candidate_labels(train_labels, partial_rate)
    ###### CS_OOC and OS_OOC
    ooc_data,ooc_partial_matrix,normal_index,cs_index,os_index,os_random_true_labels=generate_ooc(train_data,os_data,partialY,train_labels,cs_rate,os_rate,partial_rate)
    all_true_labels = torch.cat([train_labels, os_random_true_labels])
    ######
    return ooc_data, ooc_partial_matrix, all_true_labels, normal_index, cs_index, os_index, test_data, test_labels


class cifar_dataloader():
    def __init__(self,batch_size,ooc_data,ooc_partial_matrix,true_labels,true_normal_index,true_cs_index,true_os_index,test_data,test_labels,num_workers=0):
        self.batch_size = batch_size
        self.ooc_data=ooc_data
        self.ooc_partial_matrix=ooc_partial_matrix
        self.true_labels=true_labels
        self.true_normal_index=true_normal_index
        self.true_cs_index=true_cs_index
        self.true_os_index=true_os_index
        self.test_data=test_data
        self.test_labels=test_labels
        self.num_workers = num_workers

        # data augmentation
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),
        ])



    def run(self,mode,selected_normal_index=[],selected_cs_index=[],selected_os_index=[]):
        if mode == 'test':
            test_dataset = ooc_partial_cifar10(self.test_data,self.test_labels,self.test_transform,mode,self.ooc_partial_matrix)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)
            return test_loader
        elif mode=='all_train':
            ooc_partial_dataset = ooc_partial_cifar10(self.ooc_data,self.true_labels,self.train_transform,mode,self.ooc_partial_matrix,selected_normal_index,selected_cs_index,selected_os_index)
            all_train_loader = torch.utils.data.DataLoader(dataset=ooc_partial_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
            return all_train_loader
        elif mode=='all_eval':
            all_eval_dataset = ooc_partial_cifar10(self.ooc_data,self.true_labels,self.test_transform,mode,self.ooc_partial_matrix)
            all_eval_loader = torch.utils.data.DataLoader(dataset=all_eval_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)
            return all_eval_loader
        elif mode=='warmup':
            all_eval_dataset = ooc_partial_cifar10(self.ooc_data,self.true_labels,self.train_transform,mode,self.ooc_partial_matrix)
            all_eval_loader = torch.utils.data.DataLoader(dataset=all_eval_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
            return all_eval_loader


class ooc_partial_cifar10(dsets.CIFAR10):
    def __init__(self,data,true_labels,transform,mode='all_train',ooc_partial_matrix=[],selected_normal_index=[],selected_cs_index=[],selected_os_index=[]):
        self.mode=mode
        self.transform=transform
        if self.mode!='test':
            self.selected_normal_index = selected_normal_index
            self.selected_cs_index=selected_cs_index
            self.selected_os_index=selected_os_index
            self.train_data=data
            self.ooc_partial_matrix = ooc_partial_matrix
            self.true_labels = true_labels
        else:
            self.test_data=data
            self.test_labels=true_labels
    def __len__(self):
        if self.mode!='test':
            return len(self.train_data)
        else:
            return len(self.test_data)
    def __getitem__(self, index):
        if self.mode == 'all_eval':
            each_image = self.train_data[index]
            eval_image = self.transform(each_image)
            each_label = self.ooc_partial_matrix[index]
            each_true_label = self.true_labels[index]
            return eval_image,each_label,each_true_label,index
        elif self.mode == 'warmup':
            each_label = self.ooc_partial_matrix[index]
            each_true_label = self.true_labels[index]
            each_image = self.train_data[index]
            each_image1 = self.transform(each_image)
            each_image2 = self.transform(each_image)
            return each_image1, each_image2, each_label, each_true_label, index
        elif self.mode == 'all_train':
            each_label = self.ooc_partial_matrix[index]
            each_true_label = self.true_labels[index]
            each_image = self.train_data[index]
            each_image1 = self.transform(each_image)
            each_image2 = self.transform(each_image)
            ###
            if index in self.selected_cs_index:
                selected_cs = 1
            else:
                selected_cs = 0
            ###
            if index in self.selected_os_index:
                selected_os = 1
            else:
                selected_os = 0
            if index in self.selected_normal_index:
                selected_normal = 1
            else:
                selected_normal = 0
            ###
            return each_image1, each_image2, each_label, each_true_label, selected_normal, selected_cs, selected_os, index
        elif self.mode == 'test':
            image=self.test_data[index]
            each_image = self.transform(image)
            label=self.test_labels[index]
            return each_image, label

class ImageNet32():
    def __init__(self, root, transform ):
        self.root=root
        self.transform=transform
        self.data=[]
        self.targets=[]
        data, target = self.load_databatch(random.randint(1, 10))
        self.data.append(data)
        self.targets.extend(target)
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data=(self.data * 255).astype(np.uint8)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        each_label = self.targets[index]
        image = self.data[index]
        each_image = self.transform(image)
        return each_image, each_label

    def load_databatch(self,idx, img_size=32):
        data_file = os.path.join(self.root, 'train_data_batch_'+str(idx))

        with open(data_file, "rb") as f:
            d = pickle.load(f, encoding="latin1")
            x = d['data']
            y = d['labels']
            mean_image = d['mean']

        x = x / np.float32(255)
        mean_image = mean_image / np.float32(255)

        # Labels are indexed from 1, shift it so that indexes start at 0
        y = [i - 1 for i in y]
        data_size = x.shape[0]

        x -= mean_image

        img_size2 = img_size * img_size

        x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

        # create mirrored images
        X_train = x[0:data_size, :, :, :]
        Y_train = y[0:data_size]
        X_train_flip = X_train[:, :, :, ::-1]
        Y_train_flip = Y_train
        X_train = np.concatenate((X_train, X_train_flip), axis=0)
        Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

        # return dict(
        #     X_train=X_train,
        #     Y_train=Y_train.astype('int32'),
        #     mean=mean_image)

        return X_train, Y_train






