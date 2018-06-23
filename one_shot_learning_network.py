from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import time
import argparse
import datetime
from neural_network_architectures import CNNNetwork, VGGLeakyReLULayerNormNetwork
from torch.autograd import Variable
import numpy as np



# def calculate_cosine_distance(support_set_embeddings, target_set_embedding):
#     #print("support_set_embedded vectors", support_set_embeddings.shape)
#     b, ncs_spc, f_s = support_set_embeddings.shape
#     b, f_t = target_set_embedding.shape
#
#     support_set_embeddings = support_set_embeddings.view(b * ncs_spc, f_s)
#     target_set_embedding = target_set_embedding.view(b, 1, f_t).repeat([1, ncs_spc, 1]).view(b * ncs_spc, f_t)
#     cosine_distance = F.cosine_embedding_loss(support_set_embeddings, target_set_embedding, dim=1)
#     cosine_distance = cosine_distance.view(b, ncs_spc)
#     return cosine_distance

def calculate_cosine_distance(support_set_embeddings, target_set_embedding):
    eps = 1e-10
    similarities = []
    #print(support_set_embeddings.shape)
    support_set_embeddings = support_set_embeddings.transpose(0, 1)
    for support_image in support_set_embeddings:
        sum_support = torch.sum(torch.pow(support_image, 2), 1)
        support_magnitude = sum_support.clamp(eps, float("inf")).rsqrt()
        dot_product = target_set_embedding.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
        cosine_similarity = dot_product * support_magnitude
        similarities.append(cosine_similarity)
    similarities = torch.stack(similarities)
    similarities = similarities.transpose(0, 1)
    return similarities

# def cosine_distance_to_preds(similarity_matrix, y_support_set):
#     b, ncs_spc_0, num_classes = y_support_set.shape
#     b, ncs_spc_1 = similarity_matrix.shape
#     similarity_matrix = similarity_matrix.view(b, ncs_spc_0, 1)
#     preds = similarity_matrix * y_support_set.float() #b, ncs, spc, num_classes
#     preds = preds.sum(1)
#     return preds

def cosine_distance_to_preds(similarity_matrix, y_support_set):
    y_support_set = y_support_set.float()
    preds = similarity_matrix.unsqueeze(1).bmm(y_support_set).squeeze()
    return preds

class MatchingNetwork(nn.Module):
    def __init__(self, im_shape, args, use_cuda):

        """
        Builds a matching network, the training and evaluation ops as well as data augmentation routines.
        :param support_set_images: A tensor containing the support set images [batch_size, sequence_size, 28, 28, 1]
        :param support_set_labels: A tensor containing the support set labels [batch_size, sequence_size, 1]
        :param target_image: A tensor containing the target image (image to produce label for) [batch_size, 28, 28, 1]
        :param target_label: A tensor containing the target label [batch_size, 1]
        :param dropout_rate: A tf placeholder of type tf.float32 denotes the amount of dropout to be used
        :param batch_size: The batch size for the experiment
        :param num_channels: Number of channels of the images
        :param is_training: Flag indicating whether we are training or evaluating
        :param rotate_flag: Flag indicating whether to rotate the images
        :param fce: Flag indicating whether to use full context embeddings (i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set: Integer indicating the number of classes per set
        :param num_samples_per_class: Integer indicating the number of samples per class
        """
        super(MatchingNetwork, self).__init__()
        self.batch_size = args.batch_size
        self.fce = args.use_full_context_embeddings
        self.use_cuda = use_cuda
        self.number_of_steps_per_iter = args.number_of_steps_per_iter
        if args.architecture_name == "VGG_batch_norm_net":
            self.classifier = CNNNetwork(im_shape=im_shape, num_output_classes=args.num_classes_per_set,
                                         args=args)
        elif args.architecture_name == "VGG_layer_norm_net":
            self.classifier = VGGLeakyReLULayerNormNetwork(im_shape=im_shape,
                                                           num_output_classes=args.num_classes_per_set,
                                                           args=args)

        self.task_learning_rate = args.task_learning_rate
        task_name_params = self.generator_to_dict(self.classifier.named_parameters())
        print("task params")
        for key, value in task_name_params.items():
            print(key, value.shape)

    def generator_to_dict(self, params):
        param_dict = dict()

        for name, param in params:
            if param.requires_grad:
                param_dict[name] = torch.zeros(param.shape).cuda() + param.cuda()
                #print(name, param.requires_grad)
                #print(name, param.shape)

        return param_dict

    def forward(self, data_batch):
        """
        Builds tf graph for Matching Networks, produces losses and summary statistics.
        :return:
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs
        y_support_set = y_support_set.view(b, ncs * spc, 1)

        [b, num_classes, spc, h, w, c] = x_support_set.shape
        num_tasks = b
        x_support_set = x_support_set.view(size=(b, ncs * spc, h, w, c))
        losses = []
        accuracies = []
        diff = []
        diff_acc = []
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task , y_target_set_task) in \
                                                                                   enumerate(zip(x_support_set,
                                                                                                 y_support_set,
                                                                                                 x_target_set,
                                                                                                 y_target_set)):
            # produce embeddings for support set images
            names_weights_copy = self.generator_to_dict(self.classifier.named_parameters())
            c, h, w = x_target_set_task.shape
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            self.classifier.zero_grad()

            # preds = self.classifier.forward(x=x_support_set_task, params=names_weights_copy)
            #
            # before_loss = F.cross_entropy(input=preds, target=y_support_set_task)
            # _, predicted = torch.max(preds.data, 1)
            # before_accuracy = np.mean(list(predicted.eq(y_support_set_task.data).cpu()))
            #
            # self.classifier.zero_grad()
            #
            # grads = torch.autograd.grad(before_loss, list(names_weights_copy.values()))
            # updated_weights = list(map(lambda p: p[1] - self.meta_learning_rate * p[0],
            #                            zip(grads, list(names_weights_copy.values()))))
            #
            # names_weights_copy = dict(zip(names_weights_copy.keys(), updated_weights))
            #
            #
            # preds = self.classifier.forward(x=x_support_set_task, params=names_weights_copy)
            # after_loss = F.cross_entropy(input=preds, target=y_support_set_task)
            # _, predicted = torch.max(preds.data, 1)
            # after_accuracy = np.mean(list(predicted.eq(y_support_set_task.data).cpu()))
            # diff.append(after_loss.data - before_loss.data)
            # diff_acc.append(after_accuracy - before_accuracy)
            #
            # preds = self.classifier.forward(x=x_target_set_task, params=names_weights_copy)
            # target_loss = F.cross_entropy(input=preds, target=y_target_set_task)
            #
            # _, predicted = torch.max(preds.data, 1)
            # accuracy = list(predicted.eq(y_target_set_task.data).cpu())
            # losses.append(target_loss)
            # accuracies.append(accuracy)

            for num_step in range(self.number_of_steps_per_iter):

                # if num_step == 0:
                #     if self.training:
                #         restore_backup_running_stats = False
                #     else:
                #         restore_backup_running_stats = True
                # else:
                #     restore_backup_running_stats = False
                #
                # if num_step == (self.number_of_steps_per_iter - 1) and task_id == (num_tasks - 1) and self.training:
                #     save_backup_running_stats = True
                # else:
                #     save_backup_running_stats = False
                # save_backup_running_stats = False
                # restore_backup_running_stats = False

                if num_step > 0:
                    restore_backup_running_stats = True
                else:
                    restore_backup_running_stats = False

                if num_step == 1 and self.training:
                    save_backup_running_stats = True
                else:
                    save_backup_running_stats = False

                support_preds = self.classifier.forward(x=x_support_set_task, params=names_weights_copy,
                                                        training=True,
                                                        save_backup_running_stats=save_backup_running_stats,
                                                        restore_backup_running_stats=restore_backup_running_stats)
                #preds = F.softmax(preds)
                y_support_set_task = y_support_set_task.view(-1)
                support_loss = F.cross_entropy(input=support_preds, target=y_support_set_task)
                self.classifier.zero_grad(names_weights_copy)
                grads = torch.autograd.grad(support_loss, list(names_weights_copy.values()))
                updated_weights = list(map(lambda p: p[1] - self.task_learning_rate * p[0], zip(grads,
                                                                                list(names_weights_copy.values()))))
                names_weights_copy = dict(zip(names_weights_copy.keys(), updated_weights))

                #print(x_target_set_task.shape)
            target_preds = self.classifier.forward(x=x_target_set_task, params=names_weights_copy,
                                                   training=False)
            #print(preds.shape)
            target_loss = F.cross_entropy(input=target_preds, target=y_target_set_task)

                #print(len(accuracy))
            losses.append(target_loss)
            _, predicted = torch.max(target_preds.data, 1)
            accuracy = list(predicted.eq(y_target_set_task.data).cpu())
            accuracies.extend(accuracy)

        #print("length", len(losses))
        loss = torch.sum(torch.stack(losses))
        #print("accuracies len", len(accuracies), "end")
        accuracies = np.mean(accuracies)
        diff = np.mean(diff)
        diff_acc = np.mean(diff_acc)
        losses = dict()
        losses['loss'] = loss
        losses['accuracy'] = accuracies
        # losses['diff'] = diff
        # losses['diff_acc'] = diff_acc

        return losses

class MatchingNetworkHandler(object):
    def __init__(self, im_shape, args):
        self.use_cuda = torch.cuda.is_available()
        self.matching_network = MatchingNetwork(im_shape, args=args, use_cuda=self.use_cuda)
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        if self.use_cuda:
            self.matching_network = self.matching_network.cuda()
            # self.matching_network = torch.nn.DataParallel(self.matching_network,
            #                                               device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True
        self.training_mode = False
        print("meta parameters")
        for name, param in self.matching_network.named_parameters():
            if param.requires_grad:
                print(name, param.shape)
        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=True)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

    def trainable_parameters(self):
        for param in self.matching_network.parameters():
            if param.requires_grad:
                if self.use_cuda:
                    yield param.cuda()
                else:
                    yield param


    def compute_losses(self, preds, y_target_set):
        losses = dict()
        loss = self.criterion(preds, y_target_set)

        _, predicted = torch.max(preds.data, 1)
        losses['opt_loss'] = loss
        losses['loss'] = loss.data.item()
        losses['accuracy'] = np.mean(list(predicted.eq(y_target_set.data).cpu()))

        return losses

    def run_train_iter(self, data_batch, epoch):

        """
        Builds the train op
        :param losses: A dictionary containing the losses
        :param learning_rate: Learning rate to be used for Adam
        :param beta1: Beta1 to be used for Adam
        :return:
        """
        self.scheduler.step(epoch=epoch)

        if not self.matching_network.training:
            self.matching_network.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float()
        x_target_set = torch.Tensor(x_target_set).float()
        y_support_set = torch.Tensor(y_support_set).long()
        y_target_set = torch.Tensor(y_target_set).long()

        if self.use_cuda:
            x_support_set = x_support_set.cuda()
            x_target_set = x_target_set.cuda()
            y_support_set = y_support_set.cuda()
            y_target_set = y_target_set.cuda()

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses = self.matching_network.forward(data_batch=data_batch)

        #print("matching_net_parameters", list(self.matching_network.parameters()))
        self.optimizer.zero_grad()
        losses['loss'].backward()  # Backward Propagation
        losses['learning_rate'] = self.scheduler.get_lr()[0]

        self.optimizer.step()  # Optimizer update
        return losses

    def run_validation_iter(self, data_batch):
        """
        Builds the train op
        :param losses: A dictionary containing the losses
        :param learning_rate: Learning rate to be used for Adam
        :param beta1: Beta1 to be used for Adam
        :return:
        """
        if self.matching_network.training:
            self.matching_network.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float()
        x_target_set = torch.Tensor(x_target_set).float()
        y_support_set = torch.Tensor(y_support_set).long()
        y_target_set = torch.Tensor(y_target_set).long()

        if self.use_cuda:
            x_support_set = x_support_set.cuda()
            x_target_set = x_target_set.cuda()
            y_support_set = y_support_set.cuda()
            y_target_set = y_target_set.cuda()

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses = self.matching_network.forward(data_batch=data_batch)

        #print("matching_net_parameters", list(self.matching_network.parameters()))
        self.optimizer.zero_grad()
        # losses['loss'].backward()  # Backward Propagation
        # #losses['learning_rate'] = self.scheduler.get_lr()[0]
        #
        # self.optimizer.step()  # Optimizer update
        return losses

    def save_model(self, model_save_dir, loss, accuracy, iter):
        state = {
            'network': self.matching_network if self.use_cuda else self.matching_network,
            'loss': loss,
            'accuracy': accuracy,
            'iter': iter
        }
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        checkpoint = torch.load(filepath)
        self.matching_network = checkpoint['network']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        start_iter = checkpoint['iter']

        if self.use_cuda:
            self.matching_network.cuda()
            self.matching_network = torch.nn.DataParallel(self.matching_network,
                                                          device_ids=range(torch.cuda.device_count()))

        return start_iter, accuracy, loss
