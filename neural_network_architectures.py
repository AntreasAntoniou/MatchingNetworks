import copy
import numbers
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

def conv_layer(input_shape, num_filters, kernel_size, stride=1, padding=0, use_bias=False):
    b, c, h, w = input_shape
    ##print("conv_input_shape", input_shape)
    ##print(c, num_filters, kernel_size)
    conv_layer = MetaConv2dLayer(in_channels=c, out_channels=num_filters, kernel_size=kernel_size, stride=stride,
                           padding=padding, use_bias=use_bias)
    return conv_layer


class MetaConvReLUBatchNorm(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, batch_norm=True):
        super(MetaConvReLUBatchNorm, self).__init__()
        self.batch_norm = batch_norm
        self.conv = conv_layer(input_shape=input_shape, num_filters=num_filters, kernel_size=kernel_size,
                               stride=stride, padding=padding, use_bias=use_bias)
        if batch_norm:
            self.bn = MetaBatchNormLayer(num_filters, track_running_stats=True)
        self.total_layers = 1

    def forward(self, x, params=None, training=False, save_backup_running_stats=False,
                restore_backup_running_stats=False):
        batch_norm_params = None
        if params is not None:
            if self.batch_norm:
                batch_norm_params = params['bn']
            conv_params = params['conv']
        else:
            conv_params = None

        if self.batch_norm:
            out = self.bn.forward(F.leaky_relu(self.conv.forward(x, conv_params), inplace=True),
                                  params=batch_norm_params, training=training,
                                  save_backup_running_stats=save_backup_running_stats,
                                  restore_backup_running_stats=restore_backup_running_stats)
        else:
            out = F.leaky_relu(self.conv(x, params=conv_params))
        return out

class MetaConvReLULayerNorm(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias, layer_norm=True):
        super(MetaConvReLULayerNorm, self).__init__()
        self.layer_norm = layer_norm
        self.conv = conv_layer(input_shape=input_shape, num_filters=num_filters, kernel_size=kernel_size,
                               stride=stride, padding=padding, use_bias=use_bias)
        input_shape_list = list(input_shape)
        input_shape_list[1] = num_filters
        b, c, h, w = input_shape
        input_shape_list[2] = int(np.ceil(input_shape_list[2] / stride))
        input_shape_list[3] = int(np.ceil(input_shape_list[3] / stride))
        if layer_norm:
            self.bn = MetaLayerNormLayer(normalized_shape=input_shape_list[1:])
        self.total_layers = 1

    def forward(self, x, params=None, save_backup_running_stats=False, restore_backup_running_stats=False):

        if params is not None:
            # if self.batch_norm:
            batch_norm_params = params['bn']
            conv_params = params['conv']
        else:
            conv_params = None
            batch_norm_params = None

        if self.layer_norm:
            out = self.bn(F.leaky_relu(self.conv(x, conv_params)), params=batch_norm_params)
        else:
            out = F.leaky_relu(self.conv(x, params=conv_params))
        return out


class BatchNormReLUConv(nn.Module):
    def __init__(self, input_shape, num_filters, kernel_size, stride, padding, use_bias):
        super(BatchNormReLUConv, self).__init__()

        self.bn = nn.BatchNorm2d(input_shape[1], track_running_stats=True)
        #input_shape[0] = num_filters
        self.conv = conv_layer(input_shape=input_shape, num_filters=num_filters, kernel_size=kernel_size,
                               stride=stride, padding=padding, use_bias=use_bias)

    def forward(self, x):
        out = self.conv(F.leaky_relu(self.bn(x)))
        return out


class MetaConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias):
        super(MetaConv2dLayer, self).__init__()
        num_filters = out_channels
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_normal_(self.weight)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        if params is not None:
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
            else:
                (weight) = params["weight"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=1, groups=1)

        return out


class MetaLinearLayer(nn.Module):
    def __init__(self, input_shape, num_filters, use_bias):
        super(MetaLinearLayer, self).__init__()
        b, c = input_shape
        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(num_filters, c))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x, params=None):
        if params is not None:
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
            else:
                (weight) = params["weights"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None
        out = F.linear(input=x, weight=weight, bias=bias)

        return out

class MetaBatchNormLayer(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(MetaBatchNormLayer, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight.data.uniform_()
            self.bias.data.zero_()
        # else:
        #     self.register_parameter('weight', None)
        #     self.register_parameter('bias', None)
        # if self.track_running_stats:
        #     self.register_buffer('running_mean', torch.zeros(num_features))
        #     self.register_buffer('running_var', torch.ones(num_features))
        # else:
        #     self.register_parameter('running_mean', None)
        #     self.register_parameter('running_var', None)
        self.running_mean = nn.Parameter(torch.Tensor(num_features), requires_grad=False)
        self.running_var = nn.Parameter(torch.Tensor(num_features), requires_grad=False)
        self.running_mean.zero_()
        self.running_var.fill_(1)
        #self.reset_parameters()

    # def reset_parameters(self):
    #     if self.track_running_stats:
    #         self.running_mean.zero_()
    #         self.running_var.fill_(1)
    #     if self.affine:
    #         self.weight.data.uniform_()
    #         self.bias.data.zero_()

    def forward(self, input, params, training=False, save_backup_running_stats=False,
                restore_backup_running_stats=False):

        if params is not None:
            (weight, bias) = params["weight"], params["bias"]
        else:
            weight, bias = self.weight, self.bias

        #print(self.training)
        #print(torch.mean(self.running_mean.data), torch.std(self.running_mean.data))

        if save_backup_running_stats:
            self.backup_running_mean = copy.deepcopy(self.running_mean)
            self.backup_running_var = copy.deepcopy(self.running_var)

        if restore_backup_running_stats:
            self.running_mean = nn.Parameter(self.backup_running_mean, requires_grad=False)
            self.running_var = nn.Parameter(self.backup_running_var, requires_grad=False)

        return \
            F.batch_norm(input, self.running_mean, self.running_var, weight, bias,
            training=training, momentum=self.momentum, eps=self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class MetaLayerNormLayer(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(MetaLayerNormLayer, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input, params):

        if params is not None:
            (weight, bias) = params["weight"], params["bias"]
        else:
            weight, bias = self.weight, self.bias

        #print(self.training)
        #print(torch.mean(self.running_mean.data), torch.std(self.running_mean.data))
        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)



class CNNNetwork(nn.Module):
    def __init__(self, im_shape, num_output_classes, args):
        super(CNNNetwork, self).__init__()
        b, c, self.h, self.w = im_shape
        self.total_layers = 0
        cnn_filters = args.cnn_num_filters
        input_shape = list(im_shape)
        self.conv0 = MetaConvReLUBatchNorm(
                    input_shape=input_shape,
                    num_filters=cnn_filters,
                    kernel_size=3, stride=2,
                    padding=1,
                    use_bias=True, batch_norm=True)
        input_shape[1] = cnn_filters
        input_shape[2] = int(input_shape[2] / 2)
        input_shape[3] = int(input_shape[3] / 2)
        self.conv1 = MetaConvReLUBatchNorm(
            input_shape=input_shape,
            num_filters=cnn_filters,
            kernel_size=3, stride=2,
            padding=1,
            use_bias=True, batch_norm=True)
        input_shape[1] = cnn_filters
        input_shape[2] = int(input_shape[2] / 2)
        input_shape[3] = int(input_shape[3] / 2)
        self.conv2 = MetaConvReLUBatchNorm(
            input_shape=input_shape,
            num_filters=cnn_filters,
            kernel_size=3, stride=2,
            padding=1,
            use_bias=True, batch_norm=True)
        input_shape[1] = cnn_filters
        input_shape[2] = int(input_shape[2] / 2)
        input_shape[3] = int(input_shape[3] / 2)
        self.conv3 = MetaConvReLUBatchNorm(
            input_shape=input_shape,
            num_filters=cnn_filters,
            kernel_size=3, stride=2,
            padding=1,
            use_bias=True, batch_norm=True)
        input_shape[1] = cnn_filters
        input_shape[2] = int(input_shape[2] / 2)
        input_shape[3] = int(input_shape[3] / 2)

        input_shape[1] = cnn_filters
        input_shape[2] = int(input_shape[2] / input_shape[2])
        input_shape[3] = int(input_shape[3] / input_shape[3])
        self.linear = MetaLinearLayer(input_shape=(args.num_classes_per_set, np.prod(input_shape[1:])), num_filters=num_output_classes, use_bias=True)

        
    def forward(self, x, params=None, training=False,
                                  save_backup_running_stats=False,
                                  restore_backup_running_stats=False):
        param_dict = dict()

        if params is None:
            param_dict["conv0"] = None
            param_dict["conv1"] = None
            param_dict["conv2"] = None
            param_dict["conv3"] = None
            param_dict["linear"] = None
        elif type(params) is dict:
            for name, param in params.items():
                path_bits = name.split(".")
                layer_name = path_bits[0]
                sub_layer_name = path_bits[1]
                sub_sub_layer_name = ".".join(path_bits[2:])

                if layer_name not in param_dict:
                    param_dict[layer_name] = dict()

                if sub_layer_name not in param_dict[layer_name]:
                    param_dict[layer_name][sub_layer_name] = dict()

                if sub_sub_layer_name is not "":
                    param_dict[layer_name][sub_layer_name][sub_sub_layer_name] = param
                else:
                    param_dict[layer_name][sub_layer_name] = param
        # else:
        #     for name, param in params:
        #         path_bits = name.split(".")
        #         layer_name = path_bits[0]
        #         sub_layer_name = path_bits[1]
        #         sub_sub_layer_name = ".".join(path_bits[2:])
        #
        #         if layer_name not in param_dict:
        #             param_dict[layer_name] = dict()
        #
        #         if sub_layer_name not in param_dict[layer_name]:
        #             param_dict[layer_name][sub_layer_name] = dict()
        #
        #         if sub_sub_layer_name is not "":
        #             param_dict[layer_name][sub_layer_name][sub_sub_layer_name] = param
        #         else:
        #             param_dict[layer_name][sub_layer_name] = param

        #print(param_dict.keys())

        out = x
        out = self.conv0(out, params=param_dict["conv0"], training=training,
                                  save_backup_running_stats=save_backup_running_stats,
                                  restore_backup_running_stats=restore_backup_running_stats)
        #print(out.shape)
        #out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
        out = self.conv1(out, params=param_dict["conv1"], training=training,
                                  save_backup_running_stats=save_backup_running_stats,
                                  restore_backup_running_stats=restore_backup_running_stats)
        #out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
        out = self.conv2(out, params=param_dict["conv2"], training=training,
                                  save_backup_running_stats=save_backup_running_stats,
                                  restore_backup_running_stats=restore_backup_running_stats)
        #out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
        out = self.conv3(out, params=param_dict["conv3"], training=training,
                                  save_backup_running_stats=save_backup_running_stats,
                                  restore_backup_running_stats=restore_backup_running_stats)
        b, c, h, w = out.shape

        #print(out.shape)
        #out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
        out = F.avg_pool2d(out, out.shape[2])

        out = out.view(out.size(0), -1)
        out = self.linear(out, params=param_dict["linear"])

        return out
    def zero_grad(self, params=None):
        if params is not None:
            param_dict = dict()
            for name, param in params.items():
                if param.grad is not None and param.requires_grad is True:
                    param.grad.zero_()
                path_bits = name.split(".")
                layer_name = path_bits[0]
                sub_layer_name = path_bits[1]
                sub_sub_layer_name = ".".join(path_bits[2:])

                if layer_name not in param_dict:
                    param_dict[layer_name] = dict()

                if sub_layer_name not in param_dict[layer_name]:
                    param_dict[layer_name][sub_layer_name] = dict()

                if sub_sub_layer_name is not "":
                    param_dict[layer_name][sub_layer_name][sub_sub_layer_name] = param
                else:
                    param_dict[layer_name][sub_layer_name] = param
        else:
            param_dict = dict()
            for name, param in self.named_parameters():
                if param.grad is not None and param.requires_grad is True:
                    param.grad.zero_()
                path_bits = name.split(".")
                layer_name = path_bits[0]
                sub_layer_name = path_bits[1]
                sub_sub_layer_name = ".".join(path_bits[2:])

                if layer_name not in param_dict:
                    param_dict[layer_name] = dict()

                if sub_layer_name not in param_dict[layer_name]:
                    param_dict[layer_name][sub_layer_name] = dict()

                if sub_sub_layer_name is not "":
                    param_dict[layer_name][sub_layer_name][sub_sub_layer_name] = param
                else:
                    param_dict[layer_name][sub_layer_name] = param



class VGGLeakyReLULayerNormNetwork(nn.Module):
    def __init__(self, im_shape, num_output_classes, args):
        super(VGGLeakyReLULayerNormNetwork, self).__init__()
        b, c, self.h, self.w = im_shape
        self.total_layers = 0
        cnn_filters = args.cnn_num_filters
        input_shape = list(im_shape)

        self.conv0 = MetaConvReLULayerNorm(
            input_shape=input_shape,
            num_filters=cnn_filters,
            kernel_size=3, stride=2,
            padding=1,
            use_bias=True, layer_norm=True)
        input_shape[1] = cnn_filters
        input_shape[2] = int(input_shape[2] / 2)
        input_shape[3] = int(input_shape[3] / 2)


        self.conv1 = MetaConvReLULayerNorm(
            input_shape=input_shape,
            num_filters=cnn_filters,
            kernel_size=3, stride=2,
            padding=1,
            use_bias=True, layer_norm=True)
        input_shape[1] = cnn_filters
        input_shape[2] = int(input_shape[2] / 2)
        input_shape[3] = int(input_shape[3] / 2)
        self.conv2 = MetaConvReLULayerNorm(
            input_shape=input_shape,
            num_filters=cnn_filters,
            kernel_size=3, stride=2,
            padding=1,
            use_bias=True, layer_norm=True)
        input_shape[1] = cnn_filters
        input_shape[2] = int(input_shape[2] / 2)
        input_shape[3] = int(input_shape[3] / 2)
        self.conv3 = MetaConvReLULayerNorm(
            input_shape=input_shape,
            num_filters=cnn_filters,
            kernel_size=3, stride=2,
            padding=1,
            use_bias=True, layer_norm=True)
        input_shape[1] = cnn_filters
        input_shape[2] = int(input_shape[2] / 2)
        input_shape[3] = int(input_shape[3] / 2)
        #print(input_shape)
        input_shape[1] = cnn_filters
        input_shape[2] = int(input_shape[2] / input_shape[2])
        input_shape[3] = int(input_shape[3] / input_shape[3])
        self.linear = MetaLinearLayer(input_shape=(args.num_classes_per_set, np.prod(input_shape[1:])),
                                      num_filters=num_output_classes, use_bias=True)

    def forward(self, x, params=None, training=False, save_backup_running_stats=False,
                                  restore_backup_running_stats=False):
        param_dict = dict()

        if params is None:
            param_dict["conv0"] = None
            param_dict["conv1"] = None
            param_dict["conv2"] = None
            param_dict["conv3"] = None
            param_dict["linear"] = None
        elif type(params) is dict:
            for name, param in params.items():
                path_bits = name.split(".")
                layer_name = path_bits[0]
                sub_layer_name = path_bits[1]
                sub_sub_layer_name = ".".join(path_bits[2:])

                if layer_name not in param_dict:
                    param_dict[layer_name] = dict()

                if sub_layer_name not in param_dict[layer_name]:
                    param_dict[layer_name][sub_layer_name] = dict()

                if sub_sub_layer_name is not "":
                    param_dict[layer_name][sub_layer_name][sub_sub_layer_name] = param
                else:
                    param_dict[layer_name][sub_layer_name] = param
        # else:
        #     for name, param in params:
        #         path_bits = name.split(".")
        #         layer_name = path_bits[0]
        #         sub_layer_name = path_bits[1]
        #         sub_sub_layer_name = ".".join(path_bits[2:])
        #
        #         if layer_name not in param_dict:
        #             param_dict[layer_name] = dict()
        #
        #         if sub_layer_name not in param_dict[layer_name]:
        #             param_dict[layer_name][sub_layer_name] = dict()
        #
        #         if sub_sub_layer_name is not "":
        #             param_dict[layer_name][sub_layer_name][sub_sub_layer_name] = param
        #         else:
        #             param_dict[layer_name][sub_layer_name] = param

        # print(param_dict.keys())

        out = x
        out = self.conv0(out, params=param_dict["conv0"])
        #out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
        out = self.conv1(out, params=param_dict["conv1"])
        #out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
        out = self.conv2(out, params=param_dict["conv2"])
        #out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
        out = self.conv3(out, params=param_dict["conv3"])
        out = F.avg_pool2d(out, out.shape[2])
        #print(out.shape)
        #out = F.max_pool2d(input=out, kernel_size=(2, 2), stride=2, padding=0)
        out = out.view(out.size(0), -1)
        out = self.linear(out, params=param_dict["linear"])

        return out

    def zero_grad(self, params=None):
        if params is not None:
            param_dict = dict()
            for name, param in params.items():
                if param.grad is not None:
                    param.grad.zero_()
                path_bits = name.split(".")
                layer_name = path_bits[0]
                sub_layer_name = path_bits[1]
                sub_sub_layer_name = ".".join(path_bits[2:])

                if layer_name not in param_dict:
                    param_dict[layer_name] = dict()

                if sub_layer_name not in param_dict[layer_name]:
                    param_dict[layer_name][sub_layer_name] = dict()

                if sub_sub_layer_name is not "":
                    param_dict[layer_name][sub_layer_name][sub_sub_layer_name] = param
                else:
                    param_dict[layer_name][sub_layer_name] = param
        else:
            param_dict = dict()
            for name, param in self.named_parameters():
                if param.grad is not None:
                    param.grad.zero_()
                path_bits = name.split(".")
                layer_name = path_bits[0]
                sub_layer_name = path_bits[1]
                sub_sub_layer_name = ".".join(path_bits[2:])

                if layer_name not in param_dict:
                    param_dict[layer_name] = dict()

                if sub_layer_name not in param_dict[layer_name]:
                    param_dict[layer_name][sub_layer_name] = dict()

                if sub_sub_layer_name is not "":
                    param_dict[layer_name][sub_layer_name][sub_sub_layer_name] = param
                else:
                    param_dict[layer_name][sub_layer_name] = param


# cnn_test = CNNNetwork(im_shape=(128, 3, 28, 28), num_output_classes=100)
# cnn_test.forward(x=torch.zeros(128, 3, 28, 28), params=None)
