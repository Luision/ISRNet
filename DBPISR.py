import torch
import networks2
from torch import nn
from util import im2tensor, tensor2im
from PIL import Image
import numpy as np
import os

class DBPISR:
    def __init__(self, conf):
        # Acquire configuration
        self.conf = conf

        # Define the GAN
        self.net = networks2.InvRescaleNet().cuda()
        # Input tensors
        self.g_input = torch.FloatTensor(1, 3, conf.input_crop_size, conf.input_crop_size).cuda()

        # Define loss function
        self.L1 = nn.L1Loss()

        # Initialize networks2 weightsZ
        self.net.apply(networks2.weights_init_I)
        # Optimizers
        self.optimizer_I = torch.optim.Adam(self.net.parameters(), lr=conf.g_lr, betas=(conf.beta1, 0.999))#weight_decay=0.0033

        print('*' * 60 + '\nSTARTED DBPI-BlindSR on: \"%s\"...' % conf.input_image_path)
        self.ds = nn.AvgPool2d(2, ceil_mode=True)

    def train(self, g_input):
        self.set_input(g_input)
        self.train_g()

    def set_input(self, g_input):
        self.g_input = g_input.contiguous()

    def train_g(self):
        # Zeroize gradients
        self.optimizer_I.zero_grad()

        # Generator forward pass
        # trans = transforms.GaussianBlur(kernel_size=(3, 3))
        g_pred = self.net(self.g_input, rev=True)
        g_downup = self.net(torch.clamp(g_pred.detach(), -1, 1), rev=False)
        g_up = self.net(self.g_input, rev=False)
        g_updown = self.net(torch.clamp(g_up.detach() + 0.1 * torch.randn_like(g_up), -1, 1), rev=True)

        rand_input = torch.rand(3).cuda().view(1, 3, 1, 1,) * 2 - 1
        rand_l = rand_input * torch.ones_like(g_pred)
        rand_h = rand_input * torch.ones_like(self.g_input)
        g_rand = self.net(rand_h, rev=True)
        u_rand = self.net(rand_l, rev=False)

        #loss_ud = self.L1(g_updown, self.g_input)
        loss_du = self.L1(g_rand, rand_l) + 0.1*self.L1(u_rand, rand_h) + self.L1(g_updown, self.g_input) + self.L1(g_downup, self.g_input)

        total_loss = loss_du
        self.loss = total_loss.cpu()
        # Calculate gradients
        total_loss.backward()

        # Update weights
        self.optimizer_I.step()

    def finish(self, image):

        with torch.no_grad():
            self.net.eval()
            image = im2tensor(image)
            sr = self.net(image, rev=False)

            if self.conf.X4:
                sr = im2tensor(tensor2im(sr))
                sr = self.net(sr, rev=False)

            sr = tensor2im(sr)

            def save_np_as_img(arr, path):
                Image.fromarray(np.uint8(arr)).save(path)

            save_np_as_img(sr, os.path.join(self.conf.output_dir_path, 'image sr.png'))
            torch.save(self.net, os.path.join(self.conf.output_dir_path, 'model.pkl'))
            print('FINISHED RUN (see --%s-- folder)\n' % self.conf.output_dir_path + '*' * 60 + '\n\n')
