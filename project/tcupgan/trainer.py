import torch
import os
import tqdm
import numpy as np
import glob
from collections import defaultdict
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from .losses import mink, kl_loss, adv_loss, fc_tversky
from torch.nn.functional import cross_entropy
from einops import rearrange

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer:
    '''
        Trainer module which contains both the full training driver
        which calls the train_batch method
    '''

    kl_beta = 0.5

    disc_alpha = 0.05

    neptune_config = None

    def __init__(self, generator, discriminator, savefolder, device=device):
        '''
            Store the generator and discriminator info
        '''

        generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.generator = generator
        self.discriminator = discriminator
        self.device = device

        if savefolder[-1] != '/':
            savefolder += '/'

        self.savefolder = savefolder
        if not os.path.exists(savefolder):
            os.mkdir(savefolder)

        self.start = 1

    def batch(self, x, y=None, train=False):
        '''
            Train the generator and discriminator on a single batch
        '''

        input_img, _ = x, y

        # convert the input image and mask to tensors
        img_tensor = torch.as_tensor(input_img, dtype=torch.float).to(device)

        x_mu, x_sig, x, c_mu, c_sig, c = self.generator.encode(img_tensor)

        gen_img = self.generator.decode(x, c)

        labels = torch.full((img_tensor.shape[0], img_tensor.shape[1], 1), 1, dtype=torch.float, device=device)

        torch.autograd.set_detect_anomaly(True)

        # Train the generator
        if train:
            self.generator.zero_grad()

        disc_fake = self.discriminator(gen_img)

        gen_loss_MSE = mink(gen_img, img_tensor)

        gen_loss_KL_x = kl_loss(x_mu, x_sig)
        gen_loss_KL_c = kl_loss(c_mu, c_sig)
        gen_loss_KL = self.kl_beta * (gen_loss_KL_x + gen_loss_KL_c)

        gen_loss_disc = adv_loss(disc_fake, labels)

        gen_loss = gen_loss_MSE + self.disc_alpha * gen_loss_disc + gen_loss_KL

        if train:
            gen_loss.backward()
            self.gen_optimizer.step()

        # Train the discriminator
        # On the real image
        if train:
            self.discriminator.zero_grad()
        disc_real = self.discriminator(img_tensor)
        labels.fill_(1)
        loss_real = adv_loss(disc_real, labels)

        if train:
            loss_real.backward()

        # on the generated image
        disc_fake = self.discriminator(gen_img.detach())
        labels.fill_(0)
        loss_fake = adv_loss(disc_fake, labels)

        if train:
            loss_fake.backward()

        disc_loss = loss_fake + loss_real

        if train:
            self.disc_optimizer.step()

        keys = ['gen', 'MSE', 'gdisc', 'KL', 'discr', 'discf', 'disc']
        mean_loss_i = [gen_loss.item(), gen_loss_MSE.item(), gen_loss_disc.item(),
                       gen_loss_KL.item(), loss_real.item(), loss_fake.item(), disc_loss.item()]

        loss = {key: val for key, val in zip(keys, mean_loss_i)}

        return loss

    def train(self, train_data, val_data, epochs, dsc_learning_rate=1.e-4,
              gen_learning_rate=1.e-3, save_freq=10, lr_decay=None, decay_freq=5,
              reduce_on_plateau=False):
        '''
            Training driver which loads the optimizer and calls the
            `train_batch` method. Also handles checkpoint saving
            Inputs
            ------
            train_data : DataLoader object
                Training data that is mapped using the DataLoader or
                MmapDataLoader object defined in patchgan/io.py
            val_data : DataLoader object
                Validation data loaded in using the DataLoader or
                MmapDataLoader object
            epochs : int
                Number of epochs to run the model
            dsc_learning_rate : float [default: 1e-4]
                Initial learning rate for the discriminator
            gen_learning_rate : float [default: 1e-3]
                Initial learning rate for the generator
            save_freq : int [default: 10]
                Frequency at which to save checkpoints to the save folder
            lr_decay : float [default: None]
                Learning rate decay rate (ratio of new learning rate
                to previous). A value of 0.95, for example, would set the
                new LR to 95% of the previous value
            decay_freq : int [default: 5]
                Frequency at which to decay the learning rate. For example,
                a value of for decay_freq and 0.95 for lr_decay would decay
                the learning to 95% of the current value every 5 epochs.
            Outputs
            -------
            G_loss_plot : numpy.ndarray
                Generator loss history as a function of the epochs
            D_loss_plot : numpy.ndarray
                Discriminator loss history as a function of the epochs
        '''

        if (lr_decay is not None) and not reduce_on_plateau:
            gen_lr = gen_learning_rate * (lr_decay)**((self.start - 1) / (decay_freq))
            dsc_lr = dsc_learning_rate * (lr_decay)**((self.start - 1) / (decay_freq))
        else:
            gen_lr = gen_learning_rate
            dsc_lr = dsc_learning_rate

        if self.neptune_config is not None:
            self.neptune_config['model/parameters/gen_learning_rate'] = gen_lr
            self.neptune_config['model/parameters/dsc_learning_rate'] = dsc_lr
            self.neptune_config['model/parameters/start'] = self.start
            self.neptune_config['model/parameters/n_epochs'] = epochs

        # create the Adam optimzers
        self.gen_optimizer = optim.NAdam(
            self.generator.parameters(), lr=gen_lr, betas=(0.9, 0.99))
        self.disc_optimizer = optim.NAdam(
            self.discriminator.parameters(), lr=dsc_lr, betas=(0.9, 0.99))

        # set up the learning rate scheduler with exponential lr decay
        if reduce_on_plateau:
            gen_scheduler = ReduceLROnPlateau(self.gen_optimizer, verbose=True)
            dsc_scheduler = ReduceLROnPlateau(self.disc_optimizer, verbose=True)
            self.neptune_config['model/parameters/scheduler'] = 'ReduceLROnPlateau'
        elif lr_decay is not None:
            gen_scheduler = ExponentialLR(self.gen_optimizer, gamma=lr_decay)
            dsc_scheduler = ExponentialLR(self.disc_optimizer, gamma=lr_decay)
            if self.neptune_config is not None:
                self.neptune_config['model/parameters/scheduler'] = 'ExponentialLR'
                self.neptune_config['model/parameters/decay_freq'] = decay_freq
                self.neptune_config['model/parameters/lr_decay'] = lr_decay
        else:
            gen_scheduler = None
            dsc_scheduler = None

        # empty lists for storing epoch loss data
        D_loss_ep, G_loss_ep = [], []
        for epoch in range(self.start, epochs + 1):
            if isinstance(gen_scheduler, ExponentialLR):
                gen_lr = gen_scheduler.get_last_lr()[0]
                dsc_lr = dsc_scheduler.get_last_lr()[0]
            else:
                gen_lr = gen_learning_rate
                dsc_lr = dsc_learning_rate

            print(f"Epoch {epoch} -- lr: {gen_lr:5.3e}, {dsc_lr:5.3e}")
            print("-------------------------------------------------------")

            # batch loss data
            pbar = tqdm.tqdm(train_data, desc='Training: ', dynamic_ncols=True)

            if hasattr(train_data, 'shuffle'):
                train_data.shuffle()

            # set to training mode
            self.generator.train()
            self.discriminator.train()

            losses = defaultdict(list)
            # loop through the training data
            for i, (input_img, target_img) in enumerate(pbar):

                # train on this batch
                batch_loss = self.batch(input_img, target_img, train=True)

                # append the current batch loss
                loss_mean = {}
                for key, value in batch_loss.items():
                    losses[key].append(value)
                    loss_mean[key] = np.mean(losses[key], axis=0)

                loss_str = " ".join([f"{key}: {value:.2e}" for key, value in loss_mean.items()])

                pbar.set_postfix_str(loss_str)

            # update the epoch loss
            D_loss_ep.append(loss_mean['disc'])
            G_loss_ep.append(loss_mean['gen'])

            if self.neptune_config is not None:
                self.neptune_config['train/gen_loss'].append(loss_mean['gen'])
                self.neptune_config['train/disc_loss'].append(loss_mean['disc'])

            # validate every `validation_freq` epochs
            self.discriminator.eval()
            self.generator.eval()
            pbar = tqdm.tqdm(val_data, desc='Validation: ')

            if hasattr(val_data, 'shuffle'):
                val_data.shuffle()

            losses = defaultdict(list)
            # loop through the training data
            for i, (input_img, target_img) in enumerate(pbar):

                # train on this batch
                batch_loss = self.batch(input_img, target_img, train=False)

                loss_mean = {}
                for key, value in batch_loss.items():
                    losses[key].append(value)
                    loss_mean[key] = np.mean(losses[key], axis=0)

                loss_str = " ".join([f"{key}: {value:.2e}" for key, value in loss_mean.items()])

                pbar.set_postfix_str(loss_str)

            if self.neptune_config is not None:
                self.neptune_config['eval/gen_loss'].append(loss_mean['gen'])
                self.neptune_config['eval/disc_loss'].append(loss_mean['disc'])

            # apply learning rate decay
            if (gen_scheduler is not None) & (dsc_scheduler is not None):
                if isinstance(gen_scheduler, ExponentialLR):
                    if epoch % decay_freq == 0:
                        gen_scheduler.step()
                        dsc_scheduler.step()
                else:
                    gen_scheduler.step(loss_mean['gen'])
                    dsc_scheduler.step(loss_mean['disc'])

            # save checkpoints
            if epoch % save_freq == 0:
                self.save(epoch)

        return G_loss_ep, D_loss_ep

    def save(self, epoch):
        gen_savefile = f'{self.savefolder}/generator_ep_{epoch:03d}.pth'
        disc_savefile = f'{self.savefolder}/discriminator_ep_{epoch:03d}.pth'

        print(f"Saving to {gen_savefile} and {disc_savefile}")
        torch.save(self.generator.state_dict(), gen_savefile)
        torch.save(self.discriminator.state_dict(), disc_savefile)

    def load_last_checkpoint(self):
        gen_checkpoints = sorted(
            glob.glob(self.savefolder + "generator_ep*.pth"))
        disc_checkpoints = sorted(
            glob.glob(self.savefolder + "discriminator_ep*.pth"))

        gen_epochs = set([int(ch.split(
            '/')[-1].replace('generator_ep_', '')[:-4]) for
            ch in gen_checkpoints])
        dsc_epochs = set([int(ch.split(
            '/')[-1].replace('discriminator_ep_', '')[:-4]) for
            ch in disc_checkpoints])

        try:
            assert len(gen_epochs) > 0, "No checkpoints found!"

            start = max(gen_epochs.union(dsc_epochs))
            self.load(f"{self.savefolder}/generator_ep_{start:03d}.pth",
                      f"{self.savefolder}/discriminator_ep_{start:03d}.pth")
            self.start = start + 1
        except Exception as e:
            print(e)
            print("Checkpoints not loaded")

    def load(self, generator_save, discriminator_save):
        print(generator_save, discriminator_save)
        self.generator.load_state_dict(torch.load(generator_save))
        self.discriminator.load_state_dict(torch.load(discriminator_save))

        gfname = generator_save.split('/')[-1]
        dfname = discriminator_save.split('/')[-1]
        print(
            f"Loaded checkpoints from {gfname} and {dfname}")

# custom weights initialization called on generator and discriminator
# scaling here means std


def weights_init(net, init_type='normal', scaling=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, scaling)
        # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, scaling)
            torch.nn.init.constant_(m.bias.data, 0.0)


class TrainerUNet(Trainer):

    tversky_beta = 0.7
    tversky_gamma = 0.5
    seg_alpha = 200
    loss_type = 'tversky'

    def batch(self, x, y, train=False):
        '''
            Train the generator and discriminator on a single batch
        '''
        # convert the input image and mask to tensors
        if not isinstance(x, torch.Tensor):
            img_tensor = torch.as_tensor(x, dtype=torch.float).to(self.device)
            target_tensor = torch.as_tensor(y, dtype=torch.float).to(self.device)
        else:
            img_tensor, target_tensor = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

        gen_img = self.generator(img_tensor)

        disc_inp_fake = torch.cat((img_tensor, gen_img), 2)
        disc_fake = self.discriminator(disc_inp_fake)
        real_labels = torch.ones_like(disc_fake)

        if self.loss_type == 'cross_entropy':
            weight = 1 - torch.sum(target_tensor, axis=(0, 1, 3, 4)) / torch.sum(target_tensor)
            gen_loss_seg = cross_entropy(rearrange(gen_img, "b d c h w -> b c d h w"),
                                         rearrange(target_tensor, "b d c h w -> b c d h w"),
                                         weight=weight) * self.seg_alpha
        elif self.loss_type == 'tversky':
            if gen_img.shape[2] > 1:
                activation = torch.nn.Softmax(dim=2)
            else:
                activation = torch.nn.Sigmoid()
            gen_loss_seg = fc_tversky(target_tensor, activation(gen_img),
                                      beta=self.tversky_beta,
                                      gamma=self.tversky_gamma) * self.seg_alpha
        gen_loss_disc = adv_loss(disc_fake, real_labels)
        gen_loss = gen_loss_seg + gen_loss_disc

        # Train the generator
        if train:
            self.generator.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()

        # Train the discriminator
        # On the real image
        if train:
            self.discriminator.zero_grad()

        disc_inp_real = torch.cat((img_tensor, target_tensor), 2)
        disc_real = self.discriminator(disc_inp_real)

        disc_inp_fake = torch.cat((img_tensor, gen_img.detach()), 2)
        disc_fake = self.discriminator(disc_inp_fake)

        real_labels = torch.ones_like(disc_fake)
        fake_labels = torch.zeros_like(disc_fake)
        loss_real = adv_loss(disc_real, real_labels)
        loss_fake = adv_loss(disc_fake, fake_labels)

        disc_loss = (loss_fake + loss_real) / 2.

        if train:
            disc_loss.backward()
            self.disc_optimizer.step()

        keys = ['gen', self.loss_type, 'gdisc', 'discr', 'discf', 'disc']
        mean_loss_i = [gen_loss.item(), gen_loss_seg.item(), gen_loss_disc.item(),
                       loss_real.item(), loss_fake.item(), disc_loss.item()]

        loss = {key: val for key, val in zip(keys, mean_loss_i)}

        return loss
