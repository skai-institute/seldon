# Adapted From: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html
import torch
import numpy as np
import math
import random
import torch.nn.functional as F
from torch import nn
from TAE.util import register_module

class Sampler:

    def __init__(self, model, latent_dim, sample_size, condition_dim=0, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.sample_size = sample_size
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.max_len = max_len
        self.examples = [(torch.randn((1, latent_dim))) for _ in range(self.sample_size)]

    def sample_new_exmps(self, condition=None, steps=60, step_size=10, device=None):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.randn((n_new, self.latent_dim))
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

        # Perform MCMC sampling
        inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size, condition=condition)

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(model, inp_latent, steps=60, step_size=10, return_img_per_step=False, condition=None):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_latent.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_latent.shape, device=inp_latent.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            noise.normal_(0, 0.005)
            inp_latent.data.add_(noise.data)
            inp_latent.data.clamp_(min=-5.0, max=5.0)

            # Part 2: calculate gradients for the current input.
            if condition is None:
                out_latent = -model(inp_latent)
            else:
                condition = condition.clone().detach()
                out_latent = -model(torch.cat([inp_latent, condition], dim=-1))
            out_latent.sum().backward()
            inp_latent.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_latent.data.add_(-step_size * inp_latent.grad.data)
            inp_latent.grad.detach_()
            inp_latent.grad.zero_()
            inp_latent.data.clamp_(min=-5.0, max=5.0)

            if return_img_per_step:
                imgs_per_step.append(inp_latent.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_latent

@register_module
class BlockedGibbsSampler:
    '''Sample from joint distribution over a discrete plus continuous variable'''

    def __init__(self, model, latent_dim, num_classes, sample_size, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            latent_dim - Shape of the latent dim
            num_classes - Number of unique class labels
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.sample_size = sample_size
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.num_classes = num_classes
        self.softmax = nn.Softmax(-1)
        self.example_latents = [torch.randn((1, latent_dim)) for _ in range(self.sample_size)]
        self.example_classes = [F.one_hot(torch.randint(0, self.num_classes, (1,), dtype=torch.long), num_classes=self.num_classes) for _ in range(self.sample_size)]

    def sample_new_exmps(self, steps=60, step_size=10, device=None, temperature=1.0):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = min(np.random.binomial(self.sample_size, 0.05), 1)
        rand_classes = F.one_hot(torch.randint(0, self.num_classes, (n_new,), dtype=torch.long), num_classes=self.num_classes)
        rand_latents = torch.randn((n_new, self.latent_dim))
        old_latents = torch.cat(random.choices(self.example_latents, k=self.sample_size-n_new), dim=0)
        old_classes = torch.cat(random.choices(self.example_classes, k=self.sample_size-n_new), dim=0)
        inp_latents = torch.cat([rand_latents, old_latents], dim=0).detach().to(device)
        inp_classes = torch.cat([rand_classes, old_classes], dim=0).detach().to(device)

        # Perform MCMC sampling
        inp_latents, inp_classes = self.generate_samples(inp_latents, inp_classes, steps=steps, step_size=step_size, temperature=temperature)

        # Add new images to the buffer and remove old ones if needed
        self.example_latents = list(inp_latents.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.example_latents
        self.example_latents = self.example_latents[:self.max_len]

        self.example_classes = list(inp_classes.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.example_classes
        self.example_classes = self.example_classes[:self.max_len]
        return inp_latents, inp_classes

 
    def generate_samples(self, inp_latent, inp_classes, steps=60, step_size=10, return_per_step=False, device=None, temperature=1.0):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_latent - latent variables to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        inp_latent.requires_grad = True
        #inp_classes.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_latent.shape, device=inp_latent.device)

        # List for storing generations at each step (for later analysis)
        latents_per_step = []
        classes_per_step = []

        # Loop over K (steps)
        for _ in range(steps):

            # Part 1: Add noise to the input.
            noise.normal_(0, 1.0)
            inp_latent.data.add_(noise.data * math.sqrt(step_size))
            #inp_latent.data.clamp_(min=-5.0, max=5.0)

            # Part 2: calculate gradients for the current input.
            out_latent = -self.model(torch.cat([inp_latent, inp_classes], dim=-1))/temperature
            out_latent.sum().backward()
            #inp_latent.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_latent.data.add_(-0.5 * step_size * inp_latent.grad.data)
            inp_latent.grad.detach_()
            inp_latent.grad.zero_()
            #inp_latent.data.clamp_(min=-5.0, max=5.0)

            if return_per_step:
                latents_per_step.append(inp_latent.clone().detach())
                classes_per_step.append(inp_classes.clone().detach())
            current_class = inp_classes.clone()

            # Part 3: Sample one_hot class label from softmax(-E(z, c)), note in our notation energy = -E
            class_table = torch.eye(self.num_classes, device=inp_latent.device)
            class_matrix = class_table[None, :, :].expand(inp_latent.shape[0], -1, -1) # batch_size x all_class_dim x true_class_dim
            energy_c_given_z = self.model(torch.cat([inp_latent[:, None, :].expand(-1, self.num_classes, -1), class_matrix], dim=-1)).sum(dim=-1)/temperature # batch_size x all_class_dim
            assert torch.all(torch.isfinite(energy_c_given_z)), (energy_c_given_z, inp_latent)
            p_c_given_z = self.softmax(energy_c_given_z)
            assert torch.all(torch.isfinite(p_c_given_z)), (energy_c_given_z, inp_latent)
            class_sample_index = torch.multinomial(p_c_given_z, 1).squeeze(1) # (batch_size)
            inp_classes = F.one_hot(class_sample_index, num_classes=self.num_classes).to(torch.float32) # (batch_size x num_classes)
            
            

        # Reactivate gradients for parameters for training
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_per_step:
            return torch.stack(latents_per_step, dim=0), torch.stack(classes_per_step, dim=0)
        else:
            return inp_latent, current_class


@register_module
class BlockedGibbsEmbeddingSampler:
    '''Sample from joint distribution over a discrete plus continuous variable
    assumes model has a class_embedding layer with that name
    '''

    def __init__(self, model, latent_dim, num_classes, sample_size, max_len=8192, MALA=False):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            latent_dim - Shape of the latent dim
            num_classes - Number of unique class labels
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.sample_size = sample_size
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.num_classes = num_classes
        self.softmax = nn.Softmax(-1)
        self.log_softmax = nn.LogSoftmax(-1)
        self.MALA = MALA
        self.example_latents = [torch.randn((1, latent_dim)) for _ in range(self.sample_size)]
        self.example_classes = [torch.randint(0, self.num_classes, (1,), dtype=torch.long) for _ in range(self.sample_size)]

    def sample_new_exmps(self, steps=60, step_size=10, device=None, temperature=1.0):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = min(np.random.binomial(self.sample_size, 0.05), 1)
        rand_classes = torch.randint(0, self.num_classes, (n_new,), dtype=torch.long)
        rand_latents = torch.randn((n_new, self.latent_dim))
        old_latents = torch.cat(random.choices(self.example_latents, k=self.sample_size-n_new), dim=0)
        old_classes = torch.cat(random.choices(self.example_classes, k=self.sample_size-n_new), dim=0)
        inp_latents = torch.cat([rand_latents, old_latents], dim=0).detach().to(device)
        inp_classes = torch.cat([rand_classes, old_classes], dim=0).detach().to(device)

        # Perform MCMC sampling
        inp_latents, inp_classes = self.generate_samples(inp_latents, inp_classes, steps=steps, step_size=step_size, temperature=temperature, return_index=True)

        # Add new images to the buffer and remove old ones if needed
        self.example_latents = list(inp_latents.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.example_latents
        self.example_latents = self.example_latents[:self.max_len]

        self.example_classes = list(inp_classes.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.example_classes
        self.example_classes = self.example_classes[:self.max_len]
        return inp_latents, inp_classes

 
    def generate_samples(self, inp_latent, inp_classes, steps=60, step_size=10, return_per_step=False, device=None, temperature=1.0, return_index=False, sample_classes=True, sample_latent=True):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_latent - latent variables to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = self.model.training
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        inp_latent.requires_grad = True
        #inp_classes.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_latent.shape, device=inp_latent.device)

        # List for storing generations at each step (for later analysis)
        latents_per_step = []
        classes_per_step = []

        # Loop over K (steps)
        for _ in range(steps): # NOTE: Step size is 2*tau

            if self.MALA:
                x0 = inp_latent.clone().detach()
                x0.requires_grad = True
                c0 = inp_classes.clone()

            if sample_latent:
                # Part 1: Add noise to the input.
                noise.normal_(0, 1.0)
                inp_latent.data.add_(noise.data * math.sqrt(step_size))
                #inp_latent.data.clamp_(min=-5.0, max=5.0)

                # Part 2: calculate gradients for the current input.
                out_latent = -self.model(torch.cat([inp_latent, self.model.class_embedding(inp_classes)], dim=-1))/temperature
                out_latent.sum().backward()
                #inp_latent.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

                # Apply gradients to our current samples
                inp_latent.data.add_(-0.5 * step_size * inp_latent.grad.data)
                inp_latent.grad.detach_()
                inp_latent.grad.zero_()
            #inp_latent.data.clamp_(min=-5.0, max=5.0)

            if return_per_step:
                latents_per_step.append(inp_latent.clone().detach())
                classes_per_step.append(self.model.class_embedding(inp_classes.clone().detach()))

            if return_index:
                current_class = inp_classes.clone()
            else:
                current_class = self.model.class_embedding(inp_classes.clone())

            # Part 3: Sample one_hot class label from softmax(-E(z, c)), note in our notation energy = -E
            if sample_classes:
                batch_size, latent_dim = inp_latent.shape
                class_idx = torch.arange(self.num_classes, device=inp_latent.device).reshape(1, -1) # 1 x Trial
                class_trials = self.model.class_embedding(class_idx).expand(batch_size, -1, -1).to(inp_latent.device) # B x Trial x Emb
                continuum_trials = inp_latent.reshape(batch_size, 1, latent_dim).expand(-1, self.num_classes, -1) # B x Trial x Latent
                energy_inputs = torch.cat([continuum_trials, class_trials], dim=-1)
                energy_c_given_z = self.model(energy_inputs).sum(dim=-1) / temperature# batch_size x Trials, sum gets rid of last dim
                inp_classes = torch.multinomial(self.softmax(energy_c_given_z), 1).squeeze(1)
                #class_sample_index
                #inp_classes = torch.argmax(energy_c_given_z, dim=-1)

            if self.MALA:
                q0 = 0
                q1 = 0
                if sample_latent:
                    x0.requires_grad = True
                    E = self.model(torch.cat([x0, self.model.class_embedding(c0)], dim=-1)) / temperature
                    E.sum().backward()
                    
                    
                    #import sys
                    #print(x0.grad, file=sys.stderr)
                    del_x0 = x0.grad.data.detach()
                    #print(del_x0, file=sys.stderr)
                    x0.grad.detach_()
                    x0.grad.zero_()
                    E0 = E.detach()

                    x1 = inp_latent.clone().detach()
                    x1.requires_grad = True
                    c1 = inp_classes.clone()
                    E = self.model(torch.cat([x1, self.model.class_embedding(c1)], dim=-1)) / temperature
                    E.sum().backward()
                    
                    del_x1 = x1.grad.data.detach()
                    x1.grad.detach_()
                    x1.grad.zero_()
                    E1 = E.detach()

                    log_q0_latent = -torch.square((x1 - x0 - step_size / 2 * del_x0)).sum(-1) * 1 / (2 * step_size)
                    log_q1_latent = -torch.square((x0 - x1 - step_size / 2 * del_x1)).sum(-1) * 1 / (2 * step_size)

                    q0 = q0 + log_q0_latent.exp()
                    q1 = q1 + log_q1_latent.exp()
                if sample_classes:
                    batch_size, latent_dim = inp_latent.shape
                    class_idx = torch.arange(self.num_classes, device=inp_latent.device).reshape(1, -1) # 1 x Trial
                    class_trials = self.model.class_embedding(class_idx).expand(batch_size, -1, -1).to(inp_latent.device) # B x Trial x Emb
                    continuum_trials = x0.reshape(batch_size, 1, latent_dim).expand(-1, self.num_classes, -1) # B x Trial x Latent
                    energy_inputs = torch.cat([continuum_trials, class_trials], dim=-1)
                    energy_c_given_z = self.model(energy_inputs).sum(dim=-1) # batch_size x Trials, sum gets rid of last dim
                    p_c0 = self.softmax(energy_c_given_z)

                    continuum_trials = x1.reshape(batch_size, 1, latent_dim).expand(-1, self.num_classes, -1) # B x Trial x Latent
                    energy_inputs = torch.cat([continuum_trials, class_trials], dim=-1)
                    energy_c_given_z = self.model(energy_inputs).sum(dim=-1) # batch_size x Trials, sum gets rid of last dim
                    p_c1 = self.softmax(energy_c_given_z)

                    # Figure out the grad of the transition probabilities between embedding vectors
                    # c0 = [0, 0, 0, 1, 0, 0]
                    # c1 = [1, 0, 0, 0, 0, 0]
                    # Think about c_i -> c_k probs.  For each, i->k there is a prob of going to each k.  But our input is a constant i
                    # so we need to compute the probs of going from c -> c_i
                    # The log probability grad will be 1/p(c) * (p(c) * (one_hot(c)*(one_hot(c_i) - p(c_i)))
                    c0_onehot = F.one_hot(c0, num_classes=self.num_classes)
                    c1_onehot = F.one_hot(c1, num_classes=self.num_classes)
                    grad_c0 = c0_onehot - p_c0 # Might have to swap the one-hots
                    grad_c1 = c1_onehot - p_c1

                    # c1_onehot * (p0 + (c0_onehot - p0) )
                    # p(c1|c0) = p0 * c1_onehot
                    # updated = p0*c1_onehot + grad_c0*c1_one_hot 
                    # = c1_onehot * (p0 + grad_c0*dt) = c1_one_hot * (p0 *(1-dt)+c0_onehot)
                    # = c_1_onehot * (p0 * (1 - dt))
                    
                    q0_class = (c1_onehot * (p_c0 + grad_c0*step_size/2)).sum(-1)
                    q1_class = (c0_onehot * (p_c1 + grad_c1*step_size/2)).sum(-1)

                    q0 = q0 * q0_class
                    q1 = q1 * q1_class

                alpha = E0.sum(-1).exp() * q0 / (E1.sum(-1).exp() * q1)
                alpha = torch.where(alpha < 1, alpha, torch.ones_like(alpha))
                acceptance = alpha > torch.rand_like(alpha)

                #import sys
                #print(acceptance.shape, inp_latent.shape, x0.shape, alpha.shape, E0.shape, q0.shape, file=sys.stderr)

                inp_latent = torch.where(acceptance[:, None].expand(-1, inp_latent.shape[1]), inp_latent, x0).detach()
                inp_latent.requires_grad = True
                inp_classes = torch.where(acceptance, inp_classes, c0)

                    


                
            

        # Reactivate gradients for parameters for training
        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_per_step:
            return torch.stack(latents_per_step, dim=0), torch.stack(classes_per_step, dim=0)
        else:
            return inp_latent, current_class
