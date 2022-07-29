from models.loss import *
from models.common import set_requires_grad


class updater(object):
    def __init__(self):
        self.advrs_loss = 'adversarial loss you choose'
    
    def update_g(self):
        pass
    
    def update_d(self):
        pass
    
    def update(self):
        pass

    
class GAN(updater):
    def __init__(self):
        self.advrs_loss = Adversarial_loss(criterion = nn.BCEWithLogitsLoss())

    def update_d(self, discriminator, optimizer, generator, cond_img, real_img):
        set_requires_grad(discriminator, True)
        optimizer.zero_grad()
        real_prob = discriminator(real_img, cond_img)
        real_loss = self.advrs_loss(real_prob, True)
        fake_img = generator(cond_img).detach()
        fake_prob = discriminator(fake_img, cond_img.detach())
        fake_loss = self.advrs_loss(fake_prob, False)
        dis_loss = (real_loss + fake_loss) * 0.5
        dis_loss.backward()
        optimizer.step()
        
    def update_g(self, discriminator, generator, optimizer, cond_img, real_img, addit_loss, add_loss_weight):
        set_requires_grad(discriminator, False)
        optimizer.zero_grad()
        fake_img = generator(cond_img)
        fake_prob = discriminator(fake_img, cond_img)
        fake_loss = self.advrs_loss(fake_prob, True)
        add_loss = add_loss_weight*addit_loss(real_img, fake_img)
        gen_loss = fake_loss + add_loss
        gen_loss.backward()
        optimizer.step()

    def update(self, discriminator, dis_optimizer, generator, gen_optimizer, addit_loss, add_loss_weight, data_loader):
        for cond_img, real_img in data_loader:
            self.update_d(
                discriminator = discriminator,
                optimizer = dis_optimizer,
                generator = generator,
                cond_img = cond_img.to(device),
                real_img = real_img.to(device)
            )
            
        for cond_img, real_img in data_loader:
            self.update_g(
                discriminator = discriminator,
                generator = generator,
                optimizer = gen_optimizer,
                cond_img = cond_img.to(device),
                real_img = real_img.to(device),
                addit_loss = addit_loss,
                add_loss_weight = add_loss_weight
            )

        
class WGAN(GAN):
    def __init__(self):
        self.advrs_loss = Wasserstein_loss()
    
    def get_gradient_penalty(self, discriminator, real_img, fake_img, cond_img):
        # Calculate interpolation
        num_batch = real_img.shape[0]
        alpha = torch.rand(num_batch, 1, 1, 1).expand_as(real_img).to(device)
        interpolated = alpha * real_img + (1 - alpha) * fake_img
        interpolated = interpolated.to(device)
        interpolated.requires_grad = True

        # Calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated, cond_img)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                            create_graph=True, retain_graph=True)[0]

        gradients = gradients.view(num_batch, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return 10 * ((gradients_norm - 1) ** 2).mean()
    
    def update_d(self, discriminator, optimizer, generator, cond_img, real_img):
        set_requires_grad(discriminator, True)
        optimizer.zero_grad()
        real_prob = discriminator(real_img, cond_img)
        real_loss = self.advrs_loss(real_prob, True)
        fake_img = generator(cond_img).detach()
        fake_prob = discriminator(fake_img, cond_img.detach())
        fake_loss = self.advrs_loss(fake_prob, False)
        dis_loss = fake_loss + real_loss + self.get_gradient_penalty(discriminator, real_img, fake_img, cond_img)
        dis_loss.backward()
        optimizer.step()
    
    def update(self, discriminator, dis_optimizer, generator, gen_optimizer, addit_loss, add_loss_weight, data_loader):
        for _ in range(5):
            for cond_img, real_img in data_loader:
                self.update_d(
                    discriminator = discriminator,
                    optimizer = dis_optimizer,
                    generator = generator,
                    cond_img = cond_img.to(device),
                    real_img = real_img.to(device)
                )
            
        for cond_img, real_img in data_loader:
            self.update_g(
                discriminator = discriminator,
                generator = generator,
                optimizer = gen_optimizer,
                cond_img = cond_img.to(device),
                real_img = real_img.to(device),
                addit_loss = addit_loss,
                add_loss_weight = add_loss_weight
            )

    
class LSGAN(GAN):
    def __init__(self):
        self.advrs_loss = Adversarial_loss(criterion = nn.MSELoss())
           
    def update_g(self, discriminator, generator, optimizer, cond_img, real_img, addit_loss, add_loss_weight):
        set_requires_grad(discriminator, False)
        optimizer.zero_grad()
        fake_img = generator(cond_img)
        fake_prob = discriminator(fake_img, cond_img)
        fake_loss = self.advrs_loss(fake_prob, True)
        add_loss = add_loss_weight*addit_loss(real_img, fake_img)
        gen_loss = 0.5*fake_loss + add_loss
        gen_loss.backward()
        optimizer.step()
    
    
    