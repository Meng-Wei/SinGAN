import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize, imresize_to_shape
from torch.utils.tensorboard import SummaryWriter
import time
from SinGAN.manipulate import prune_SinGAN_generate

# Some pruning parameters
structured = False
fine_tune = False

if structured:
    warmup_steps = 100
else:
    warmup_steps = 1000
pruning_amount = 0.5

def train(opt,Gs,Zs,reals,NoiseAmp):
    real_ = functions.read_image(opt)
    in_s = 0
    # cur_scale_level: current level from coarest to finest.
    cur_scale_level = 0
    # scale1: for the largest patch size, what ratio wrt the image shape
    reals = functions.creat_reals_pyramid(real_,reals,opt)
    nfc_prev = 0

    # Train including opt.stop_scale
    while cur_scale_level < opt.stop_scale+1:
        # nfc: number of out channels in conv block
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(cur_scale_level / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(cur_scale_level / 4)), 128)

        # out_: output directory
        # outf: output folder, with scale
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,cur_scale_level)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale.png' %  (opt.outf), functions.convert_image_np(reals[cur_scale_level]), vmin=0, vmax=1)

        D_curr,G_curr = init_models(opt)
        # Notice, as the level increases, the architecture of CNN block might differ. (every 4 levels according to the paper)
        if (nfc_prev==opt.nfc):
            G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,cur_scale_level-1)))
            D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,cur_scale_level-1)))

        # in_s: guess: initial signal? it doesn't change during the training, and is a zero tensor.
        if fine_tune:
          z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt, warmup_steps)
        else:
          z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt, opt.niter)


        G_curr = functions.reset_grads(G_curr,False)
        # D_curr = functions.reset_grads(D_curr,False)
        G_curr.eval()
        # D_curr.eval()

        #################################################################################
        # Visualzie weights
        def visualize_weights(modules, fig_name):
            ori_weights = torch.tensor([]).cuda()
            for m in modules:
                cur_params = m.weight.data.flatten()
                ori_weights = torch.cat((ori_weights, cur_params))
                # cur_params = m.bias.data.flatten()
                # ori_weights = torch.cat((ori_weights, cur_params))
            sparsity = torch.sum(ori_weights == 0) * 1.0 / (ori_weights.nelement())
            print(sparsity, ori_weights.nelement())
            ori_weights = ori_weights.cpu().numpy()
            ori_weights = plt.hist(ori_weights[ori_weights != 0], bins=100)
            plt.savefig("%s/%s.png" % (opt.outf, fig_name))
            plt.close()

        # Pruning weights Structured or Non-structured
        if not structured:
            modules = [G_curr.head.conv, G_curr.head.norm,
                    G_curr.body.block1.conv, G_curr.body.block1.norm,
                    G_curr.body.block2.conv, G_curr.body.block2.norm,
                    G_curr.body.block3.conv, G_curr.body.block3.norm,
                    G_curr.tail[0]]
            parameters_to_prune = (
                (G_curr.head.conv, 'weight'),
                (G_curr.head.norm, 'weight'),
                (G_curr.body.block1.conv, 'weight'),
                (G_curr.body.block1.norm, 'weight'),
                (G_curr.body.block2.conv, 'weight'),
                (G_curr.body.block2.norm, 'weight'),
                (G_curr.body.block3.conv, 'weight'),
                (G_curr.body.block3.norm, 'weight'),
                (G_curr.tail[0], 'weight'),
                (G_curr.head.conv, 'bias'),
                (G_curr.head.norm, 'bias'),
                (G_curr.body.block1.conv, 'bias'),
                (G_curr.body.block1.norm, 'bias'),
                (G_curr.body.block2.conv, 'bias'),
                (G_curr.body.block2.norm, 'bias'),
                (G_curr.body.block3.conv, 'bias'),
                (G_curr.body.block3.norm, 'bias'),
                (G_curr.tail[0], 'bias'),
            )

            visualize_weights(modules, 'ori')

            # Prune weights
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_amount,
            )
        else:
            modules = [G_curr.head.conv,
            G_curr.body.block1.conv,
            G_curr.body.block2.conv,
            G_curr.body.block3.conv]

            visualize_weights(modules, 'ori')
            # pytorch_total_params = sum(p.numel() for p in G_curr.parameters())
            # print(pytorch_total_params)

            for module in modules:
                m = prune.ln_structured(module, name="weight", amount=pruning_amount, n=1, dim=0)
                # m = prune.ln_structured(module, name="bias", amount=pruning_amount, n=1, dim=0)

        torch.save(G_curr.state_dict(), '%s/raw_prune_netG.pth' % (opt.outf))
        visualize_weights(modules, 'raw-prune')
        if cur_scale_level > 0:
            fake_Gs = Gs.copy()
            fake_Gs.append(G_curr) 
            fake_Zs = Zs.copy()
            fake_Zs.append(z_curr)
            fake_noise = NoiseAmp.copy()
            fake_noise.append(opt.noise_amp)
            fake_reals = reals[:cur_scale_level+1].copy()
            prune_SinGAN_generate(fake_Gs, fake_Zs, fake_reals, fake_noise, opt, gen_start_scale=0, num_samples=1, level=cur_scale_level)

        # Fine-tuning
        if fine_tune:
            G_curr = functions.reset_grads(G_curr, True)
            G_curr.train()

            if not structured:
                # Keep training using inherited weights
                z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt, opt.niter - warmup_steps, prune=True)
            else:
                # Training from scratch
                # G_curr.apply(models.weights_init)
                # D_curr.apply(models.weights_init)
                z_curr, in_s, G_curr = train_single_scale(D_curr, G_curr, reals, Gs, Zs, in_s, NoiseAmp, opt, opt.niter, prune=True)
        G_curr = functions.reset_grads(G_curr,False)
        G_curr.eval()
        visualize_weights(modules, 'fine-tune')

        for m in modules:
            prune.remove(m, 'weight')
            if not structured:
              prune.remove(m, 'bias')
        
        # pytorch_total_params = sum(p.numel() for p in G_curr.parameters())
        # print(pytorch_total_params)

        #################################################################################
        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/pruned_Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        cur_scale_level+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return



def train_single_scale(netD,netG,reals,Gs,Zs,in_s,NoiseAmp,opt,steps,centers=None, prune=False):

    real = reals[len(Gs)]
    opt.nzx = real.shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real.shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    # generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1)
    # size: [opt.nc_z, opt.nzx, opt.nzy]
    # z_opt: input noise for calculating reconstruction loss in Generator
    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD,milestones=[1600],gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG,milestones=[1600],gamma=opt.gamma)

    # TODO: these plots are not visualized
    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(steps):
        start_time = time.time()
        if (Gs == []) & (opt.mode != 'SR_train'):
            # Bottom generator, here without zero init
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
            # noise_: input noise for the discriminator
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD.zero_grad()

            output = netD(real).to(opt.device)
            #D_real_map = output.detach()
            errD_real = -output.mean()#-a
            errD_real.backward(retain_graph=True)
            D_x = -errD_real.item()

            # train with fake
            if (j==0) & (epoch == 0):
                # Initialize prev and z_prev
                # prev: image outputs from previous level
                # z_prev: image outputs from previous level of fixed noise z_opt
                if (Gs == []) & (opt.mode != 'SR_train'):
                    # in_s and prev are both noise
                    # z_prev are also noise
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1
                elif opt.mode == 'SR_train':
                    z_prev = in_s
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init * RMSE
                    z_prev = m_image(z_prev)
                    prev = z_prev
                else:
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    prev = m_image(prev)
                    z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                    criterion = nn.MSELoss()
                    RMSE = torch.sqrt(criterion(real, z_prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                prev = m_image(prev)

            if opt.mode == 'paint_train':
                prev = functions.quant2centers(prev,centers)
                plt.imsave('%s/prev.png' % (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_+prev

            fake = netG(noise.detach(),prev)
            output = netD(fake.detach())
            errD_fake = output.mean()
            errD_fake.backward(retain_graph=True)
            D_G_z = output.mean().item()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            errD = errD_real + errD_fake + gradient_penalty
            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG.zero_grad()
            output = netD(fake)
            #D_fake_map = output.detach()
            errG = -output.mean()
            errG.backward(retain_graph=True)
            if alpha!=0:
                loss = nn.MSELoss()
                if opt.mode == 'paint_train':
                    z_prev = functions.quant2centers(z_prev, centers)
                    plt.imsave('%s/z_prev.png' % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)
                Z_opt = opt.noise_amp*z_opt+z_prev
                rec_loss = alpha*loss(netG(Z_opt.detach(),z_prev),real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        D_fake2plot.append(D_G_z)
        z_opt2plot.append(rec_loss)

        if epoch > 1 and (epoch % 100 == 0 or epoch == (steps-1)):
            total_time = time.time() - start_time
            start_time = time.time()
            print('scale %d:[%d/%d], total time: %f' % (len(Gs), epoch, opt.niter, total_time))
            memory = torch.cuda.max_memory_allocated()
            # print('allocated memory: %dG %dM %dk %d' % 
            #         ( memory // (1024*1024*1024), 
            #           (memory // (1024*1024)) % 1024,
            #           (memory // 1024) % 1024,
            #           memory % 1024 ))
            print('allocated memory: %.03f GB' % (memory / (1024*1024*1024*1.0) ))

        if epoch > 1 and (epoch % 500 == 0 or epoch == (steps-1)):
            if not prune:
                plt.imsave('%s/fake_sample%d.png' %  (opt.outf, steps), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
                plt.imsave('%s/G(z_opt)%d.png'    % (opt.outf, steps),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
                plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
                plt.plot(np.arange(epoch+1), errG2plot, label='errG')
                plt.plot(np.arange(epoch+1), D_real2plot, label='D_real')
                plt.plot(np.arange(epoch+1), D_fake2plot, label='D_fake')
                plt.legend(loc="best")
                plt.savefig('%s/error.png' % (opt.outf))
                plt.close()
            else:
                plt.imsave('%s/prune_fake_sample%d.png' %  (opt.outf, steps), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
                plt.imsave('%s/prune_G(z_opt)%d.png'    % (opt.outf, steps),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
                plt.imsave('%s/prune_prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
                plt.plot(np.arange(epoch+1), errG2plot, label='errG')
                plt.plot(np.arange(epoch+1), D_real2plot, label='D_real')
                plt.plot(np.arange(epoch+1), D_fake2plot, label='D_fake')
                plt.legend(loc="best")
                plt.savefig('%s/prune_error.png' % (opt.outf))
                plt.close()

            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD.step()
        schedulerG.step()
    if not prune:
        functions.save_networks(netG,netD,z_opt,opt)
    else:
        torch.save(netG.state_dict(), '%s/fine_tune_netG.pth' % (opt.outf))
    return z_opt,in_s,netG    

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            if opt.mode == 'animation_train':
                pad_noise = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                # z is a random noise.
                # z_in is a 
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals,reals[1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                G_z = G(z_in.detach(),G_z)
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    # print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    # print(netD)

    return netD, netG
