from config import get_arguments
from SinGAN.manipulate import *
import SinGAN.functions as functions

if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_dir', help='input image dir', default='Input/Images')
    parser.add_argument('--input_name', help='input image name', required=True)
    #==========================
    parser.add_argument('--no_recons', action='store_true', 
                        help='training the model to fit the Laplacian Pyramid', default=False)
    #==========================
    # Modes including:
    # train
    # random_samples
    # random_samples_arbitrary_sizes
    # harmonization
    # editing
    # SR_train: super resolution train
    # SR: super resolution
    # paint_train
    # paint2image
    # animation_train
    # animation
    parser.add_argument('--mode', help='task to be done', default='train')
    opt = parser.parse_args()

    if opt.no_recons:
        from SinGAN.no_training import *
    else:
        from SinGAN.training import *

    opt = functions.post_config(opt)
    Gs = []
    Zs = []
    reals = []
    NoiseAmp = []
    dir2save = functions.generate_dir2save(opt)

    if (os.path.exists(dir2save)):
        print('trained model already exist')
    else:
        try:
            os.makedirs(dir2save)
        except OSError:
            pass
    real = functions.read_image(opt)
    functions.adjust_scales2image(real, opt)
    train(opt, Gs, Zs, reals, NoiseAmp)

    # # Generating images
    # Gs, Zs, reals, NoiseAmp = functions.load_trained_pyramid(opt)
    # torch.cuda.reset_max_memory_allocated()
    # memory = torch.cuda.max_memory_allocated()
    # # TODO: Find generating start scale using the real images or the z_opt
    # SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=0)
    # SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=1)
    # SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=2)
    # SinGAN_generate(Gs, Zs, reals, NoiseAmp, opt, gen_start_scale=3)
    # print('allocated memory: %.03f GB' % (memory / (1024*1024*1024*1.0) ))