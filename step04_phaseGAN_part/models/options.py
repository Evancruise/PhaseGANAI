import argparse
from datetime import datetime
import os
class ParamOptions():
    """This class defines options used during both training and test time."""
    def __init__(self):
        self.initialized = False
        self.time = datetime.now()
        self.cwd = os.getcwd()

    def initialize(self,parser):
        parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                            help='unpaired phase reconstruction using propagation-enhanced cycle-consistent adversarial network')
        parser.add_argument('--load_path', type=str, default = './dataset/IMGS_E_40_SKE_BKSc', help='path to training h5 files (should have a subfolder named test') # Change training path here
        parser.add_argument('--run_path', type=str, default = './results/fig', help='path to save results')
        parser.add_argument('--run_name', type=str, default = self.time.strftime('%b%d_%H_%M'), help='folder name of this run') #TODO: modify save_path and run_name
        parser.add_argument('--run_load_name', type=str, default = self.time.strftime('%b%d_%H_%M'), help='folder name of this run') #TODO: modify save_path and run_name
        parser.add_argument('--load_weights', type=str, default = 'False', help='folder name of this run') #TODO: modify save_path and run_name
        parser.add_argument('--model_A', type=str, default = 'UNet', help='detector-plane generator model')
        parser.add_argument('--model_B', type=str, default = 'UNet', help='detector-plane generator model')
        parser.add_argument('--batch_size', '-b', type=int, default=16, help='input batch size')
        parser.add_argument('--lambda_GA', type=float, default=0.1, help='weight for adversarial loss of generator A')
        parser.add_argument('--lambda_GB', type=float, default=0.1, help='weight for adversarial loss of generator B')
        parser.add_argument('--lambda_FSCA', type=float, default=1.0, help='weight for Fourier ring correlation loss A')
        parser.add_argument('--lambda_FSCB', type=float, default=1.0, help='weight for Fourier ring correlation loss B')
        parser.add_argument('--lambda_A', type=float, default=0.2, help='weight for cycle consistency loss between real_A and rec_A')
        parser.add_argument('--lambda_B', type=float, default=0.3, help='weight for cycle consistency loss between real_B and rec_B')
        parser.add_argument('--no_pretrain', action='store_true', help='no pretrain for the generator, by default it is pretrained with vgg11')
        parser.add_argument('--isTest', action='store_true', help='not train the model')
        parser.add_argument('--lr_g', type=float, default=0.002, help='initial learning rate for the generator')
        parser.add_argument('--lr_d', type=float, default=0.001, help='initial learning rate for the discriminator')
        parser.add_argument('--num_epochs','-n', type=int, default=250, help='total number of epochs')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--clip_max', type=float, default=1.0, help='maximum value for the gradient clipping, set to 0 if do not want to use gradient clipping.')
        parser.add_argument('--image_stats', type=list,default=[0,1,0,1,0,1],
                            help='statistics of training images written as [real_A_mean, real_A_std, real_B_ch1_mean, real_B_ch1_std, real_B_ch2_mean, real_B_ch2_std]')
        parser.add_argument('--energy', type = float, default=12.4, help='X-ray photon energy in keV')
        parser.add_argument('--pxs', type=float, default=1e-6, help='pixel size')
        parser.add_argument('--z', type=float, default=[0.009, 0.078], help='propagation distance')
        # [0.009, 0.078, 0.132, 0.145, 0.252, 0.267]
        parser.add_argument('--adjust_lr_epoch', type=int, default=30, help='set the learning rate to the initial learning rate decayed by 10 every certain epochs')
        parser.add_argument('--log_note', type=str, default=' ', help='run note which will be saved to the log file')
        parser.add_argument('--save_model_freq_epoch', type=int, default=1, help='frequency of saving models (epoch)')
        parser.add_argument('--print_loss_freq_iter', type=int, default=2, help='frequency of print loss (iteration)')
        parser.add_argument('--choice', type=int, default=1, help='loss function type')
        parser.add_argument('--save_cycleplot_freq_iter', type=int, default=1, help='freqency of save cycle plots for train images')
        parser.add_argument('--save_val_freq_epoch',type=int, default=1, help='frequency of save cycle plots for validation images')
        parser.add_argument('--savefig',type=int, default=1, help='Sve the figures or not (0 for no, 1 for yes)')
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized: 
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,add_help=False)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        opt = self.gather_options()
        self.opt = opt
        return self.opt
