import torch
import resource
import numpy as np
from models.options import ParamOptions
from models.trainer import TrainModel
from tensorboardX import SummaryWriter

if __name__ == '__main__':
    opt = ParamOptions().parse()
    model = TrainModel(opt)
    step, step_test, save_name = model.init_model(opt.model_A, opt.model_B)
    writer = SummaryWriter()
    init_epoch=0
    
    init_epoch = str(input('input the initial epoch:'))
    init_time = str(input('input the corresponding date (ex: Jun04_19_31_{}_{}, {}: UNet, WNet, SRResNet):'))
    save_name = init_time
    path = 'pretrained_model/' + init_time + '/save/' # path = 'dataset/IMGS_1_10e-6_E_40_SKE_BKSc/' + init_time + '/save/'
    model.load_models(model, str(init_epoch), str(path), str(init_time) + '_' + opt.model_A + '_' + opt.model_B + '_netG_A_' + str(init_epoch) + 'ep.pt', str(init_time) + '_' + opt.model_A + '_' + opt.model_B + '_netD_A_' + str(init_epoch) + 'ep.pt', str(init_time) + '_' + opt.model_A + '_' + opt.model_B + '_netG_B_' + str(init_epoch) + 'ep.pt', str(init_time) + '_' + opt.model_A + '_' + opt.model_B +'_netD_B_' + str(init_epoch) + 'ep.pt')
    
    print('******************************* Test Part *******************************')
    
    with torch.no_grad():
        for k,test_data in enumerate(model.test_loader):
            model.set_input(test_data)
                    
            model.optimization('test')
            
            model.visual_val(epoch,k, opt.model_A, opt.model_B)
                    
            val_losses = model.get_current_losses('test')
            
            model.print_current_losses(epoch,k,val_losses)
