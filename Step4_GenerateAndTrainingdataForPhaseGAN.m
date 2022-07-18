command = 'python write_h5_files.py';
status = system(command);

cd step04_phaseGAN_part
command = 'python3 train.py --load_path ./../case1/IMGS_PCI -b 10 --model_A UNet --model_B UNet';
status = system(command);