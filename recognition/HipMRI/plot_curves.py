import json

import matplotlib.pyplot as plt

is_3d = True

if is_3d:
    with open('3D_Model/unet3d_plot_data.json') as file:
        data = json.load(file)
    epochs = data['epoch']

    # loss vs epoch
    plt.figure()
    plt.plot(epochs, data['train_loss'], label='Train loss')
    plt.plot(epochs, data['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('3D_Model/unet3d_loss.png')

    # dice vs epoch
    plt.figure()
    for name, values in data['val_dice'].items():
        plt.plot(epochs, values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation DSC')
    plt.legend()
    plt.savefig('3D_Model/unet3d_dice.png')
else:
    with open('2D_Unet_output/unet2d_plot_data.json') as file:
        data = json.load(file)
    epochs = data['epoch']

    # loss vs epoch
    plt.figure()
    plt.plot(epochs, data['train_loss'], label='Train loss')
    plt.plot(epochs, data['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('2D_Unet_output/unet2d_loss.png')

    # dice vs epoch
    plt.figure()
    for name, values in data['val_dsc'].items(): # the 2D script saves this as val_dsc instead of val_dice
        plt.plot(epochs, values, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Validation DSC')
    plt.legend()
    plt.savefig('2D_Unet_output/unet2d_dice.png')

plt.show()
