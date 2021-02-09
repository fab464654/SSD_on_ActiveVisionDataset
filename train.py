import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *

# Data parameters
data_folder = 'google_drive/MyDrive/ColabNotebooks/Project/trainDataset' # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 30  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    #import active_vision_dataset_processing.data_loading
    import transforms, active_vision_dataset

    #only inlcude labels for instances with id in range(5)
    pick_trans = transforms.PickInstances(range(5), max_difficulty=5)

    TRAIN_PATH = "./google_drive/MyDrive/ColabNotebooks/Project/trainDataset"
    train_dataset = active_vision_dataset.AVD(root=TRAIN_PATH, train=True,
                                        target_transform=pick_trans,
                                        scene_list=['Home_005_1', 'Home_006_1']
                                        )

    train_loader = torch.utils.data.DataLoader(train_dataset,
                              batch_size=12,
                              shuffle=True,
                              collate_fn=active_vision_dataset.collate
                              )
    """
    # Custom dataloaders
    train_dataset = PascalVOCDataset(data_folder,
                                     split='train',
                                     keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here
    """
    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // 32)
    decay_lr_at = [it // (len(train_dataset) // 32) for it in decay_lr_at]

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    CATEGORY_NAMES = ['background','advil_liqui_gels','aunt_jemima_original_syrup',
    'bumblebee_albacore','cholula_chipotle_hot_sauce','coca_cola_glass_bottle',
    'crest_complete_minty_fresh','crystal_hot_sauce','expo_marker_red','hersheys_bar',
    'honey_bunches_of_oats_honey_roasted','honey_bunches_of_oats_with_almonds','hunts_sauce',
    'listerine_green','mahatma_rice','nature_valley_granola_thins_dark_chocolate',
    'nutrigrain_harvest_blueberry_bliss','pepto_bismol','pringles_bbq',
    'progresso_new_england_clam_chowder','quaker_chewy_low_fat_chocolate_chunk',
    'red_bull','softsoap_clear','softsoap_gold','softsoap_white','spongebob_squarepants_fruit_snaks',
    'tapatio_hot_sauce','vo5_tea_therapy_healthful_green_tea_smoothing_shampoo',
    'nature_valley_sweet_and_salty_nut_almond','nature_valley_sweet_and_salty_nut_cashew',
    'nature_valley_sweet_and_salty_nut_peanut','nature_valley_sweet_and_salty_nut_roasted_mix_nut',
    'paper_plate','red_cup']
    import numpy as np
    # Batches
    for i, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - start)
        #print(len(images))
        #print(labels)
        # Move to default device
        data = images
        a = np.asarray(data)
        print(a.shape)
        #a = np.squeeze(a, axis=1) # shape should now be (L, 224, 224, 3)
        

        #image = torch.from_numpy(a) 
        #image = image.permute(0,3,1,2)
        #print(image.shape)

        #Pre-processing:        
        from torchvision import transforms as transf
        preprocess = transf.Compose([
                      transf.ToPILImage(),
                      transf.Resize(300),
                      transf.CenterCrop(300),
                      transf.ToTensor(),
                      transf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                  ])
                  
        #print(input_batch)
        for j in range(12):   
          if j == 0:   
            input_tensor = preprocess(images[j])
            input_tensor = input_tensor.unsqueeze(0)
            input_batch = input_tensor
          else:
            input_tensor = preprocess(images[j])
            #print(input_tensor)
            input_tensor = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
            #print(input_tensor.shape)
            input_batch = torch.cat((input_batch, input_tensor), 0)
            #print(input_batch.shape) 

          print(j)
          print(labels[j][0])         
          print(labels[j][0][4])
         
          """
          boxes = [b.to(device) for b in torch.tensor(labels[j][0])]

          if not labels[j][0]:
            labels = []        
            print("coasc")  
          else:              
            labels = [l.to(device) for l in torch.tensor(labels[j][0][4])]
          """
        images = input_batch.to(device)  # (batch_size (N), 3, 300, 300)
        #print(images.shape)
     
        """print(boxes)
        print(labels)"""
        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
