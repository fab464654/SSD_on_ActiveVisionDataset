from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = "google_drive/MyDrive/checkpointsIeri/checkpoint_ssd300.pth.tar"
data_folder = 'google_drive/MyDrive/ColabNotebooks/Project/_provaSSD/test'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
"""
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)
"""

#With AVD dataset:

#import active_vision_dataset_processing.data_loading
import transforms, active_vision_dataset

#Include all instances
pick_trans = transforms.PickInstances(range(34))

TEST_PATH = "./google_drive/MyDrive/ColabNotebooks/Project/testDataset"
"""
 scene_list=['Home_001_2',                                                    
                                                'Home_003_2',                                                   
                                                'Home_004_2',
                                                'Home_005_2',
                                                'Home_013_1',
                                                'Home_014_2',
                                                'Home_015_1',
                                                'Home_016_1']
"""
test_dataset = active_vision_dataset.AVD(root=TEST_PATH,
                                    target_transform=pick_trans,
                                    scene_list=['Home_001_2'],
                                      fraction_of_no_box=-1)
  
test_loader = torch.utils.data.DataLoader(test_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=active_vision_dataset.collate
                          )

def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        #for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
        for i, (images, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):          
          #COPIED CODE  FROM TRAIN.PY    
          #Pre-processing:        
          from torchvision import transforms as transf
          preprocess = transf.Compose([
                        transf.ToPILImage(),
                        transf.Resize(300),
                        transf.CenterCrop(300),
                        transf.ToTensor(),                      
                        transf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
          
          #numImagesLastBatch = len(test_dataset) - batch_size*(18-1)  
          #print(numImagesLastBatch)
          numImagesLastBatch = -100
          #print("len:", len(test_loader))
          
          if i == len(test_loader)-1:
              numImagesLastBatch = len(test_dataset) - batch_size*(i)  
              print("we_ ", numImagesLastBatch)


          for j in range(batch_size):             
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
              #print("shape images: ",input_batch.shape) 

            # In the Active Vision Dataset we have this formatting:
            # [xmin ymin xmax ymax instance_id difficulty]

            box_id_diff = [b for b in labels[j][0]]  
          
            box = [l[0:4] for l in box_id_diff]

            #Boundary coordinates as requested
            for k in range(len(box)):  
              box[k][0] = box[k][0]/1920.0
              box[k][2] = box[k][2]/1920.0          
              box[k][1] = box[k][1]/1080.0
              box[k][3] = box[k][3]/1080.0 
            
            box_tensor = torch.FloatTensor(box).to(device)

            if j == 0:            
              box_list = [box_tensor]
            else:
              box_list.append(box_tensor)               

            label = [l[4] for l in box_id_diff]
            label_tensor = torch.LongTensor(label).to(device)
            if j == 0: 
              label_list = [label_tensor]
            else:
              label_list.append(label_tensor)        
      
            #According to the code, difficulty is used to compute the mAP and should be
            #zero (not diff.) or one (diff), while in AVD Dataset it's a number from 0 to 5            
            difficulty = [l[5] for l in box_id_diff]
            for k in range(len(difficulty)):
              if difficulty[k] >= 5:
                difficulty[k] = 1
              else:
                difficulty[k] = 0

            difficulty_tensor = torch.LongTensor(difficulty).to(device)
            if j == 0: 
              difficulty_list = [difficulty_tensor]
            else:
              difficulty_list.append(difficulty_tensor)      

            if j == numImagesLastBatch-1:
              j = batch_size
              break
          
            
          #endFor j
          images = input_batch.to(device)  # (batch_size (N), 3, 300, 300)
        
          boxes = box_list
          labels = label_list
          difficulties = difficulty_list


          # Forward prop.
          predicted_locs, predicted_scores = model(images)

          # Detect objects in SSD output
          det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                      min_score=0.01, max_overlap=0.45,
                                                                                      top_k=200)

          #print(det_scores_batch)
          # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

          # Store this batch's results for mAP calculation
          #boxes = [b.to(device) for b in boxes]
          #labels = [l.to(device) for l in labels]
          difficulties = [d.to(device) for d in difficulties]
          
          #print(difficulties) #Just to check

          det_boxes.extend(det_boxes_batch)
          det_labels.extend(det_labels_batch)
          det_scores.extend(det_scores_batch)
          true_boxes.extend(boxes)
          true_labels.extend(labels)
          true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    # Print AP for each class
    pp.pprint(APs)

    print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model)
