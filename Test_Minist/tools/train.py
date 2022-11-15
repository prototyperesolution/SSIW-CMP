import os, sys
CODE_SPACE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CODE_SPACE)
os.chdir(CODE_SPACE)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import cv2
import logging
import numpy as np
import torch
import torch.nn.functional as F
import json


import utils.config as config
from utils.config import CfgNode
from utils.transforms_utils import get_imagenet_mean_std, normalize_img, pad_to_crop_sz, resize_by_scaled_short_side
import matplotlib.pyplot as plt
from utils.color_seg import color_seg

import glob
from PIL import Image
from utils.labels_dict import UNI_UID2UNAME, ALL_LABEL2ID, UNAME2EM_NAME, CMP_FACADE_NAME
import torch.multiprocessing as mp
from utils.segformer import get_configured_segformer
from tqdm import tqdm
import torch.distributed as dist
from segmentation_models_pytorch.losses import DiceLoss
import segmentation_models_pytorch
from utils.get_class_emb import create_embs_from_names
from test import single_scale_single_crop_cuda, visual_segments, get_prediction
import ast
import PIL

"""
background = (0,0,170)
facade = (0,0,255)
window = (0,85,255)
doors = (0,170,255)
cornice = (0,255,255)
sill = (85,255,170)
balcony = (170,255,85)
blind = (255,255,0)
deco = (255,170,0)
molding = (255,85,0)
pillar = (255,0,0)
shop = (170,0,0)
"""

def get_logger():
    """
    """
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger

logger = get_logger()

def get_parser() -> CfgNode:
    """
    TODO: add to library to avoid replication.
    """
    parser = argparse.ArgumentParser(description='Yvan Yin\'s Semantic Segmentation Model.')
    parser.add_argument('--root_dir', type=str, help='root dir for the data')
    parser.add_argument('--cam_id', type=str, help='camera ID')
    parser.add_argument('--img_folder', default='image_', type=str, help='the images folder name except the camera ID')
    parser.add_argument('--img_file_type', default='jpeg', type=str, help='the file type of images, such as jpeg, png, jpg...')

    parser.add_argument('--config', type=str, default='test_720_ss', help='config file')
    parser.add_argument('--gpus_num', type=int, default=1, help='number of gpus')
    parser.add_argument('--save_folder', type=str, default='ann/semantics', help='the folder for saving semantic masks')

    parser.add_argument('--user_label', nargs='*', help='the label user identified for semantic segmentation')
    parser.add_argument('--new_definitions', type=ast.literal_eval, help='new label definitions identified by user')
    parser.add_argument('opts', help='see mseg_semantic/config/test/default_config_360.yaml for all options, models path should be passed in',
        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    config_path = os.path.join('configs', f'{args.config}.yaml')
    args.config = config_path

    # test on samples
    if args.root_dir is None:
        args.root_dir = f'{CODE_SPACE}/test_imgs'
        args.cam_id='01'
        args.img_file_type = 'png'


    if args.user_label:
        args.user_label = [i for i in args.user_label]

    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    cfg.root_dir = args.root_dir
    cfg.cam_id = args.cam_id
    cfg.img_folder = args.img_folder
    cfg.img_file_type = args.img_file_type
    cfg.gpus_num = args.gpus_num
    cfg.save_folder = args.save_folder

    cfg.user_label = args.user_label
    cfg.new_definitions = args.new_definitions
    return cfg



def cuda_train(model,
                                  image: np.ndarray,
                                  h: int, w: int, gt_embs_list,
                                  args=None) -> np.ndarray:
    ori_h, ori_w, _ = image.shape
    mean, std = get_imagenet_mean_std()
    crop_h = (np.ceil((ori_h - 1) / 32) * 32).astype(np.int32)
    crop_w = (np.ceil((ori_w - 1) / 32) * 32).astype(np.int32)

    image, pad_h_half, pad_w_half = pad_to_crop_sz(image, crop_h, crop_w, mean)
    image_crop = torch.from_numpy(image.transpose((2, 0, 1))).float()
    normalize_img(image_crop, mean, std)
    image_crop = image_crop.unsqueeze(0).cuda()
    model.train()
    """same as single_scale_single_crop_cuda, but with gradients enabled"""
    with torch.set_grad_enabled(True):
        emb, _, _ = model(inputs=image_crop, label_space=['universal'])
    logit = get_prediction(emb, gt_embs_list)
    logit_universal = F.softmax(logit * 100, dim=1).squeeze()

    # disregard predictions from padded portion of image
    prediction_crop = logit_universal[:, pad_h_half:pad_h_half + ori_h, pad_w_half:pad_w_half + ori_w]
    logit = prediction_crop.unsqueeze(0)

    ## CHW -> HWC
    #prediction_crop = prediction_crop.permute(1, 2, 0)
    #prediction_crop = prediction_crop.data.cpu().numpy()

    # upsample or shrink predictions back down to scale=1.0
    #prediction = cv2.resize(prediction_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    return logit


def prepare_CMP_dataset(image_paths, mask_paths):
    """returns images and one hotted masks"""
    images = []
    masks = []
    color_dict = {'background':[0,0,170],'facade':[0,0,255],'window':[0,85,255],'doors':[0,170,255],'cornice':[0,255,255],'sill':[85,255,170],
                  'balcony':[170,255,85],'blind':[255,255,0],'deco':[255,170,0],'molding':[255,85,0],'pillar':[255,0,0],'shop':[170,0,0]}


    for i in range(len(mask_paths)):
        mask = cv2.imread(mask_paths[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        new_mask = np.zeros((mask.shape[0],mask.shape[1],len(color_dict.items())))
        for j in range(len(list(color_dict.values()))):
            layer = np.zeros((new_mask.shape[0],new_mask.shape[1]))
            layer[(mask == np.array(list(color_dict.values())[j])).all(axis = 2)] = 1
            new_mask[:,:,j] = layer

        #new_mask = np.transpose(new_mask, (2,0,1))
        masks.append(new_mask)
        facade_pic = cv2.imread(image_paths[i])
        facade_pic = cv2.cvtColor(facade_pic, cv2.COLOR_BGR2RGB)
        facade_pic = facade_pic
        images.append(facade_pic)

    return images, masks

def zero_shot(TRAIN_TEST_SPLIT = 0.8, dataset_path = 'D:/CMP dataset/base', ckpt_path = 'D:/SSIW-master/SSIW-master/Test_Minist/models/segformer_7data.pth',
              base_size = 128, batch_size = 10, save_dir = 'D:/CMP dataset/zero shot results'):
    print('preparing dataset...')
    # images, masks = prepare_CMP_dataset(dataset_path)
    image_paths = glob.glob(dataset_path + '/*.jpg')
    mask_paths = glob.glob(dataset_path + '/*.png')
    mask_paths = sorted(mask_paths)
    image_paths = sorted(image_paths)

    train_image_paths = image_paths[:int(len(image_paths) * TRAIN_TEST_SPLIT)]
    train_mask_paths = mask_paths[:int(len(mask_paths) * TRAIN_TEST_SPLIT)]

    test_image_paths = image_paths[int(len(image_paths) * TRAIN_TEST_SPLIT):]
    test_mask_paths = mask_paths[int(len(mask_paths) * TRAIN_TEST_SPLIT):]
    print('dataset prepared')

    print('loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_configured_segformer(n_classes=512,
                                     criterion=None,
                                     load_imagenet_model=False)

    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
    ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
    model.load_state_dict(ckpt_filter, strict=False)

    model.to(device)
    print('model loaded')
    return test_model(model, test_image_paths, test_mask_paths, device, save_dir, base_size, batch_size)

def test_model(model, test_image_paths, test_mask_paths, device, save_dir, base_size = 128, batch_size = 10):
    classes = ['background','facade','window','door','cornice','sill','balcony','blind','deco','molding','pillar','shop']
    model.eval()
    idx = 0

    test_iou = 0
    test_f1 = 0
    test_accuracy = 0
    test_recall = 0

    while idx < len(test_image_paths):
        current_batch_size = min(batch_size, len(test_image_paths) - idx)
        batch_image_paths = test_image_paths[idx:idx + current_batch_size]
        batch_mask_paths = test_mask_paths[idx:idx + current_batch_size]

        batch_images, batch_masks = prepare_CMP_dataset(batch_image_paths, batch_mask_paths)
        for j in range(len(batch_images)):
            batch_images[j] = resize_by_scaled_short_side(batch_images[j], base_size, 1)
            batch_masks[j] = resize_by_scaled_short_side(batch_masks[j], base_size, 1)

        """using a new dictionary which I created for this task. Based on what the authors of the paper did, I included a short sentence from wikipedia describing each object within the mask
        I chose to do this instead of using the sentences provided in the dataset's accompanying report, as I believe the Wikipedia sentences contain more pertinent semantic information about each class"""

        gt_embs_list = create_embs_from_names(classes, CMP_FACADE_NAME).float()
        #print(type(gt_embs_list))
        #print(gt_embs_list.shape)

        for i in range(len(batch_images)):
            save_path = os.path.join(save_dir, test_image_paths[idx+i][20:])
            #save_path = os.path.splitext(save_path)[0] + '.png'
            h, w, _ = batch_images[i].shape
            out_logit = single_scale_single_crop_cuda(model, batch_images[i], h, w, gt_embs_list=gt_embs_list, args=None)
            out_logit_transposed = np.transpose(out_logit, (2,0,1))
            out_logit_transposed = torch.tensor(torch.from_numpy(out_logit_transposed[None,...]), device = device, requires_grad = False)
            #print(out_logit.shape)
            """transpose so channels first"""
            gt_mask = np.transpose(batch_masks[i], (2,0,1))
            gt_mask = torch.from_numpy(gt_mask[None,...]).cuda().long()
            tp, fp, fn, tn = segmentation_models_pytorch.metrics.get_stats(out_logit_transposed, gt_mask, mode='multilabel', threshold=0.5)
            iou_score = segmentation_models_pytorch.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            f1_score = segmentation_models_pytorch.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            accuracy = segmentation_models_pytorch.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
            recall = segmentation_models_pytorch.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
            test_iou += iou_score
            test_f1 += f1_score
            test_accuracy += accuracy
            test_recall += recall

            prediction = out_logit.argmax(axis=-1).squeeze()
            probs = out_logit.max(axis=-1).squeeze()  # (h, w)
            high_prob_mask = probs > 0.5

            mask = high_prob_mask
            prediction[~mask] = 255
            pred_color = color_seg(prediction)
            vis_seg = visual_segments(pred_color, batch_images[i])
            #print(os.path.splitext(save_path)[0] + '_vis.png')
            vis_seg = vis_seg.resize((1000,1000))
            vis_seg.save(os.path.splitext(save_path)[0] + '_vis.png')
            cv2.imwrite(save_path, prediction.astype(np.uint8))


        idx += current_batch_size

    test_iou = test_iou/len(test_mask_paths)
    test_f1 = test_f1/len(test_mask_paths)
    test_accuracy = test_accuracy/len(test_mask_paths)
    test_recall = test_recall/len(test_mask_paths)

    return float(test_iou.cpu().numpy()), float(test_f1.cpu().numpy()), float(test_accuracy.cpu().numpy()), float(test_recall.cpu().numpy())


def train(TRAIN_TEST_SPLIT = 0.8, dataset_path = 'D:/CMP dataset/base', ckpt_path = 'D:/SSIW-master/SSIW-master/Test_Minist/models/segformer_7data.pth',
          batch_size = 4, epochs = 30, base_size = 128):
    classes = ['background', 'facade', 'window', 'door', 'cornice', 'sill', 'balcony', 'blind', 'deco', 'molding',
               'pillar', 'shop']
    print('preparing dataset...')
    #images, masks = prepare_CMP_dataset(dataset_path)
    image_paths = glob.glob(dataset_path + '/*.jpg')
    mask_paths = glob.glob(dataset_path+'/*.png')
    mask_paths = sorted(mask_paths)
    image_paths = sorted(image_paths)

    train_image_paths = image_paths[:int(len(image_paths)*TRAIN_TEST_SPLIT)]
    train_mask_paths = mask_paths[:int(len(mask_paths)*TRAIN_TEST_SPLIT)]

    test_image_paths = image_paths[int(len(image_paths) * TRAIN_TEST_SPLIT):]
    test_mask_paths = mask_paths[int(len(mask_paths) * TRAIN_TEST_SPLIT):]
    print('dataset prepared')



    print('loading model...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = get_configured_segformer(n_classes=512,
                                     criterion=None,
                                     load_imagenet_model=False)

    model = torch.nn.DataParallel(model)
    checkpoint = torch.load(ckpt_path, map_location='cpu')['state_dict']
    ckpt_filter = {k: v for k, v in checkpoint.items() if 'criterion.0.criterion.weight' not in k}
    model.load_state_dict(ckpt_filter, strict=False)

    model.to(device)
    print('model loaded')
    """using Adam optimizer with small learning rate, as we are fine tuning the model """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    """using dice loss for multi-class semantic segmentation, https://arxiv.org/pdf/2006.14822.pdf"""
    criterion = DiceLoss(mode = 'multilabel', from_logits = False)
    training_loss = []
    training_iou = []
    test_iou = []
    test_f1 = []
    test_recall = []
    test_accuracy = []

    #model.train()
    """going through dataset once per epoch"""
    for i in range(epochs):
        batches = 0
        epoch_iou = 0
        epoch_loss = 0
        print(len(train_image_paths))
        idx = 0
        while idx < len(train_image_paths):
            current_batch_size = min(batch_size, len(train_image_paths)-idx)
            batch_image_paths = train_image_paths[idx:idx+current_batch_size]
            batch_mask_paths = train_mask_paths[idx:idx+current_batch_size]

            batch_images, batch_masks = prepare_CMP_dataset(batch_image_paths, batch_mask_paths)
            for j in range(len(batch_images)):
                batch_images[j] = resize_by_scaled_short_side(batch_images[j], base_size, 1)
                batch_masks[j] = resize_by_scaled_short_side(batch_masks[j], base_size, 1)

            """using a new dictionary which I created for this task. Based on what the authors of the paper did, I included a short sentence from wikipedia describing each object within the mask
                    I chose to do this instead of using the sentences provided in the dataset's accompanying report, as I believe the Wikipedia sentences contain more pertinent semantic information about each class"""

            gt_embs_list = create_embs_from_names(classes, CMP_FACADE_NAME).float()
            logits =[]
            gt_masks_transposed = []
            batch_loss = 0
            for j in range(len(batch_images)):
                h, w, _ = batch_images[j].shape
                gt_mask = np.transpose(batch_masks[j], (2,0,1))
                gt_mask = torch.from_numpy(gt_mask).cuda().long().unsqueeze(0)
                logit = cuda_train(model, batch_images[j], h, w, gt_embs_list=gt_embs_list, args=None)
                tp, fp, fn, tn = segmentation_models_pytorch.metrics.get_stats(logit, gt_mask, mode='multilabel',threshold=0.5)
                epoch_iou += float(segmentation_models_pytorch.metrics.iou_score(tp, fp, fn, tn, reduction="micro").cpu().numpy())
                #print(epoch_iou)
                loss = criterion(logit.contiguous(), gt_mask)
                batch_loss += loss.item()
                epoch_loss += loss.item()
                loss.backward()

            #print(f'trained on one batch, batch loss = {batch_loss/current_batch_size}')


            idx += current_batch_size
            batches += 1

            optimizer.step()
            optimizer.zero_grad()

        print(batches)
        print(f'epoch {i} loss = {epoch_loss/len(train_image_paths)} iou = {epoch_iou/len(train_image_paths)}')
        training_iou.append(epoch_iou / len(train_image_paths))
        training_loss.append(epoch_loss/len(train_image_paths))
        save_dir_name = 'D:/CMP dataset/epoch'+str(i)
        if os.path.exists(save_dir_name) == False:
            os.mkdir(save_dir_name)
        iou, f1, accuracy, recall = test_model(model, test_image_paths, test_mask_paths, device, save_dir_name)
        test_iou.append(iou)
        test_f1.append(f1)
        test_accuracy.append(accuracy)
        test_recall.append(recall)

        torch.save(model.state_dict(), f'D:/CMP dataset/epoch{i}.pth')

        iou_np = np.expand_dims(np.array(test_iou),0)
        f1_np = np.expand_dims(np.array(test_f1),0)
        accuracy_np = np.expand_dims(np.array(test_accuracy),0)
        recall_np = np.expand_dims(np.array(test_recall),0)
        test_metrics = np.concatenate((iou_np, f1_np, accuracy_np, recall_np), axis = 0)


        train_iou_np = np.expand_dims(np.array(training_iou), 0)
        train_loss_np = np.expand_dims(np.array(training_loss), 0)
        train_metrics = np.concatenate((train_loss_np, train_iou_np),0)

        numpy_directory = 'D:/SSIW-master/numpy_results'
        with open(numpy_directory+'/'+'test_results.npy', 'wb') as f:
            np.save(f, test_metrics)
        with open(numpy_directory+'/'+'train_results.npy', 'wb') as f:
            np.save(f, train_metrics)





    #for layer in model.modules():
    #    print(layer)


def main_worker(local_rank: int, cfg: dict):
    """using this as I don't have multiple GPUs to train with"""
    global_rank = local_rank
    world_size = cfg.gpus_num

    torch.cuda.set_device(global_rank)
    dist.init_process_group( backend="gloo", init_method="tcp://localhost:23456", rank=0, world_size=1 )
    train()


if __name__ == '__main__':

    args = get_parser()
    logger.info(args)

    dist_url = 'tcp://127.0.0.1:6769'
    dist_url = dist_url[:-2] + str(os.getpid() % 100).zfill(2)
    args.dist_url = dist_url

    num_gpus = torch.cuda.device_count()
    if num_gpus != args.gpus_num:
        raise RuntimeError(
            'The set gpus number cannot match the detected gpus number. Please check or set CUDA_VISIBLE_DEVICES')

    if num_gpus > 1:
        args.distributed = True
    else:
        args.distributed = False

    save_path = os.path.join(args.root_dir, args.save_folder, 'id2labels.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(UNI_UID2UNAME, f)

    if not args.distributed:
        print('not distributed')
        main_worker(0, args)
    else:
        mp.spawn(main_worker, nprocs=args.gpus_num, args=(args,))