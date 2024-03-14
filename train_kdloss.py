import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import cycle
from os.path import join as pjoin
from torch.utils.data import Dataset
from vag_net import VAG_Net
import argparse
import segmentation_models_pytorch as smp
from lib.utils import *
from lib.dataset import *
from lib.losses import *
import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)

#2 GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device_ids = [0, 1]

DEVICE = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=500, help='total train epoch')
parser.add_argument('--snapshot', type=int, default=5, help='snapshot ensembles')
parser.add_argument('--miu_sd', type=float, default=0.25, help='coefficient of side-output loss')
parser.add_argument('--miu_kdloss', type=float, default=0.5, help='coefficient of kd loss')
parser.add_argument('--miu_seg', type=float, default=1.0, help='coefficient of segmentation loss')
parser.add_argument('--fold_num', type=int, default=0, help='5-fold valid number')
parser.add_argument('--num_classes', type=int, default=5, help='number of segmentation classes')
parser.add_argument('--batch_size',type=int, default=4, help='batch size')
parser.add_argument('--min_lr', type=float, default=1e-8, help='min learning rate')
parser.add_argument('--lr', type=float, default=1e-4, help='max learning rate')
parser.add_argument('--model_name', type=str, default='VAG-Net-kdloss', help='model name for save')
parser.add_argument('--DATA_4class_T1',type=str, default='./train/MRI_T1', help='path of T1 MRI train image')
parser.add_argument('--DATA_4class_T2',type=str, default='./train/MRI_T2', help='path of T2 MRI train image')
parser.add_argument('--T1_mask',type=str, default='./train/T1_4class_mask', help='path of T1 MRI train mask')
parser.add_argument('--T2_mask',type=str, default='./train/T2_4class_mask', help='path of T2 MRI train mask')
parser.add_argument('--WEIGHT_dir',type=str, default='./weights', help='path to save model weight')

def return_img_ls(img_path):
    '''
    get image dataset list
    :param img_path:
    :return: img_ls
    '''
    data_ls = []
    for sub in os.listdir(img_path):
        name = sub.split('_')
        if name[-1] == 'm.png':
            pass
        else:
            data_ls.append(os.sep.join([img_path,sub]))
    return data_ls

def get_data_id(data_dir,fold_num):
    '''
    :param data_dir:
    :param fold_num:
    :return: train list,valid list
    '''
    data_ls = return_img_ls(data_dir)
    data_num = len(data_ls)
    data_df = pd.DataFrame(columns=('fold','id'))
    data_df['fold'] = (list(range(0, 5)) * data_num)[:data_num]
    data_df['id'] = data_ls
    all_id = data_df['id'].values
    fold = []
    for i in range(5):
        fold.append(data_df.loc[data_df['fold'] == i, 'id'].values)
    train_id = np.setdiff1d(all_id, fold[int(fold_num)])
    val_id = fold[int(fold_num)]
    return train_id,val_id

def get_train_loader_all(train_id,args):
    X_train, _, y_train = ImageFetch(train_id,args.T1_mask,args.T2_mask)
    train_data = SaltDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=RandomSampler(train_data),
                                               drop_last=True)
    return train_loader

def get_dataset(args):
    t1_train_id, t1_val_id = get_data_id(args.DATA_4class_T1, args.fold_num)
    t1_train_loader = get_train_loader_all(t1_train_id,args)
    t1_valid_loader = get_train_loader_all(t1_val_id,args)

    t2_train_id, t2_val_id = get_data_id(args.DATA_4class_T2, args.fold_num)
    t2_train_loader = get_train_loader_all(t2_train_id,args)
    t2_valid_loader = get_train_loader_all(t2_val_id,args)

    return t1_train_loader,t1_valid_loader,t2_train_loader,t2_valid_loader

class Network(nn.Module):
    def __init__(self, n_class):
        super(Network, self).__init__()
        self.s_model = VAG_Net(n_class)
        self.t_model = VAG_Net(n_class)

    def forward(self, s_img, t_img):

        s_logit, s_side8, s_side7, s_side6, s_side5 = self.s_model(s_img)
        t_logit, t_side8, t_side7, t_side6, t_side5 = self.t_model(t_img)
        return s_logit, s_side8, s_side7, s_side6, s_side5, t_logit, t_side8, t_side7, t_side6, t_side5


def Train(s_train_loader, t_train_loader, model,args,prob_return=False):
    '''
    training process
    '''
    model.train()
    running_loss = 0
    source_prob_ls = []
    target_prob_ls = []
    for t2_data, t1_data in zip(cycle(s_train_loader), t_train_loader):
        s_img, s_mask = t2_data[0], t2_data[1]
        t_img, t_mask = t1_data[0], t1_data[1]
        s_img, s_mask, t_img, t_mask = s_img.to('cuda'), s_mask.to('cuda'), t_img.to('cuda'), t_mask.to('cuda')
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # VAG_Net
            s_logit, s_side8, s_side7, s_side6, s_side5, \
            t_logit, t_side8, t_side7, t_side6, t_side5 = model(s_img, t_img)
            s_diceloss,s_celoss= get_segmentation_cost(seg_logits=s_logit, seg_gt=s_mask)
            t_diceloss,t_celoss= get_segmentation_cost(seg_logits=t_logit, seg_gt=t_mask)
            js_loss_logits, source_prob, target_prob = get_js_cost(s_logit, s_mask, t_logit, t_mask, output=True)
            js_loss_side8 = get_js_cost(s_side8, s_mask, t_logit, t_mask)
            js_loss_side7 = get_js_cost(s_side7, s_mask, t_side7, t_mask)
            js_loss_side6 = get_js_cost(s_side6, s_mask, t_side6, t_mask)
            js_loss_side5 = get_js_cost(s_side5, s_mask, t_side5, t_mask)

            seg_dice_side_s, seg_ce_side_s = get_seg_side_loss(s_side8, s_side7, s_side6, s_side5, s_mask)
            seg_dice_side_t, seg_ce_side_t = get_seg_side_loss(t_side8, t_side7, t_side6, t_side5, t_mask)

            source_prob_ls.append(source_prob)
            target_prob_ls.append(target_prob)

            loss = (s_diceloss+s_celoss+t_diceloss+t_celoss)* args.miu_seg + \
                   (js_loss_side8+js_loss_side7+js_loss_side6+js_loss_side5)* args.miu_kdloss* args.miu_sd+ \
                   (seg_dice_side_s+seg_dice_side_t+seg_ce_side_s+seg_ce_side_t)* args.miu_sd+ \
                   js_loss_logits * args.miu_kdloss

            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(s_train_loader)

    if prob_return:
        return epoch_loss, source_prob_ls, target_prob_ls
    else:
        return epoch_loss


def Test(s_val_loader, t_val_loader, model):
    s_predicts = []
    s_truths = []
    t_predicts = []
    t_truths = []
    model.eval()
    for t2_data, t1_data in zip(cycle(s_val_loader), t_val_loader):
        s_img, s_mask = t2_data[0], t2_data[1]
        t_img, t_mask = t1_data[0], t1_data[1]
        s_img, s_mask, t_img, t_mask = s_img.to('cuda'), s_mask.to('cuda'), t_img.to('cuda'), t_mask.to('cuda')
        with torch.no_grad():
            s_logit, _, _, _, _, t_logit, _, _, _, _ = model(s_img, t_img)

            s_logit = nn.Softmax2d()(s_logit)
            t_logit = nn.Softmax2d()(t_logit)

            s_logit = s_logit.detach().cpu().numpy()
            t_logit = t_logit.detach().cpu().numpy()
            s_mask = s_mask.detach().cpu().numpy()
            t_mask = t_mask.detach().cpu().numpy()

            s_predicts.append(s_logit)
            t_predicts.append(t_logit)
            s_truths.append(s_mask)
            t_truths.append(t_mask)

    s_predicts = np.concatenate(s_predicts)
    t_predicts = np.concatenate(t_predicts)
    s_truths = np.concatenate(s_truths)
    t_truths = np.concatenate(t_truths)

    s_predicts = np.where(s_predicts >= 0.5, 1.0, 0.0)
    source_dice_eval_arr = MulticlassDice()(s_predicts, s_truths)

    t_predicts = np.where(t_predicts >= 0.5, 1.0, 0.0)
    target_dice_eval_arr = MulticlassDice()(t_predicts, t_truths)
    mean_dice = (source_dice_eval_arr + target_dice_eval_arr) * 0.5

    return mean_dice

if __name__ == '__main__':

    args = parser.parse_args()
    schedulder_step = args.epoch // args.snapshot

    t1_train_loader, t1_valid_loader, t2_train_loader, t2_valid_loader = get_dataset(args)

    model = Network(args.num_classes)
    model = torch.nn.DataParallel(model.to('cuda'), device_ids=device_ids, output_device=device_ids[0])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr)

    logger = logger(os.sep.join([LOG_dir, 'log_' + args.model_name + '.txt']))
    logger.info('epoch:{},snapshot:{},batch_size:{},weight_name:{}'.format(args.epoch, args.snapshot, args.batch_size, args.model_name))

    max_score = 0.0

    for epoch_ in range(args.epoch):
        train_loss = Train(t2_train_loader, t1_train_loader, model,args)
        val_dice = Test(t2_valid_loader, t1_valid_loader,model)
        scheduler.step()

        if val_dice > max_score:
            max_score = val_dice
            best_param = model.state_dict()

        if (epoch_ + 1) % schedulder_step == 0:
            torch.save(best_param, pjoin(args.WEIGHT_dir, args.model_name + '_' + str(max_score) + '.pth'))
            print('Model saved!')

        logger.info('epoch: {} train_loss: {:.9f} val_dice: {:.9f}'
                    .format(epoch_ + 1, train_loss, val_dice))
        torch.cuda.empty_cache()



