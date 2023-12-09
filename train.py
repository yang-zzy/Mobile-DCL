import argparse

import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from DCL_model import DCL
from dataset import *
from utils import *


class DCLTrainer(object):

    def __init__(self, config: dict):
        print('Start training prepare...')
        self.config = config
        self.epoch = 0
        self.start_epoch = 0
        self.total_epoch = self.config['epoch']
        self.device = torch.device('cuda' if self.config['use_cuda'] and torch.cuda.is_available() else 'cpu')
        set_random_seed(self.config['seed'])
        print(f"Using specific random seed: {self.config['seed']}")
        self.num_classes = self.config['num_classes']
        self.transformers = self.get_transformers()
        self.collate_fn = self.get_collate_fn()
        self.datasets = self.get_dataset()
        self.dataloader = self.get_dataloader()
        print("Building DCL Model...")
        self.model = self.get_model().to(self.device)
        print("Building DCL Model OK!")
        self.criterion = self.get_criterion()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()

        # build meters
        self.performance_meters = self.get_performance_meters()
        self.average_meters = self.get_average_meters()

        # timer
        self.timer = Timer()
        print('Training Preparation Done!')

    def get_performance_meters(self):
        return {
            'train': {
                metric: PerformanceMeter(higher_is_better=False if 'loss' in metric else True)
                for metric in ['acc', 'loss']
            },
            'val': {
                metric: PerformanceMeter() for metric in ['acc']
            },
            'val_first': {
                metric: PerformanceMeter() for metric in ['acc']
            }
        }

    def get_average_meters(self):
        meters = ['acc', 'loss']  # Reset every epoch. 'acc' is reused in train/val/val_first stage.
        return {
            meter: AverageMeter() for meter in meters
        }

    def reset_average_meters(self):
        for meter in self.average_meters:
            self.average_meters[meter].reset()

    def get_model(self):
        model = DCL(self.num_classes, self.config['cls_2'], self.config['cls_2xmul'])
        if self.config['pretrained']:
            state_dict = torch.load(self.config['model'], map_location='cpu')
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if
                               k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        return model

    def get_transformers(self):
        resize_reso = 256
        crop_reso = 224
        swap_num = [3, 3]

        return {
            'swap': transforms.Compose([
                RandomSwap((swap_num[0], swap_num[1])),
            ]),
            'common_aug': transforms.Compose([
                transforms.Resize((resize_reso, resize_reso)),
                transforms.RandomRotation(degrees=15),
                transforms.RandomCrop((crop_reso, crop_reso)),
                transforms.RandomHorizontalFlip(),
            ]),
            'train_totensor': transforms.Compose([
                transforms.Resize((crop_reso, crop_reso)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val_totensor': transforms.Compose([
                transforms.Resize((crop_reso, crop_reso)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'test_totensor': transforms.Compose([
                transforms.Resize((resize_reso, resize_reso)),
                transforms.CenterCrop((crop_reso, crop_reso)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'None': None,
        }

    def get_collate_fn(self):
        return {
            'train': collate_fn4train,
            'val': collate_fn4val
        }

    def get_dataset(self):
        splits = ['train', 'val']
        meta_paths = {
            split: os.path.join(self.config['meta_dir'], split + '.txt') for split in splits
        }
        return {
            split: DCLDataset(
                self.config['data_dir'], meta_paths[split], transforms=self.transformers, mode=split,
                cls_2=self.config['cls_2'], cls_2xmul=self.config['cls_2xmul']
            ) for split in splits
        }

    def get_dataloader(self):
        splits = ['train', 'val']
        dataloaders = {
            split: DataLoader(
                self.datasets[split],
                self.config[split + '_batch_size'], num_workers=self.config['num_workers'], pin_memory=True,
                shuffle=split == 'train',
                drop_last=False,
                collate_fn=self.collate_fn[split]
            ) for split in splits
        }
        return dataloaders

    def get_criterion(self):
        return DCLLoss(self.config['abg'])

    def get_optimizer(self):
        ignored_params1 = list(map(id, self.model.classifier.parameters()))
        ignored_params2 = list(map(id, self.model.classifier_swap.parameters()))
        ignored_params3 = list(map(id, self.model.Convmask.parameters()))
        ignored_params = ignored_params1 + ignored_params2 + ignored_params3
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())

        classifier_lr = self.config['lr_ratio'] * self.config['lr']
        optimizer = torch.optim.SGD([
            {'params': base_params},
            {'params': self.model.classifier.parameters(), 'lr': classifier_lr},
            {'params': self.model.classifier_swap.parameters(), 'lr': classifier_lr},
            {'params': self.model.Convmask.parameters(), 'lr': classifier_lr},
        ], momentum=0.9, lr=self.config['lr'])
        return optimizer

    def get_scheduler(self):
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config['step_size'],
                                               gamma=self.config['gamma'])

    def train(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.total_epoch):
            self.epoch = epoch
            self.reset_average_meters()
            print(f"Starting epoch {epoch + 1}...")
            self.timer.tick()
            training_bar = tqdm(self.dataloader['train'], ncols=100)
            for data in training_bar:
                self.batch_training(data)
                training_bar.set_description(f'Train Epoch [{self.epoch + 1}/{self.total_epoch}]')
                training_bar.set_postfix(acc=self.average_meters['acc'].avg, loss=self.average_meters['loss'].avg)
            duration = self.timer.tick()
            print(f'Training duration {duration:.2f}s!')
            self.update_performance_meter('train')
            print(f'Starting validation stage in epoch {epoch + 1} ...')
            self.timer.tick()
            # validate
            self.validate()
            duration = self.timer.tick()
            print(f'\nValidation duration {duration:.2f}s!')

            # val stage metrics
            val_acc = self.average_meters['acc'].avg
            if self.performance_meters['val']['acc'].best_value is not None:
                is_best = epoch >= 5 and val_acc > self.performance_meters['val']['acc'].best_value
            else:
                is_best = epoch >= 5
            self.update_performance_meter('val')
            self.do_scheduler_step()
            print(f'Epoch {epoch + 1} Done!')
            # save model
            if is_best or epoch == self.total_epoch - 1:
                print('Saving best model ...')
                self.save_model()

        print(f'best acc:{self.performance_meters["val"]["acc"].best_value}')

    def batch_training(self, data):
        inputs, labels, labels_swap, swap_law, img_names = data

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        labels_swap = labels_swap.to(self.device)
        swap_law = swap_law.to(self.device)

        # forward
        outputs = self.model(inputs)
        # print(outputs[0].shape, outputs[1].shape, outputs[2].shape, labels.shape, labels_swap.shape, swap_law.shape)
        loss = self.criterion(outputs, labels, labels_swap, swap_law)
        if self.config['cls_2xmul']:
            logit = outputs[0] + outputs[1][:, 0:self.num_classes] + outputs[1][:,
                                                                     self.num_classes:2 * self.num_classes]
        else:
            logit = outputs[0]
        acc = accuracy(logit, labels, 1)

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # record accuracy and loss
        self.average_meters['acc'].update(acc, labels.size(0))
        self.average_meters['loss'].update(loss.item(), labels.size(0))

    def validate(self):
        self.model.train(False)
        self.reset_average_meters()

        with torch.no_grad():
            val_bar = tqdm(self.dataloader['val'], ncols=80)
            for data in val_bar:
                self.batch_validate(data)
                val_bar.set_description(f'Val Epoch [{self.epoch + 1}/{self.total_epoch}]')
                val_bar.set_postfix(acc=self.average_meters['acc'].avg)
        self.model.train(True)

    def batch_validate(self, data):
        inputs = data[0].to(self.device)
        labels = data[1].long().to(self.device)

        # forward
        outputs = self.model(inputs)
        if self.config['cls_2xmul']:
            logit = outputs[0] + outputs[1][:, 0:self.num_classes] + outputs[1][:,
                                                                     self.num_classes:2 * self.num_classes]
        else:
            logit = outputs[0]
        acc = accuracy(logit, labels, 1)

        self.average_meters['acc'].update(acc, labels.size(0))

    def do_scheduler_step(self):
        self.scheduler.step()

    def update_performance_meter(self, split):
        if split == 'train':
            self.performance_meters['train']['acc'].update(self.average_meters['acc'].avg)
            self.performance_meters['train']['loss'].update(self.average_meters['loss'].avg)
        elif split == 'val':
            self.performance_meters['val']['acc'].update(self.average_meters['acc'].avg)

    def save_model(self):
        """将模型转为TorchScript，保存到指定路径."""
        output_model_path = os.environ.get("PAI_OUTPUT_MODEL")
        # output_model_path = '../models'
        # os.makedirs(output_model_path, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(output_model_path,
                                                         f"model_acc_{self.average_meters['acc'].avg:.2f}.pth"))
        print('model saved to: ', output_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch DCL Training")
    parser.add_argument(
        "--pretrained",
        default=False,
        type=lambda x: x.lower() == 'true',
        help="use pretrained model (default:False)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='',
        metavar="S",
        help="saved model path",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        metavar="N",
        help="num classes (default:10)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        metavar="N",
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        metavar="N",
        help="num workers (default: 1)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=5,
        metavar="N",
        help="Learning rate step size (default: 5)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--use-cuda", type=lambda x: x.lower() == 'true', default=True, help="enable CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )

    #  下面为DCL专用超参
    parser.add_argument(
        "--lr-ratio",
        type=float,
        default=10,
        metavar="M",
        help="dcl特殊层学习倍率",
    )
    parser.add_argument(
        "--abg",
        type=list,
        default=[1, 1, 1],
        metavar="L",
        help="DCL Loss 计算",
    )

    args = parser.parse_args()
    config = vars(args)
    config['cls_2'] = config['num_classes'] == 2
    config['cls_2xmul'] = config['num_classes'] > 2
    config["data_dir"] = os.environ.get("PAI_INPUT_TRAIN")
    config["meta_dir"] = os.environ.get("PAI_INPUT_META")
    trainer = DCLTrainer(config)
    trainer.train()
