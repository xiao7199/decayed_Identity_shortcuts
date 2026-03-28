# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import builtins
import math
import numpy as np
import datetime
import os
import glob
import tqdm
import time
from collections import defaultdict, deque
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import inf
nn = torch.nn
F = torch.nn.functional
#from torch._six import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank,
                                         timeout=datetime.timedelta(seconds=6000000))
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, save_last = False):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        if save_last:
            checkpoint_paths = [output_dir / ('checkpoint-last.pth')]
        else:
            checkpoint_paths = [output_dir / ('checkpoint-{:05d}.pth'.format(epoch))]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }
            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-{:05d}".format(epoch), client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        elif args.resume == 'last':
            output_dir = Path(args.output_dir)
            model_list = glob.glob(args.output_dir + '/checkpoint-last.pth')
            if len(model_list) == 0:
                print('No checkpoint found at {}'.format(args.output_dir))
                return
            checkpoint = torch.load(model_list[-1], map_location='cpu')
            args.resume = model_list[-1]
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x

def eval_model(args, epoch, model, num_of_gt = 100):
    model.eval()
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    feat_list = []

    transform_train = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_train)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train, shuffle = False, drop_last = False)
    train_loader = torch.utils.data.DataLoader(
        dataset_train, sampler=train_sampler,
        shuffle = False,
        batch_size=128,
        num_workers=10,
        pin_memory=False,
        drop_last=False,
        persistent_workers = False,
    )
    test_loader= torch.utils.data.DataLoader(
        dataset_val, sampler=None,
        batch_size=256,
        num_workers=5,
        pin_memory=False,
        drop_last=False,
        persistent_workers = False,
    )

    with torch.no_grad():
        all_feat_list = None
        label_list = []
        for img, label in tqdm.tqdm(train_loader):
            label = label.to(args.gpu)
            with torch.no_grad():
                img = img.to(args.gpu)
                with torch.cuda.amp.autocast():
                    feat_list = model.module.forward_feat(img)
                if all_feat_list is None:
                    all_feat_list = [[] for _ in range(len(feat_list))]
            for m in range(len(all_feat_list)):
                all_feat_list[m].append(feat_list[m].mean(dim = [1]))
            label_list.append(label)
        train_label = torch.cat(label_list).to(args.gpu)
        train_label = concat_all_gather(train_label)
        train_feat_list =[]
        for m in range(len(all_feat_list)):
            train_feat = torch.cat(all_feat_list[m])
            train_feat = concat_all_gather(train_feat)
            train_feat_list.append(train_feat)

    class LinearProb(nn.Module):
        def __init__(self, feat_dim_list, num_of_gt):
            super().__init__()
            self.cls = nn.ModuleList()
            for feat_dim in feat_dim_list:
                self.cls.append(nn.Linear(feat_dim, num_of_gt))

        def forward(self, x_list):
            out = []
            for cls, x in zip(self.cls, x_list):
                out.append(cls(x))
            return out

    total_gpu = get_world_size()
    batch_size = 256
    total_batch_size = batch_size * total_gpu
    all_feat_list = None
    prob_model = None
    label_list = []
    learning_rate = 1e-2
    total_epochs = 200
    def adjust_learning_rate(optimizer, epoch):
        """Decay the learning rate based on schedule"""
        #lr = args.lr
        #if args.cos:  # cosine lr schedule
        lr = learning_rate
        lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / total_epochs))
        ##else:  # stepwise lr schedule
        #for milestone in milestone_list:
        #    lr *= 0.1 if epoch >= milestone else 1.0
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    for ep in tqdm.tqdm(range(total_epochs)):
        idx = torch.randperm(train_label.shape[0])
        idx_array = idx[:train_label.shape[0] // total_batch_size * total_batch_size].reshape(-1, total_batch_size)
        for bid in range(idx_array.shape[0]):
            total_bid = idx_array[bid].reshape(-1, batch_size).to(args.gpu)
            torch.distributed.broadcast(total_bid, src = 0)
            batch_bid = total_bid[args.gpu]
            batch_label = train_label[batch_bid]
            batch_feat_list = [train_feat[batch_bid] for train_feat in train_feat_list]
            if prob_model is None:
                prob_model = LinearProb([feat.shape[-1] for feat in feat_list], num_of_gt)
                prob_model.to(args.gpu)
                prob_model = torch.nn.parallel.DistributedDataParallel(prob_model, device_ids=[args.gpu], find_unused_parameters=True)
                #optimizer = torch.optim.SGD(prob_model.parameters(), learning_rate, momentum = 0.9, weight_decay = 1e-4)
                optimizer = torch.optim.AdamW(prob_model.parameters(), learning_rate)
            out_list = prob_model(batch_feat_list)
            loss = 0
            for out in out_list:
                loss += torch.nn.CrossEntropyLoss()(out, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        adjust_learning_rate(optimizer, ep)

        if ep % 50 == 0 or ep == total_epochs - 1:
            with torch.no_grad():
                pred_list = [[] for i in range(len(out_list))]
                feat_norm_list = [[] for i in range(len(out_list))]
                for img, label in test_loader:
                    label = label.to(args.gpu)
                    with torch.no_grad():
                        img = img.to(args.gpu)
                        with torch.cuda.amp.autocast():
                            feat_list = [feat.mean(dim = 1) for feat in model.module.forward_feat(img)]
                        out_list = prob_model(feat_list)
                    for feat, norm in zip(feat_list, feat_norm_list):
                        norm.append(feat.norm(dim = -1).mean())
                    for out, pred in zip(out_list, pred_list):
                        pred.append((out.argmax(dim = -1) == label).float())
    if args.gpu == 0:
        norm_list = []
        acc_list = []
        for idx, (pred, feat_norm) in enumerate(zip(pred_list, feat_norm_list)):
            norm_list.append(torch.stack(feat_norm).mean().item())
            #clf.fit(train_feat.cpu().data.numpy(), train_label.cpu().data.numpy())
            acc_list.append(torch.cat(pred).mean().item())
            print(f'idx:{idx}, accuracy:{acc_list[-1]}')
            fig, ax1 = plt.subplots(1,1,figsize=(10, 5))
            color = 'tab:red'
            ax1.set_xlabel('layer index')
            ax1.set_ylabel('accuracy', color=color)
            ax1.plot(np.arange(len(acc_list)), np.array(acc_list), color=color, label = 'accuracy')
            ax1.tick_params(axis='y', labelcolor=color)
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('feat norm', color=color)  # we already handled the x-label with ax1
            ax2.plot(np.arange(len(norm_list)), np.array(norm_list).reshape(-1), color=color, label = 'accuracy')
            ax2.tick_params(axis='y', labelcolor=color)
            # fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.title(f'{acc_list[-1]}', fontsize=10)
            plt.savefig(f'{args.output_dir}/eval_{epoch:05d}.png')
            plt.close()

    torch.distributed.barrier()
    model.train()

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
