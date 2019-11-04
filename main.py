import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CyclicLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import models
from run import train, val
from sched import CosineLR
from utils import flops_benchmark
from utils.cross_entropy import CrossEntropyLoss
from utils.data import get_loaders
from utils.lookahead import RAdam
from utils.optimizer_wrapper import OptimizerWrapper
from utils.utils import is_bn


def get_args():
    parser = argparse.ArgumentParser(description='Hemmorhage')
    parser.add_argument('--dataroot', required=True, metavar='PATH',
                        help='Path to dataset')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('-j', '--workers', default=7, type=int, metavar='N',
                        help='Number of data loading workers (default: 6)')
    parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')

    # distributed
    parser.add_argument('--world-size', default=-1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int, help='rank of distributed processes')
    parser.add_argument('--dist-init', default='env://', type=str, help='init used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')

    # Optimization options
    parser.add_argument('--optim', dest='optim', type=str, default='sgd', help='Optimizer')
    parser.add_argument('--sched', dest='sched', type=str, default='multistep')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=4e-3, help='The learning rate for batch of 64 '
                                                                                 '(scaled for bigger/smaller batches).')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=1e-2, help='Weight decay for batch of 64 '
                                                                        '(scaled for bigger/smaller batches).')
    parser.add_argument('--gamma', type=float, default=0.7, help='LR is multiplied by gamma at scheduled epochs.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 75],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma-step', type=int, default=1, help='Decrease learning rate after those epochs.')
    parser.add_argument('--step', type=int, default=40, help='Decrease learning rate each time.')
    parser.add_argument('--warmup', default=0, type=int, metavar='N', help='Warmup length')
    parser.add_argument('--smooth-eps', type=float, default=0.1, help='Label smoothing epsilon value.')
    parser.add_argument('--clip-grad', type=float, default=0, help='Clip gradients')

    parser.add_argument('--lookahead', dest='lookahead', action='store_true', help='use lookahead optimizer')
    parser.add_argument('--la-k', type=int, default=6, help='k of lookahead.')
    parser.add_argument('--la-alpha', type=float, default=0.5, help='alpha of lookahead.')
    # Checkpoints
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
    parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='Number of batches between log messages')
    parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: random)')
    parser.add_argument('--determenistic', dest='deter', action='store_true', help='use determenistic environment')
    parser.add_argument('--grad-debug', dest='grad_debug', action='store_true', help='use anomaly detection')

    # Architecture
    parser.add_argument('-a', default='mxresnet50', type=str, metavar='ARCH',
                        help='Architecture. '
                             'Options are {}'.format([x for x in models.__dict__ if not x.startswith('__')]))
    parser.add_argument('--sync-bn', dest='sync_bn', action='store_true', help='use synchronized BN')
    parser.add_argument('--self-attention', '-sa', dest='self_attention', action='store_true',
                        help='use self-attention')
    parser.add_argument('--sa-symmetry', '-sym', dest='sa_symmetry', action='store_true',
                        help='use symmetry for self-attention')
    parser.add_argument('--experiment-name', default='exp1', type=str, help='experiment name')

    parser.add_argument('--num-classes', type=int, default=10, help='Number of classes.')
    parser.add_argument('--input-size', default=128, type=int, help='input size')

    args = parser.parse_args()

    args.distributed = args.local_rank >= 0 or args.world_size > 1
    args.child = args.distributed and args.local_rank > 0
    if not args.distributed:
        args.local_rank = 0
        args.world_size = 1
    if args.local_rank >= args.world_size:
        raise ValueError('World size inconsistent with local rank!')
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save is '':
        args.save = time_stamp
    args.save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(args.save_path) and not args.child:
        os.makedirs(args.save_path)

    if args.device == 'cuda' and torch.cuda.is_available():
        cudnn.enabled = True
        if args.deter:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.benchmark = True
        if args.grad_debug:
            torch.autograd.set_detect_anomaly(True)
        args.gpus = [args.local_rank]
        args.device = 'cuda:' + str(args.gpus[0])
        torch.cuda.set_device(args.gpus[0])
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpus = []
        args.device = 'cpu'

    if args.type == 'float64':
        args.dtype = torch.float64
    elif args.type == 'float32':
        args.dtype = torch.float32
    elif args.type == 'float16':
        args.dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8

    # Adjust lr for batch size
    args.learning_rate *= args.batch_size / 256. * args.world_size
    args.decay *= args.batch_size / 256. * args.world_size

    args.epochs -= 1

    return args


def get_logger(args):
    # LOGGING SETUP
    logger = logging.getLogger('rmish')
    logger.setLevel(logging.DEBUG)
    # file handler
    fh = logging.FileHandler(os.path.join(args.save_path, args.experiment_name + '.log'))
    fh.setLevel(logging.DEBUG)
    # console handler with a higher log level (TODO: tqdm)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    if not args.child:
        logger.addHandler(ch)

    # put generic info on args
    logger.debug('{}'.format(args))
    logger.info('Seed = {}'.format(args.seed))
    logger.info('Device, dtype = {}, {}'.format(args.device, args.dtype))
    return logger


def main():
    import warnings

    # filter out corrupted images warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    start = datetime.now()
    args = get_args()
    logger = get_logger(args)
    device, dtype = args.device, args.dtype

    train_loader, val_loader = get_loaders(args.dataroot, args.batch_size, args.batch_size, args.input_size,
                                           args.workers, args.world_size, args.local_rank)
    model_class, model_args = models.__dict__[args.a], {'c_out': args.num_classes, 'sa': args.self_attention,
                                                        'sym': args.sa_symmetry}
    model = model_class(**model_args)
    criterion = CrossEntropyLoss(smooth_eps=args.smooth_eps)

    model, criterion = model.to(device=device, dtype=dtype), criterion.to(device=device, dtype=dtype)
    if args.dtype == torch.float16:
        for module in model.modules():  # FP batchnorm
            if is_bn(module):
                module.to(dtype=torch.float32)  # github.com/pytorch/pytorch/issues/20634

    num_parameters = sum([l.nelement() for l in model.parameters()])
    flops = flops_benchmark.count_flops(model_class, 2, device, dtype, args.input_size, 3, **model_args)

    logger.debug(model)
    logger.info('number of parameters: {}'.format(num_parameters))
    logger.info('GFLOPs: {}'.format(flops / 10 ** 9))
    logger.info('Results saved to {}'.format(args.save_path))

    if args.distributed:
        args.device_ids = [args.local_rank]
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_init, world_size=args.world_size,
                                rank=args.local_rank)
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
        print('Node #{}'.format(args.local_rank))
        logger.info('Node #{}'.format(args.local_rank))
    else:
        model = torch.nn.parallel.DataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    best_test = 0

    # optionally resume from a checkpoint
    writer_log_dir = os.path.join(args.save_path, 'tb{}'.format(args.local_rank))  # TODO only main
    writer = SummaryWriter(log_dir=args.save_path)
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            args.start_step = len(train_loader) * args.start_epoch
            optim = init_optimizer(args, train_loader, model, writer, args.experiment_name, checkpoint['optimizer'],
                                   clip_grad=args.clip_grad)
            best_test = checkpoint['best_ll']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint{}.pth.tar'.format(args.local_rank))
            logger.info("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = checkpoint['epoch']
            args.start_step = len(train_loader) * args.start_epoch
            optim = init_optimizer(args, train_loader, model, writer, args.experiment_name, checkpoint['optimizer'],
                                   clip_grad=args.clip_grad)
            best_test = checkpoint['best_ll']
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
            optim = init_optimizer(args, train_loader, model, writer, args.experiment_name, clip_grad=args.clip_grad)
    else:
        optim = init_optimizer(args, train_loader, model, writer, args.experiment_name, clip_grad=args.clip_grad)

    logger.info("Prepared in {}".format(datetime.now() - start))

    train_network(args.start_epoch, args.epochs, optim, model, train_loader, val_loader, criterion, device, dtype,
                  writer, best_test, args.child, args.experiment_name, logger)


def train_network(start_epoch, epochs, optim, model, train_loader, val_loader, criterion, device, dtype, writer,
                  best_test, child, experiment_name, logger):
    my_range = range if child else trange
    train_it, val_it = 0, 0
    for epoch in my_range(start_epoch, epochs + 1):
        train_it, _, train_accuracy1, train_accuracy5 = train(model, train_loader, logger, writer, experiment_name,
                                                              epoch, train_it, optim, criterion, device, dtype, child)
        val_it, _, val_accuracy1, val_accuracy5 = val(model, val_loader, logger, criterion, writer, experiment_name,
                                                      epoch, val_it, device, dtype, child)
        optim.epoch_step()

        if val_accuracy1 > best_test:
            best_test = val_accuracy1
        logger.debug('Best validation accuracy so far is {:.2f}% top-1'.format(best_test * 100.))

    logger.info('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))


def init_optimizer(args, train_loader, model, writer=None, experiment_name=None, optim_state_dict=None, clip_grad=None):
    if args.optim == 'sgd':
        optimizer_class = torch.optim.SGD
        optimizer_params = {"lr": args.learning_rate, "momentum": args.momentum, "weight_decay": args.decay,
                            "nesterov": True}
    elif args.optim == 'adam':
        optimizer_class = torch.optim.Adam
        optimizer_params = {"lr": args.learning_rate, "weight_decay": args.decay}
    elif args.optim == 'adamw':
        optimizer_class = torch.optim.AdamW
        optimizer_params = {"lr": args.learning_rate, "weight_decay": args.decay, "amsgrad": args.amsgrad}
    elif args.optim == 'radam':
        optimizer_class = RAdam
        optimizer_params = {"lr": args.learning_rate, "weight_decay": args.decay}
    else:
        raise ValueError('Wrong optimizer!')

    if args.sched == 'clr':
        scheduler_class = CyclicLR
        scheduler_params = {"base_lr": args.min_lr, "max_lr": args.max_lr,
                            "step_size_up": args.epochs_per_step * len(train_loader), "mode": args.mode,
                            "last_epoch": args.start_step - 1}
    elif args.sched == 'multistep':
        scheduler_class = MultiStepLR
        scheduler_params = {"milestones": args.schedule, "gamma": args.gamma, "last_epoch": args.start_epoch - 1}
    elif args.sched == 'cosine':
        scheduler_class = CosineLR
        scheduler_params = {"max_epochs": args.epochs, "warmup_epochs": args.warmup, "iter_in_epoch": len(train_loader),
                            "last_epoch": args.start_step - 1}
    elif args.sched == 'gamma':
        scheduler_class = StepLR
        scheduler_params = {"step_size": args.gamma_step, "gamma": args.gamma, "last_epoch": args.start_epoch - 1}
    else:
        raise ValueError('Wrong scheduler!')
    optim = OptimizerWrapper(model, optimizer_class=optimizer_class, optimizer_params=optimizer_params,
                             optimizer_state_dict=optim_state_dict, scheduler_class=scheduler_class,
                             scheduler_params=scheduler_params, use_shadow_weights=args.dtype == torch.float16,
                             writer=writer, experiment_name=experiment_name, clip_grad=clip_grad,
                             lookahead=args.lookahead, lookahead_params={'k': args.la_k, 'alpha': args.la_alpha})
    return optim


if __name__ == '__main__':
    main()
