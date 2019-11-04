import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm

from utils.utils import correct


def train(model, loader, logger, writer, experiment_name, epoch, iter, optim, criterion, device, dtype, child):
    model.train()
    correct1, correct5 = 0, 0

    enum_load = enumerate(loader) if child else enumerate(tqdm(loader))
    for batch_idx, (data, target) in enum_load:
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optim.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optim.batch_step()

        corr = correct(output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

        writer.add_scalars('batch/train_loss', {experiment_name: loss.item()}, iter)
        writer.add_scalars('batch/top1', {experiment_name: corr[0] / data.shape[0]}, iter)
        writer.add_scalars('batch/top5', {experiment_name: corr[1] / data.shape[0]}, iter)
        iter += 1
    logger.debug(
        'Train Epoch: {}\tLoss: {:.6f}. '
        'Top-1 accuracy: {:.2f}%. '
        'Top-5 accuracy: {:.2f}%.'.format(epoch, loss.item(), 100 * correct1 / len(loader.sampler),
                                          100 * correct5 / len(loader.sampler)))
    return iter, loss.item(), correct1 / len(loader.sampler), correct5 / len(loader.sampler)


def val(model, loader, logger, criterion, writer, experiment_name, epoch, iter, device, dtype, child):
    model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0

    enum_load = enumerate(loader) if child else enumerate(tqdm(loader))

    with torch.no_grad():
        for batch_idx, (data, target) in enum_load:
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()  # sum up batch loss
            corr = correct(output, target, topk=(1, 5))
            correct1 += corr[0]
            correct5 += corr[1]

            writer.add_scalars('batch/val_loss', {experiment_name: loss.item()}, iter)
            iter += 1

    test_loss /= len(loader)

    logger.debug(
        'Validation Epoch: {}\tLoss: {:.6f}. '
        'Top-1 accuracy: {:.2f}%. '
        'Top-5 accuracy: {:.2f}%.'.format(epoch, test_loss, correct1 / len(loader.sampler),
                                          correct5 / len(loader.sampler)))
    return iter, test_loss, correct1 / len(loader.sampler), correct5 / len(loader.sampler)
