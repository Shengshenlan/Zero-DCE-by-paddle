import paddle
import paddle.optimizer
import os
import argparse
import dataloader
import model
import Myloss
import paddle
import paddle.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.initializer.Normal(mean=0.0, std=0.02)(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.initializer.Normal(mean=1.0, std=0.02)(m.weight)
        nn.initializer.Constant(value=0)(m.bias)


def train(config):
    if paddle.is_compiled_with_cuda():
        paddle.set_device('gpu:0')
        print("Using CUDA device.")
    else:
        paddle.set_device('cpu')
        print("CUDA device not found. Using CPU.")
    DCE_net = model.enhance_net_nopool()
    DCE_net.apply(weights_init)
    if config.load_pretrain == True:
        DCE_net.load_state_dict(paddle.load(config.pretrain_dir))
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
    train_loader = paddle.io.DataLoader(train_dataset,
                                        batch_size=config.train_batch_size,
                                        shuffle=True,
                                        num_workers=config.num_workers)
    L_color = Myloss.L_color()
    L_spa = Myloss.L_spa()
    L_exp = Myloss.L_exp(16, 0.6)
    L_TV = Myloss.L_TV()

    clip = paddle.nn.ClipGradByNorm(clip_norm=config.grad_clip_norm)

    optimizer = paddle.optimizer.Adam(learning_rate=config.lr,
                                      parameters=DCE_net.parameters(),
                                      weight_decay=config.weight_decay,
                                      grad_clip=clip
                                      )
    DCE_net.train()
    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
            Loss_TV = 200 * L_TV(A)
            loss_spa = paddle.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5 * paddle.mean(L_color(enhanced_image))
            loss_exp = 10 * paddle.mean(L_exp(enhanced_image))
            loss = Loss_TV + loss_spa + loss_col + loss_exp
            optimizer.clear_grad()
            loss.backward()

            optimizer.step()
            if (iteration + 1) % config.display_iter == 0:
                print('Loss at iteration', iteration + 1, ':', loss.item())
            if (iteration + 1) % config.snapshot_iter == 0:
                paddle.save(DCE_net.state_dict(), config.snapshots_folder +
                            'Epoch' + str(epoch) + '.pdiparams')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lowlight_images_path', type=str, default= \
        'data/train_data/')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default='snapshots/')
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default= \
        'snapshots/Epoch99.pdiparams')
    config = parser.parse_args()
    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    train(config)
