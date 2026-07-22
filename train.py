import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from utils.utils import *
from utils.model import *
from tqdm import tqdm
from torchvision.utils import save_image


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default='./content_data',
                        help='Location of content dataset')
    parser.add_argument('--style_dir', type=str, default='./style_data',
                        help='Location of style dataset')
    parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth',
                        help='Location of pre-trained VGG')
    parser.add_argument('--experiment', type=str, default='experiment1',
                        help='Name of experiment')

    parser.add_argument('--final_size', type=int, default=256,
                        help='Size of final image')
    parser.add_argument('--content_size', type=int, default=512,
                        help='Size of content image')
    parser.add_argument('--style_size', type=int, default=512,
                        help='Size of style image')
    parser.add_argument('--crop', action='store_true', default=True,
                        help='Crop image')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5,
                        help='Learning rate decay')

    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs')

    parser.add_argument('--content_weight', type=float, default=1.0,
                        help='Content weight')
    parser.add_argument('--style_weight', type=float, default=5,
                        help='Style weight')

    parser.add_argument('--log_interval', type=int, default=1,
                        help='Log interval')

    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save interval')

    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training')

    parser.add_argument('--decoder_path', type=str, default=None,
                        help='Path to decoder checkpoint')

    parser.add_argument('--optimizer_path', type=str, default=None,
                        help='Path to optimizer checkpoint')

    return parser.parse_args()


def main():
    args = parse_arguments()

    # ── Device setup ──────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Using device: {device}")
    print(f"Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── Save directory ────────────────────────────────────
    if os.path.exists('/kaggle/working'):
        save_dir = Path('/kaggle/working/experiment') / args.experiment
    else:
        save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # ── Save arguments ────────────────────────────────────
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')
    print("args.txt written!")

    # ── Transforms and datasets ───────────────────────────
    content_transform = get_transform(
        args.content_size, args.crop, args.final_size
    )
    style_transform = get_transform(
        args.style_size, args.crop, args.final_size
    )

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset   = ImageFolderDataset(args.style_dir,   style_transform)

    print(f"Content images: {len(content_dataset)}")
    print(f"Style images:   {len(style_dataset)}")

    # ── DataLoaders ───────────────────────────────────────
    content_dataloader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=4,
        drop_last=True
    )
    style_dataloader = DataLoader(
        style_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=4,
        drop_last=True
    )

    print(f"Content batches: {len(content_dataloader)}")
    print(f"Style batches:   {len(style_dataloader)}")

    # ── Models ────────────────────────────────────────────
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    # ── Multi-GPU setup ───────────────────────────────────
    if num_gpus > 1:
        print(f"Using DataParallel across {num_gpus} GPUs!")
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    print("Encoder and Decoder loaded!")

    # ── Optimizer ─────────────────────────────────────────
    decoder_params = (
        decoder.module.parameters()
        if hasattr(decoder, 'module')
        else decoder.parameters()
    )
    optimizer = optim.Adam(decoder_params, lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    # ── Resume from checkpoint ────────────────────────────
    if args.resume:
        print(f"Resuming from {args.decoder_path}")
        state_dict = torch.load(
            args.decoder_path, map_location=device
        )
        if hasattr(decoder, 'module'):
            decoder.module.load_state_dict(state_dict)
        else:
            decoder.load_state_dict(state_dict)

        optimizer.load_state_dict(
            torch.load(args.optimizer_path, map_location=device)
        )
        print("Checkpoint loaded successfully!")

    # ── Training loop ─────────────────────────────────────
    print("Training...")
    mse_loss = nn.MSELoss()
    encoder.eval()

    for epoch in range(args.epochs):
        progress_bar = tqdm(
            zip(content_dataloader, style_dataloader),
            total=min(len(content_dataloader), len(style_dataloader))
        )

        running_loss  = 0
        running_closs = 0
        running_sloss = 0

        for content_batch, style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch   = style_batch.to(device)

            # ── Encode ────────────────────────────────────
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            # handle DataParallel tuple output
            c_feats_last = c_feats[-1] if isinstance(
                c_feats, (list, tuple)) else c_feats
            s_feats_last = s_feats[-1] if isinstance(
                s_feats, (list, tuple)) else s_feats

            # ── AdaIN + Decode ────────────────────────────
            t = adaptive_instance_normalization(c_feats_last, s_feats_last)
            g = decoder(t)

            # ── Encode generated image ────────────────────
            g_feats = encoder(g)

            g_feats_last = g_feats[-1] if isinstance(
                g_feats, (list, tuple)) else g_feats

            # ── Content loss ──────────────────────────────
            loss_c = mse_loss(g_feats_last, t) * args.content_weight

            # ── Style loss ────────────────────────────────
            loss_s = 0
            g_feats_list = g_feats if isinstance(
                g_feats, (list, tuple)) else [g_feats]
            s_feats_list = s_feats if isinstance(
                s_feats, (list, tuple)) else [s_feats]

            for g_f, s_f in zip(g_feats_list, s_feats_list):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)
            loss_s = loss_s * args.style_weight

            # ── Total loss + backprop ─────────────────────
            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(
                f'Loss:{loss.item():.4f} '
                f'C:{loss_c.item():.4f} '
                f'S:{loss_s.item():.4f}'
            )

            running_loss  += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()

        scheduler.step()

        n = len(content_dataloader)
        running_loss  /= n
        running_closs /= n
        running_sloss /= n

        if (epoch + 1) % args.log_interval == 0:
            tqdm.write(
                f'Epoch {epoch+1}/{args.epochs} — '
                f'Loss:{running_loss:.4f} '
                f'C:{running_closs:.4f} '
                f'S:{running_sloss:.4f}'
            )

        if (epoch + 1) % args.save_interval == 0:
            # save decoder weights
            decoder_state = (
                decoder.module.state_dict()
                if hasattr(decoder, 'module')
                else decoder.state_dict()
            )
            torch.save(
                decoder_state,
                save_dir / f'decoder_{epoch+1}.pth'
            )
            torch.save(
                optimizer.state_dict(),
                save_dir / f'optimizer_{epoch+1}.pth'
            )

            # save sample image
            with torch.no_grad():
                output = torch.cat(
                    [content_batch, style_batch, g], dim=0
                )
                save_image(
                    output,
                    save_dir / f'output_{epoch+1}.png',
                    nrow=args.batch_size
                )
            print(f"Saved checkpoint and sample at epoch {epoch+1}")


if __name__ == '__main__':
    main()