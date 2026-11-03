import argparse
from pathlib import Path

import torch

from residual_connections.data import load_data
from residual_connections.resnet import ResNet
from residual_connections.training import train


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(
        description="Train a CIFAR-10 ResNet, evaluate it, and save its weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epochs", type=positive_int, default=15, help="Training epochs"
    )
    parser.add_argument(
        "--batch-size", type=positive_int, default=64, help="Batch size"
    )
    parser.add_argument(
        "--augment", action="store_true", help="Use random training flips and crops"
    )
    parser.add_argument(
        "--blocks-per-stage",
        type=positive_int,
        default=3,
        help="Residual blocks per stage; network depth is 6 * blocks + 2",
    )
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Initial SGD learning rate"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument(
        "--weight-decay", type=float, default=0.0001, help="SGD weight decay"
    )
    parser.add_argument(
        "--device", default="auto", help="Torch device, e.g. cpu, cuda, cuda:0, or mps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Torch random seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("resnet.pt"),
        help="Destination checkpoint (overwritten if it exists)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_loader, val_loader, test_loader = load_data(
        batch_size=args.batch_size, augment=args.augment
    )
    model = ResNet(n=args.blocks_per_stage)
    print(
        f"ResNet-{6 * args.blocks_per_stage + 2} | "
        f"Train: {len(train_loader.dataset):,} | "
        f"Validation: {len(val_loader.dataset):,} | "
        f"Test: {len(test_loader.dataset):,} | Augmentation: {args.augment}"
    )
    train(
        model,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        device=None if args.device == "auto" else torch.device(args.device),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy:.2%}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(args.output)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "test_accuracy": test_accuracy,
        },
        args.output,
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
