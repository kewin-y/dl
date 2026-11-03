import torch
from torch import nn, optim


def train(
    model,
    train_dataloader,
    val_dataloader,
    *,
    num_epochs=15,
    device=None,
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
):
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using device: {device}")

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    lr_milestones = {32_000, 48_000}

    it = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_dataloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

            it += 1
            if it in lr_milestones:
                for group in optimizer.param_groups:
                    group["lr"] /= 10
                print(f"  [iter {it:,}] lr -> {optimizer.param_groups[0]['lr']:g}")

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_dataloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                valid_loss += loss.item() * labels.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

        valid_loss /= total
        valid_acc = correct / total

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
            f"Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc:.2%}"
        )
