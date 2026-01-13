import torch
from torch import nn


def train_forecast_model(
    model,
    dataloader,
    epochs,
    optimizer,
    loss_fn=None,
    device=None,
    save_path=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = loss_fn or nn.MSELoss()
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            inputs = batch["LR"].to(device)
            targets = batch["HR"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(1, len(dataloader))
        history.append(avg_loss)
        print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f}")

    if save_path:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epochs": epochs,
                "loss": history[-1] if history else None,
            },
            save_path,
        )
        print(f"Checkpoint saved to: {save_path}")

    return model, history
