import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os


# =========================
# Train for one epoch
# =========================
def train_one_epoch(epoch_index, model, loss_fun, optimizer, device, train_loader, tb_writer, use_AMP=False):

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Clear GPU cache (optional but helps in long runs)
    torch.cuda.empty_cache()

    # Mixed precision scaler (for faster training on GPU)
    scaler = torch.amp.GradScaler("cuda")

    model.train()  # Ensure model is in training mode

    for i, (inputs, labels) in enumerate(train_loader):

        # Move data to device (GPU / CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # =========================
        # Forward + Backward pass
        # =========================
        if use_AMP:
            # Mixed precision forward pass
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = loss_fun(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)
            loss.backward()
            optimizer.step()

        # =========================
        # Statistics
        # =========================
        running_loss += loss.item()

        predicted = outputs.argmax(dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Log every 50 batches
        if i % 50 == 49:
            avg_loss = running_loss / 50
            print(f"Batch {i+1}, Loss: {avg_loss:.4f}")

            step = epoch_index * len(train_loader) + i + 1
            tb_writer.add_scalar("Loss/train", avg_loss, step)

            running_loss = 0.0

    # Epoch accuracy
    train_accuracy = 100.0 * correct_predictions / total_samples

    print(f"\nEpoch {epoch_index+1} Summary")
    print(f"Train Accuracy: {train_accuracy:.2f}%")

    return avg_loss, train_accuracy


# =========================
# Full Training Loop
# =========================
def train(model, Epochs, val_loader, device, loss_fun, optimizer,
          train_loader, save_dir, writer, use_AMP=False, epoch_number=0):

    print("Training started...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for epoch in range(Epochs):

        print(f"\n================ Epoch {epoch_number + 1} ================")

        # =========================
        # TRAINING PHASE
        # =========================
        model.train()

        avg_loss, train_accuracy = train_one_epoch(
            epoch_number,
            model,
            loss_fun,
            optimizer,
            device,
            train_loader,
            writer,
            use_AMP
        )

        # =========================
        # VALIDATION PHASE
        # =========================
        model.eval()

        running_vloss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for vinputs, vlabels in val_loader:

                vinputs, vlabels = vinputs.to(device), vlabels.to(device)

                voutputs = model(vinputs)
                vloss = loss_fun(voutputs, vlabels)

                running_vloss += vloss.item()

                vpredicted = voutputs.argmax(dim=1)

                correct_predictions += (vpredicted == vlabels).sum().item()
                total_samples += vlabels.size(0)

        # Validation metrics
        avg_vloss = running_vloss / len(val_loader)
        val_accuracy = 100.0 * correct_predictions / total_samples

        print(f"Train Acc: {train_accuracy:.2f}% | Val Acc: {val_accuracy:.2f}%")
        print(f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_vloss:.4f}")

        # =========================
        # LOGGING (TensorBoard)
        # =========================
        writer.add_scalars(
            "Loss",
            {"train": avg_loss, "val": avg_vloss},
            epoch_number + 1
        )

        writer.add_scalars(
            "Accuracy",
            {"train": train_accuracy, "val": val_accuracy},
            epoch_number + 1
        )

        writer.flush()

        # =========================
        # SAVE MODEL CHECKPOINT
        # =========================
        model_path = os.path.join(
            save_dir,
            f"model_{timestamp}_epoch_{epoch_number}.pth"
        )

        torch.save(model.state_dict(), model_path)

        print(f"Model saved at: {model_path}")

        epoch_number += 1

    return val_accuracy, avg_vloss, avg_loss, train_accuracy