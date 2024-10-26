def early_stopping(val_loss, best_val_loss, epochs_without_improvement, patience):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1
    return best_val_loss, epochs_without_improvement
