def learning_rate_schedule(base_lr, epoch, total_epochs):
    alpha = epoch / total_epochs
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return factor * base_lr


def l2_regularizer(weight_decay):
    return lambda model: 0.5 * weight_decay * model.l2


def adjust_learning_rate():
    # For PyTorch:
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # return lr
    pass


def check_batch_normalization():
    pass
