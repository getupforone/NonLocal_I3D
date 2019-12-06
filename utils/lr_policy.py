import math

def get_lr_func(lr_policy):
    policy = "lr_func_" + lr_policy
    if policy not in globals():
        raise NotImplementedError("Unknown LR policy: {}".format(lr_policy))
    else:
        return globals()[policy]
def get_lr_at_epoch(cfg, cur_epoch):
    lr = get_lr_func(cfg.SOLVER.LR_POLICY)(cfg.cur_epoch)
    if cur_epoch < cfg.SOLVER.WARMUP_EPOCHS:
        lr_start = cfg.SOLVER.WARMUP_START_LR
        lr_end = get_lr_func(cfg.SOLVER.LR_POLICY)(
            cfg, cfg.SOLVER.WARMUP_EPOCHS
        )
        alpha = (lr_end - lr_start) / cfg.SOLVER.WARMUP_EPOCHS
        lr = cur_epoch * alpha + lr_start

    return lr
def lr_func_cosine(cfg, cur_epoch):
    return (
        cfg.SOLVER.BASE_LR
        * (math.cos(math.pi * cur_epoch / cfg.SOLVER.MAX_EPOCH) + 1.0)
        * 0.5
    )
def get_step_index(cfg, cur_epoch):
    steps = cfg.SOLVER.STEPS + [cfg.SOLVER.MAX_EPOCH]
    for ind, step in enumerate(steps):
        if cur_epoch < step:
            break
    return ind - 1

def lr_func_steps_with_relative_lrs(cfg, cur_epoch):
    ind = get_step_index(cfg,cur_epoch)
    return cfg.SOLVER.LRS[ind]* cfg.SOLVER.BASE_LR


