from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

def create_warmup_cosine_scheduler(optimizer, total_epochs, warmup_epochs=5, warmup_start_factor=0.01, eta_min=0):
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=eta_min
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    return scheduler

def create_warmup_constant_cosine_scheduler(optimizer, warmup_steps, constant_steps, cosine_steps, warmup_start_factor=0.01, eta_min=0):
    linear_phase_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=eta_min
    )
    
    scheduler = SequentialLR(
        optimizer,
        schedulers=[linear_phase_scheduler, cosine_scheduler],
        milestones=[warmup_steps + constant_steps]
    )
    
    return scheduler