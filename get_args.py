import itertools

PARAMS = [
    ('dataset_path', ['/lustre4/2/managed_datasets/LIDC-IDRI/npy/lanczos_3d/']),
    ('final_resolution', [512]),
    ('final_zdim', [128]),
    ('--starting_phase', [2]),
    ('--ending_phase', [4]),
    ('--base_dim', [256]),
    ('--latent_dim', [256]),
    ('--base_batch_size', [4]),  # Uncomment and fill to use.
    ('--mixing_nimg', [2 ** 16]),  # Maybe half these to speed up the search?
    ('--stabilizing_nimg', [2 ** 16]),
    ('--learning_rate', [1e-3]),
    ('--gp_center', [1]),
    ('--gp_weight', [10]),  # Maybe change to add 20 if computation allows.
    ('--activation', ['leaky_relu']),  # Maybe add swish or leaky celu layer.
    ('--leakiness', [0.3]),
    ('--seed', [42]),
    ('--horovod', ['']),
    # ('--fp16_allreduce', ['']),  # Uncomment if you want this.
    ('--calc_metrics', ['']),
    # ('--use_ext_clf', ['']),  # Comment to disable
    ('--g_annealing', [1]),
    ('--d_annealing', [1]),
    ('--num_metric_samples', [128]),
    ('--beta1', [0]),
    ('--beta2', [.99]),
    ('--d_scaling', ['sqrt']),
    ('--g_scaling', ['sqrt']),
    # ('--lr_warmup_epochs', [5])
]

PARAMS_LISTS = list(PARAMS[i][1] for i in range(len(PARAMS)))
combinations = list(itertools.product(*PARAMS_LISTS))
# print(f"Number of hyperparameter combinations: {len(combinations)}")

ARGS = ''
for i, combination in enumerate(combinations):
    ARG = ''

    for j, value in enumerate(combination):
        param = PARAMS[j][0]

        if param.startswith('--'):
            ARG += f' {param} {value}'

        else:
            ARG += f' {value}'

    ARGS += ARG
    if i < len(combinations) - 1:
        ARGS += ' ; '

# for arg in ARGS.split(';'):
#     print(f'\n > {arg} \n')

print(ARGS)
