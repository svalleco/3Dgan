import argparse
import horovod as hvd




def dataset():
    """TODO: Docstring for dataset.

    :function: TODO
    :returns: Return NumpyDataset

    """
    data_path = os.path.join(args.dataset_path, f'{size}x{size}/')
    npy_data = NumpyPathDataset(data_path, args.scratch_path, copy_files=local_rank == 0, is_correct_phase=phase >= args.starting_phase)
    
    return npy_data


def optimizers(arg1):
    """TODO: Docstring for optimizers.

    :function: TODO
    :returns: TODO

    """
    


def run(args,config):
    """TODO: Docstring for run.
    The main function, training done here 

    :a: TODO
    :returns: TODO

    """

    if args.horovod:
        verbose = hvd.rank() == 0
        global_size = hvd.size()
        global_rank = hvd.rank()
        local_rank = hvd.local_rank()
    else:
        verbose = True
        global_size = 1
        global_rank = 0
        local_rank = 0

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', args.architecture, timestamp)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    if verbose:
        writer = tf.summary.FileWriter(logdir=logdir)
        print("Arguments passed:")
        print(args)
        print(f"Saving files to {logdir}")

    else:
        pass

    
    final_shape = parse_tuple(args.final_shape)
    image_channels = final_shape[0]
    final_resolution = final_shape[-1]
    num_phases = int(np.log2(final_resolution) - 1)
    base_dim = num_filters(-num_phases + 1, num_phases, size=args.network_size)

    var_list = list()
    global_step = 0

    
    # -------------
    # Phasing Loop
    #-------------

    for phase in range(1, num_phases + 1):
        
        tf.reset_default_graph()
        npy_data = dataset() 

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str)

    args = parser.parse_args()

    if args.horovod:
        hvd.init()

    config = ''
    
    discriminator = importlib.import_module(f'networks.{args.architecture}.discriminator').discriminator
    generator = importlib.import_module(f'networks.{args.architecture}.generator').generator

    run(args,config)
