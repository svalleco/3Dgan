import argparse
import horovod as hvd




def dataset(arg1):
    """TODO: Docstring for dataset.

    :function: TODO
    :returns: TODO

    """
    


def optimizers(arg1):
    """TODO: Docstring for optimizers.

    :function: TODO
    :returns: TODO

    """
    


def run(args,config):
    """TODO: Docstring for run.

    :a: TODO
    :returns: TODO

    """
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('architecture', type=str)

    args = parser.parse_args()

    if args.horovod:
        hvd.init()

    config = '' 
    run(args,config)
