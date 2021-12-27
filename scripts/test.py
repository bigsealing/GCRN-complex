import argparse
import pprint

import torch

from utils.models import Model
from utils.utils import getLogger
import yaml

logger = getLogger(__name__)


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    # parse the configuarations
    parser = argparse.ArgumentParser(description='Additioal configurations for testing',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu_ids',
                        type=str,
                        default='-1',
                        help='IDs of GPUs to use (please use `,` to split multiple IDs); -1 means CPU only')
    parser.add_argument('--tt_list',
                        type=str,
                        required=False,
                        help='Path to the list of testing files')
    parser.add_argument('--ckpt_dir',
                        type=str,
                        required=False,
                        help='Name of the directory to write log')
    parser.add_argument('--model_file',
                        type=str,
                        required=False,
                        help='Path to the model file')
    parser.add_argument('--est_path',
                        type=str,
                        default='../data/estimates',
                        help='Path to dump estimates')
    parser.add_argument('--write_ideal',
                        default=False,
                        action='store_true',
                        help='Whether to write ideal signals (the speech signals resynthesized from the ideal training targets; ex. for time-domain enhancement, it is the same as clean speech)')

    args = parser.parse_args()
    args = vars(args)

    if args['ckpt_dir'] ==None:
        config_file = "../config/predict.yaml"
        with open(config_file, "r") as ymlfile:
            args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
            for key,value in args.items():
                if key in args_yaml.keys():
                    args[key]=args_yaml[key]
            for key in args_yaml.keys():
                if key not in args.keys():
                    args[key]=args_yaml[key]


    logger.info('Arguments in command:\n{}'.format(pprint.pformat((args))))

    model = Model()
    model.test(args)
    

if __name__ == '__main__':
    main()
