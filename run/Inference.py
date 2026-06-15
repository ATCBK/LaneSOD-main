import argparse
import os
import sys

import tqdm

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.utils import load_config
from webapp.inference_service import LaneWebPredictor


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/HighwayLane.yaml')
    parser.add_argument('--source', type=str)
    parser.add_argument('--dest', type=str, default=None)
    parser.add_argument('--mask', type=str, default='data/mask.png')
    parser.add_argument('--type', type=str, choices=['rgba', 'map', 'overlay'], default='map')
    parser.add_argument('--verbose', action='store_true', default=False)
    return parser.parse_args()


def inference(opt, args):
    if not os.path.exists(args.source):
        print(f"Error: source path '{args.source}' does not exist.")
        return

    if os.path.isdir(args.source):
        source_dir = args.source
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        source_list = [f for f in os.listdir(args.source) if f.lower().endswith(valid_extensions)]
        if not source_list:
            print(f"No image files found in {args.source}")
            return
        save_dir = os.path.join('results', args.source.split(os.sep)[-1])
    elif os.path.isfile(args.source):
        source_dir = os.path.split(args.source)[0]
        source_list = [os.path.split(args.source)[1]]
        save_dir = 'results'
    else:
        print(f"Error: source '{args.source}' is neither a file nor a directory.")
        return

    if args.dest is not None:
        save_dir = args.dest

    os.makedirs(save_dir, exist_ok=True)

    predictor = LaneWebPredictor(
        upload_dir='.uploads',
        result_dir=save_dir,
        config_path=args.config,
        mask_path=args.mask,
        public_result_prefix='',
        output_type=args.type,
    )

    sources = enumerate(source_list)
    if args.verbose:
        sources = tqdm.tqdm(
            sources,
            desc='Inference',
            total=len(source_list),
            position=1,
            leave=False,
            bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}',
        )

    for _, source in sources:
        img_path = os.path.join(source_dir, source)
        save_path = os.path.join(save_dir, os.path.splitext(source)[0] + '.png')
        try:
            predictor.predict_path(img_path, save_path)
            if not args.verbose:
                print(f"Saved {save_path}")
        except Exception as exc:
            print(f"Skipping {img_path}: {exc}")


if __name__ == "__main__":
    args = _args()
    opt = load_config(args.config)
    inference(opt, args)
