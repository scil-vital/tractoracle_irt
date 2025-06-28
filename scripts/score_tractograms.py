import argparse
import os
import tempfile
from tractoracle_irt.filterers.tractometer.tractometer_filterer import TractometerFilterer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Score tractograms')
    parser.add_argument('tractogram', type=str,
                        help='Glob pattern to the tractograms to score.', nargs='+')
    parser.add_argument('scoring_data_dir', type=str,
                        help='Path to the directory containing the scoring data.')
    parser.add_argument('--reference', type=str, required=True,
                        help='Reference file to load the tractograms when needed.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output path to save the tractograms.\n'
                        'If not specified, the tractograms will be saved in the same directory.')
    parser.add_argument('--invalid_score', type=int, default=0,
                        help='Score to assign to invalid streamlines.')
    parser.add_argument('--no_bbox_valid_check', action='store_true',
                        help='Disable the bounding box check when loading the tractograms.')

    return parser.parse_args()


def filter_tractograms(tractograms,
                       reference,
                       scoring_data_dir,
                       output_dir,
                       invalid_score=0,
                       bbox_valid_check=True):
    filterer = TractometerFilterer(
        scoring_data_dir, reference, invalid_score=invalid_score, bbox_valid_check=bbox_valid_check)
    for tractogram in tractograms:
        print(f"Scoring {tractogram}", end='...')
        filterer(tractogram, output_dir)
        print("Done and saved: ", output_dir)


def main():
    args = parse_args()

    # Make sure files exist
    assert os.path.exists(
        args.scoring_data_dir), f"Scoring data directory {args.scoring_data_dir} does not exist."
    assert os.path.exists(
        args.reference), f"Reference {args.reference} does not exist."

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    filter_tractograms(args.tractogram, args.reference,
                       args.scoring_data_dir, args.output_dir,
                       args.invalid_score,
                       not args.no_bbox_valid_check)


if __name__ == '__main__':
    main()
