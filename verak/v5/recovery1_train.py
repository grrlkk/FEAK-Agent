"""Bounded Korean-only continuation using the established two-GPU trainer."""
import argparse
from .recovery1 import ROOT, verify_data, prepare
from verak.v4.fix1_train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    design = prepare()
    train('korean', resume=args.resume, experiment={
        'root': str(ROOT), 'phase': 'v5_recovery1_observed_rejection_continuation',
        'learning_rate': design['recipe']['learning_rate'], 'verify': verify_data})
