"""Load authorized existing runtime credentials without copying them to the worktree."""
from dotenv import load_dotenv
from ..train.teacher_bulk import load_environment as base_environment


def load_environment(config):
    load_dotenv(config['paths']['repo'] / '.env', override=False)
    base_environment(config)
