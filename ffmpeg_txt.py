"""
ffmpeg_txt.py

This script generates a metadata text file called `create_video_metadata.txt`
for creating videos from PNG images using ffmpeg.

CLI Args:
    --fdir: Top-level source directory containing subdirectories with PNG images.
    --frame_rate: Reciprocal of frame rate (default is 0.5, which is 2 frames per second).
    --skip: Names of the directories to skip (default is None, meaning no directories are skipped).

Example:
    python ffmpeg_txt.py --fdir /path/to/source --frame_rate 0.5 --skip dir1 dir2

"""
import os
import datetime
from argparse import ArgumentParser


def sort_filenames_by_datetime(filenames):
    """
    Sort filenames in chronological order.

    Expected filename format:
        YYYYMMDD_HHMMZ.png

    Example:
        20240730_0815Z.png
        20240730_1620Z.png
        20240731_0930Z.png

    Args:
        filenames: List of PNG filenames.

    Returns:
        List of filenames sorted chronologically.
    """
    return sorted(filenames)


if __name__ == "__main__":

    START_TIME = datetime.datetime.now()

    parser = ArgumentParser(prog='ffmpeg_txt')
    parser.add_argument(
        '--fdir',
        type=str,
        metavar='',
        help='Top-level source directory.\n'
    )
    parser.add_argument(
        '--frame_rate',
        type=float,
        metavar='',
        default=0.5,
        help='Reciprocal of frame rate i.e., --frame_rate=0.5 is 2 frames per second.\n'
    )
    parser.add_argument(
        '--skip',
        nargs='+',
        type=str,
        metavar='',
        default=None,
        help='Names of the directories to skip. By default, no directories are skipped.'
    )

    args = parser.parse_args()

    if args.skip is None:
        skip_dirs = []
    else:
        skip_dirs = args.skip

    # Sort sub-directories by name
    subs = sorted([
        f for f in os.listdir(args.fdir)
        if os.path.isdir(os.path.join(args.fdir, f))
    ])

    # Make videos one by one
    for sub in subs:

        if sub in skip_dirs:
            print("Message [ffmpeg_txt]: Skipping {}...".format(sub))
            continue

        outpath = os.path.join(
            args.fdir,
            sub,
            'create_video_metadata.txt'
        )

        if os.path.isfile(outpath):
            print(
                "File {} already exists...deleting before creating new file"
                .format(outpath)
            )
            os.remove(outpath)

        # Get PNG filenames
        fpngs = [
            png
            for png in os.listdir(os.path.join(args.fdir, sub))
            if png.endswith('.png')
        ]

        # Sort PNGs chronologically based on YYYYMMDD_HHMMZ filename format
        fpngs = sort_filenames_by_datetime(fpngs)

        with open(outpath, "w") as f:
            for i in range(len(fpngs)):
                f.write("file '{}'\n".format(fpngs[i]))
                f.write("duration {}\n".format(args.frame_rate))

    print("Finished creating video metadata file.\n")

    END_TIME = datetime.datetime.now()

    print(
        'Time taken to execute {}: {}'
        .format(os.path.basename(__file__), END_TIME - START_TIME)
    )
