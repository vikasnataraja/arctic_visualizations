import os
import datetime
import subprocess
from argparse import ArgumentParser
from joblib import Parallel, delayed


def process_subdirectory(sub, fdir):
    """
    Processes a single subdirectory to create an MP4 video file.

    Args:
        sub (str): The name of the subdirectory.
        fdir (str): The top-level source directory.
    """
    mp4name = os.path.join(fdir, sub) + '.mp4'

    if os.path.isfile(mp4name):  # if it already exists then delete it
        print(f"Message [create_video]: File {mp4name} already exists...deleting before creating new file")
        os.remove(mp4name)

    meta_file = os.path.join(fdir, sub, 'create_video_metadata.txt')
    if not os.path.isfile(meta_file):
        print(f'"Message [create_video]: Metadata file not found for {os.path.join(fdir, sub)}, therefore this directory will be skipped.')
        return

    command = ["ffmpeg", "-f", "concat", "-i", meta_file, "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", mp4name]

    try:
        print(f"Creating video for {sub}...")
        ret = subprocess.run(command, capture_output=True, check=True, text=True)
        if ret.returncode != 0:
            print(f"Error creating video for {sub}:\n{ret.stderr}")

    except subprocess.CalledProcessError as e:
        print(f"Error [create_video] processing {sub}: {e}\n{e.stderr}")
    except Exception as err:
        print(f"An unexpected error occurred processing {sub}: {err}")


def create_videos(fdir, skip=None, parallel=False):
    """
    Generates MP4 video files from metadata files in subdirectories.

    Args:
        fdir (str): Top-level source directory.
        skip (list, optional): List of directory names to skip. Defaults to None.
        parallel (bool, optional): If True, run the video creation in parallel. Defaults to False.
    """
    if skip is None:
        skip_dirs = []
    else:
        skip_dirs = skip

    subs = sorted([f for f in os.listdir(fdir) if os.path.isdir(os.path.join(fdir, f))])

    # Filter out skipped directories
    subs_to_process = [s for s in subs if s not in skip_dirs]
    for sub in skip_dirs:
        if sub in subs:
            print(f"Message [create_video]: Skipping {sub}...")

    max_cores = min([8, os.cpu_count(), len(subs_to_process)])  # Limit to 8 cores max
    if parallel: # run in parallel mode but limit to 8 cores due to overhead
        print("Message [create_video]: Running in parallel mode.")
        Parallel(n_jobs=max_cores)(delayed(process_subdirectory)(sub, fdir) for sub in subs_to_process)

    else:
        print("Message [create_video]: Running in sequential mode.")
        for sub in subs_to_process:
            process_subdirectory(sub, fdir)


if __name__ == "__main__":

    START_TIME = datetime.datetime.now()
    parser = ArgumentParser(prog='create_video')
    parser.add_argument('--fdir', type=str, required=True, metavar='',
                        help='Top-level source directory\n')
    parser.add_argument('--skip', nargs='+', type=str, metavar='', default=None,
                        help='Names of the directories to skip. By default, no directories are skipped.')
    parser.add_argument('--parallel', action='store_true',
                        help='Run video creation in parallel')
    args = parser.parse_args()

    create_videos(args.fdir, args.skip, args.parallel)

    print("Finished creating video files in {}.\n".format(args.fdir))
    END_TIME = datetime.datetime.now()
    print('Time taken to execute {}: {}'.format(os.path.basename(__file__), END_TIME - START_TIME))
