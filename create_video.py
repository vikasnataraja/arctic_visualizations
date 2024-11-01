import os
import sys
import datetime
import subprocess
from argparse import ArgumentParser


if __name__ == "__main__":

    START_TIME = datetime.datetime.now()
    parser = ArgumentParser(prog='create_video')
    parser.add_argument('--fdir', type=str, metavar='',
                        help='Top-level source directory\n')
    args = parser.parse_args()

    subs = sorted([f for f in os.listdir(args.fdir) if os.path.isdir(os.path.join(args.fdir, f))])

    for sub in subs:

        mp4name = os.path.join(args.fdir, sub) + '.mp4'

        if os.path.isfile(mp4name): # if it already exists then delete it
            print("Message [create_video]: File {} already exists...deleting before creating new file".format(mp4name))
            os.remove(mp4name)

        meta_file = os.path.join(args.fdir, sub, 'create_video_metadata.txt')
        if not os.path.isfile(meta_file):
            print('"Message [create_video]: Metadata file not found for {}, therefore this directory will be skipped.'.format(os.path.join(args.fdir, sub)))
            continue

        # command = "ffmpeg -f concat -i {} -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" -pix_fmt yuv420p {}".format(os.path.join(args.fdir, sub, 'create_video_metadata.txt'), mp4name)

        command = ["ffmpeg", "-f", "concat", "-i", "{}".format(meta_file), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p", "{}".format(mp4name)]

        # os.system(command)
        try:
            ret = subprocess.run(command, capture_output=True, check=False)

        except Exception as err:
            print("Error [create_video]: ", err)

    print("Finished creating video files in {}.\n".format(args.fdir))
    END_TIME = datetime.datetime.now()
    print('Time taken to execute {}: {}'.format(os.path.basename(__file__), END_TIME - START_TIME))
