import os
import datetime
from argparse import ArgumentParser


no_video_dirs = []
if __name__ == "__main__":

    START_TIME = datetime.datetime.now()
    parser = ArgumentParser(prog='ffmpeg_txt')
    parser.add_argument('--fdir', type=str, metavar='',
                        help='Top-level source directory\n')
    parser.add_argument('--frame_rate', type=float, metavar='', default=0.5,
                        help='Reciprocal of frame rate i.e., --frame rate=0.5 is 2 frames per second.\n')
    args = parser.parse_args()

    # sort sub-directories by date
    subs = sorted([f for f in os.listdir(args.fdir) if os.path.isdir(os.path.join(args.fdir, f))])

    # make videos one by one
    for sub in subs:
        if sub in no_video_dirs:
            print("Message [ffmpeg_txt]: Skipping {}...".format(sub))
            continue

        outpath = os.path.join(args.fdir, sub, 'create_video_metadata.txt')
        if os.path.isfile(outpath): # if it already exists then delete it
            print("File {} already exists...deleting before creating new file".format(outpath))
            os.remove(outpath)

        fpngs = [png for png in os.listdir(os.path.join(args.fdir, sub)) if png.endswith('.png')]
        fpngs = sorted(fpngs, key=lambda x: x.split('_')[1]) # sorted by time

        with open(outpath, "w") as f:
            for i in range(len(fpngs)):
                f.write("file '{}'\n".format(fpngs[i]))
                f.write("duration {}\n".format(args.frame_rate))

    print("Finished creating video metadata file.\n")
    END_TIME = datetime.datetime.now()
    print('Time taken to execute {}: {}'.format(os.path.basename(__file__), END_TIME - START_TIME))
