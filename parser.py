import argparse


parser  = argparse.ArgumentParser()
parser.add_argument('--video_name', type=str)
args = parser.parse_args()
print("video", args.video_name)