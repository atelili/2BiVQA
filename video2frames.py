import cv2
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--video_dir',  type=str, help='path to the input video directory')
parser.add_argument('--nbr_frame', type=int, help='number of frames to be extracted')
args = parser.parse_args()
import sys



	
def TemporalCrop(input_video_path):
	out = []

	cap = cv2.VideoCapture(input_video_path)
	N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	fps = int(cap.get(cv2.CAP_PROP_FPS))

	
	while(cap.isOpened()):
		
			ret, frame = cap.read()
			if ret:

				out.append(frame)
			else:
				break
	return(out, N, fps)

if not os.path.exists('./frames'):
	os.makedirs('./frames')

all_videos = os.listdir(args.video_dir)
for v in tqdm(all_videos):
	
	name = v.split('.')[0] 
	i = 0
	j = 0
	out, N, fps = TemporalCrop(os.path.join(args.video_dir, v))
	nb = args.nbr_frame
	if nb>N:
		print('Error! number of frames to be extracted > number of frames ')
		sys.exit()
	step = int(N/nb)
	nb = args.nbr_frame


	while i < nb :
		if i < 10:
			img = out[j]
			j = j +step
			filename = 'frames/' + name +'0'+ str(i) + '.png'
			cv2.imwrite(filename, img)
			i = i+1
		else : 
			img = out[j]
			j = j +step
			filename = 'frames/' + name + str(i) + '.png'
			cv2.imwrite(filename, img)
			i = i+1
