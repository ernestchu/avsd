import csv
import os

import numpy as np
import cv2
from facexlib.visualization.vis_headpose import draw_axis
from tqdm.auto import tqdm

max_frame = 200 # number of processing frames, set to `None` to process the entire video
fname = '02de9500-d574-446e-95fc-3e82c367385a'
save_dir = 'output'
os.makedirs(save_dir, exist_ok=True)

labels = {}
with open(f'{fname}.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # skip the headers
    for attr in reader:
        frame_id = int(attr[0])
        attr = np.array([float(a) for a in attr[1:]]) # exclude frame_id
        if frame_id in labels:
            labels[frame_id].append(attr)
        else:
            labels[frame_id] = [attr]

        
colors = np.random.rand(32, 3)
cap = cv2.VideoCapture(f'{fname}.mp4')
frame_id = 0
pbar = tqdm(total=max_frame or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            unit='frames', desc='Annotating')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break # end of the video
    if max_frame and frame_id >= max_frame:
        break

    if frame_id not in labels:
        # no detected face
        cv2.imwrite(os.path.join(save_dir, f'{frame_id:04}.png'), frame)
        frame_id += 1
        pbar.update(1)
        continue

    attrs = labels[frame_id]
    for attr in attrs:
        d = attr
        score = d[5]
        d = d.astype(np.int32)
        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colors[d[4] % 32, :] * 255, 3)
        # bbox, id, conf
        cv2.putText(frame, f'ID {d[4]:d} CONF {score:.2f}',
                    (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, colors[d[4] % 32, :] * 255, 2)
        # cv2.putText(frame, 'DETECTOR', (5, 45),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.75, (1, 1, 1), 2)

        # landmark
        cv2.circle(frame, (d[6], d[7]), 1, (0, 0, 255), 4)
        cv2.circle(frame, (d[8], d[9]), 1, (0, 255, 255), 4)
        cv2.circle(frame, (d[10], d[11]), 1, (255, 0, 255), 4)
        cv2.circle(frame, (d[12], d[13]), 1, (0, 255, 0), 4)
        cv2.circle(frame, (d[14], d[15]), 1, (255, 0, 0), 4)

        # headpose
        draw_axis(frame, d[16], d[17], d[18], tdx=d[10], tdy=d[11], size=(d[2]-d[0]) / 2)

    cv2.imwrite(os.path.join(save_dir, f'{frame_id:04}.png'), frame)
    frame_id += 1
    pbar.update(1)

cap.release()

