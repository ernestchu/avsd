import csv
import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import normalize
from tqdm import tqdm
from accelerate import Accelerator

from facexlib.detection import init_detection_model
from facexlib.headpose import init_headpose_model
from facexlib.tracking.sort import SORT
from facexlib.utils.misc import img2tensor
from facexlib.visualization.vis_headpose import draw_axis

class ListDataset(Dataset):
    def __init__(self, list_):
        self.list = list_

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]

def collate_fn(batch):
    return batch[0]


def visualize(colors, trackers, frame, face_list):
    for d in trackers:
        score = d[5]
        d = d.astype(np.int32)
        cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), colors[d[4] % 32, :] * 255, 3)
        if len(face_list) != 0:
            # bbox, id, conf
            cv2.putText(frame, 'ID : %d  DETECT' % (d[4]) + f' {score:.2f}',
                        (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, colors[d[4] % 32, :] * 255, 2)
            cv2.putText(frame, 'DETECTOR', (5, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (1, 1, 1), 2)

            # landmark
            cv2.circle(frame, (d[6], d[7]), 1, (0, 0, 255), 4)
            cv2.circle(frame, (d[8], d[9]), 1, (0, 255, 255), 4)
            cv2.circle(frame, (d[10], d[11]), 1, (255, 0, 255), 4)
            cv2.circle(frame, (d[12], d[13]), 1, (0, 255, 0), 4)
            cv2.circle(frame, (d[14], d[15]), 1, (255, 0, 0), 4)

            # headpose
            draw_axis(frame, d[16], d[17], d[18], tdx=d[10], tdy=d[11], size=(d[2]-d[0]) / 2)
        else:
            cv2.putText(frame, 'ID : %d' % (d[4]) + f' {score:.2f}',
                        (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        colors[d[4] % 32, :] * 255, 2)
    return frame

@torch.no_grad()
def main(args):
    csv_header = ['frame_id', 'x1', 'y1', 'x2', 'y2', 'bbox_id', 'confidence',
                  'lm_x1', 'lm_y1', 'lm_x2', 'lm_y2', 'lm_x3', 'lm_y3', 'lm_x4', 'lm_y4',
                  'lm_x5', 'lm_y5', 'yaw', 'pitch', 'roll']

    accelerator = Accelerator()

    detect_interval = args.detect_interval
    margin = args.margin
    face_score_threshold = args.face_score_threshold

    save_frame = args.save_frame
    colors = np.random.rand(32, 3)

    # init detection model
    with accelerator.main_process_first():
        det_net = init_detection_model('retinaface_resnet50', half=False, device=accelerator.device)
        headpose_net = init_headpose_model('hopenet', half=False, device=accelerator.device)
    # print('Start track...')

    # track over all frames
    video_paths = glob.glob(os.path.join(args.input_folder, '*.mp4'))
    video_paths = accelerator.prepare(DataLoader(ListDataset(video_paths), collate_fn=collate_fn))
    global_pbar = tqdm(video_paths, disable=not accelerator.is_local_main_process)
    for video_path in global_pbar:
        global_pbar.set_description(f'Processing: ...{video_path[-10:]}')
        csv_path = os.path.join(args.save_folder,
                                os.path.basename(video_path)[:-4] + '.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)

        if save_frame:
            preview_save_dir = os.path.join(args.save_folder, 'preview',
                                    os.path.basename(video_path)[:-4])
            os.makedirs(preview_save_dir)

        tracker = SORT(max_age=1, min_hits=1, iou_threshold=0.2)
        cap = cv2.VideoCapture(video_path)
        pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    unit='frames', desc='Extract', leave=False,
                    disable=not accelerator.is_local_main_process)
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_size = frame.shape[0:2]

            # detection face bboxes
            bboxes = det_net.detect_faces(frame, face_score_threshold)

            additional_attr = []
            face_list = []

            for idx_bb, bbox in enumerate(bboxes):
                attr = bbox[4:]
                bbox = bbox[0:5]
                det = bbox[0:4]

                # face rectangle
                det[0] = np.maximum(det[0] - margin, 0)
                det[1] = np.maximum(det[1] - margin, 0)
                det[2] = np.minimum(det[2] + margin, img_size[1])
                det[3] = np.minimum(det[3] + margin, img_size[0])

                # crop the face pixels
                qt_det = det.astype(np.int32)
                det_face =  frame[qt_det[1]:qt_det[3], qt_det[0]:qt_det[2], :].astype(np.float32) / 255.
                det_face = cv2.resize(det_face, (224, 224), interpolation=cv2.INTER_LINEAR)
                det_face = img2tensor(np.copy(det_face), bgr2rgb=False)
                normalize(det_face, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)
                det_face = det_face.unsqueeze(0).to(accelerator.device)

                # headpose prediction
                pose = headpose_net(det_face)
                attr = np.concatenate((attr, [p.item() for p in pose]))

                face_list.append(bbox)
                additional_attr.append(attr)

            trackers = tracker.update(np.array(face_list), img_size, additional_attr, detect_interval)

            # save result
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                for d in trackers:
                    # len(d) = 4 + 1 + 1 + 10 + 3
                    # bbox(4), id(1), conf(1), landmark(10), headpose(3)
                    writer.writerow([idx] + d.round(decimals=args.save_precision).tolist())


            # save frame
            if save_frame:
                frame= visualize(colors, trackers, frame, face_list)
                save_path = os.path.join(preview_save_dir, f'{idx:04}.png')
                cv2.imwrite(save_path, frame)

            pbar.update(1)
            pbar.set_description(f'frame {idx}: {len(bboxes)} faces detected')
            idx += 1
            # end of one frame

        cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='Path to the input folder', type=str)
    parser.add_argument('--save_folder', help='Path to save visualized frames', type=str, default=None)

    parser.add_argument(
        '--detect_interval',
        help=('how many frames to make a detection, trade-off '
              'between performance and fluency'),
        type=int,
        default=1)
    # if the face is big in your video ,you can set it bigger for easy tracking
    parser.add_argument('--margin', help='add margin for face', type=int, default=20)
    parser.add_argument('--save_precision',
        help='number of digits after decimal point to keep when saving result', type=int, default=5)
    parser.add_argument('--save_frame', help='save results into images for validation', action='store_true')
    parser.add_argument(
        '--face_score_threshold',
        help='The threshold of the extracted faces,range 0 < x <=1', type=float, default=0.7)

    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    main(args)

    # add verification
    # remove last few frames
