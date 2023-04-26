# Audio-visual Speaker Diarization

## Face tracking

### Environment setup
```sh
conda env create -f environment.yml
```

Install the modified version of facexlib by
```sh
cd facexlib
pip install -e .
```

### Visualizing track results
See [viz_demo](./viz_demo) directory.


### Tracking

You can download the track results [here](https://github.com/ernestchu/avsd/releases/download/v0.0.1/attributes.tar.gz) (1 csv per video),
or you can try to run the script yourself

Run the inference script by
```sh
python ego4d_face_tracking.py --input_folder /scratch2/users/dataset/EGO4D/v1/clips/ --save_folder test
```

#### Optional: use HF Accelerate to do multi-GPU inference

Setup [HF Accelerate](https://huggingface.co/docs/accelerate/index)
to leverage multiple GPUs
```sh
accelerate config
```

Run the face tracking script by
```sh
accelerate launch ego4d_face_tracking.py --input_folder /scratch2/users/dataset/EGO4D/v1/clips/ --save_folder test
```

### Issues
- video clips are compressed with VP9-crf = 41 (approx. H264-crf = 28)
    - motion blur
- wide-angle camera (fisheye)
- out-of-frame tracking (set merging)
    - recognition: issues with face(front)/hair(back)

