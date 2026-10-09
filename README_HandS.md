# IPN HandS: hand landmarks and refined temporal annotations for IPN Hand

This page describes the data released for **IPN HandS** ([Applied Sciences 2025](https://doi.org/10.3390/app15116321)), built on top of the 200 videos of the [IPN Hand](README.md) dataset:

- **Manually corrected 2D hand landmarks** (21 keypoints per hand) for every frame of the 200 videos.
- **Refined temporal annotations** with the IPN HandS taxonomy (14 gestures + non-gesture).
- **MediaPipe predictions** (Holistic and Hands) for the same frames, as off-the-shelf baselines.

## Download

The files are shared on Google Drive: **[download link](https://drive.google.com/drive/folders/1eILZNuPRM53_Ige2jxGqXV3OdIeoMeIb?usp=sharing)**

| File | Content | Size |
|---|---|---|
| `IPN_HandS_annotations.zip` | Temporal annotations, class list, train/test lists, frame counts, frame extraction script | [<1 MB](https://drive.google.com/file/d/1HybcqZqnNKZGgJJXCMjj6tLFbpsNP_a1/view?usp=drive_link) |
| `IPN_HandS_landmarks_GT.zip` | Manually corrected hand landmarks (ground truth) | [674 MB](https://drive.google.com/file/d/1pbCWxfzGZyexHe89lZtGgnHWfa958XJv/view?usp=drive_link) |
| `IPN_HandS_landmarks_MPHolistic.zip` | MediaPipe Holistic predictions (hands + body pose) | [1.1 GB](https://drive.google.com/file/d/1hdtmr4JpK14rt1zMcbN39-oTedn2sBx1/view?usp=drive_link) |
| `IPN_HandS_landmarks_MPHands.zip` | MediaPipe Hands predictions (hands only) | [375 MB](https://drive.google.com/file/d/1Y-3_MUHJ6OYpv-BNof5vYDRR4YNgrGQ2/view?usp=drive_link) |

SHA-256 checksums are listed [below](#checksums).

## Frames: read this first

All IPN HandS files are indexed on frames extracted from the **original IPN Hand videos (640x480)** with:

```bash
ffmpeg -i <video>.avi -vf fps=30 -q:v 2 <video>_%06d.jpg
```

`extract_frames.sh` (included in `IPN_HandS_annotations.zip`) runs this for a whole folder of videos. 
Frame indices are **1-based** (`<video>_000001.jpg` is the first frame).

These frames are **not** the 320x240 frames distributed with the original IPN Hand release. 
Those frames had near-duplicate frames removed due to frame rate <30 fps, so their indices drift with respect to the videos: the original `Annot_List.txt` and the IPN HandS annotations are on different timelines, and the two must not be mixed. `frame_counts.csv` lists, for every video, the number of frames of the IPN HandS timeline and of the original release.

## Landmarks

One JSON file per video, with the same schema in the three landmark folders (`landmarks_gt/`, `landmarks_mp_holistic/` and `landmarks_mp_hands/`):

```json
{
  "1CM1_1_R_#217_000301.jpg": {
    "pose_landmarks": {"0": [0.5602, 0.3883, -1.1892], "...": "..."},
    "hand_landmarks": [
      {"handedness": "Right",
       "landmarks": {"0": [0.4267, 0.7895, 0.0000], "...": "...", "20": [0.4598, 0.7349, -0.0723]}}
    ]
  }
}
```

- **Keys** are frame file names (`<video>_<6-digit 1-based index>.jpg`). Every frame of every video is present.
- **Coordinates** are `[x, y, z]`. `x` and `y` are normalized by the image size (multiply by 640 and 480 for pixels). Keypoints outside the image are kept and can be `< 0` or `> 1`. `z` is MediaPipe's relative depth and was **not** manually verified.
- **`hand_landmarks`** is a list with 0, 1 or 2 hands. Each hand has a `handedness` label (`"Right"` or `"Left"`, the subject's anatomical hand) and the 21 keypoints `"0"`–`"20"` in the [MediaPipe hand order](https://developers.google.com/edge/mediapipe/images/solutions/hand-landmarks.png).
- **`pose_landmarks`** depends on the source:
  - `landmarks_gt`: 12 upper-body points, MediaPipe Pose indices `0, 2, 5, 9, 10, 11, 12, 13, 14, 23, 24` plus `"33"` = midpoint of the shoulders (`11`, `12`). Auxiliary, **not** manually corrected.
  - `landmarks_mp_holistic`: the 33 MediaPipe Pose points (`"0"`–`"32"`), or `null` when no body was detected.
  - `landmarks_mp_hands`: not included.

The ground truth was initialized with MediaPipe and manually corrected ([see the paper](https://doi.org/10.3390/app15116321) for the annotation tool and protocol).
The MediaPipe files are the raw predictions and keep their usual errors (missed hands, swapped or duplicated handedness labels, etc.).

## Temporal annotations

`temporal_annotations/<video>.txt` has one segment per line:

```
class_id, start_frame, end_frame
```

Frames are 1-based and inclusive, on the IPN HandS timeline.
The segments of each video are contiguous and cover all its frames.

| id | Code | Gesture | id | Code | Gesture |
|---|---|---|---|---|---|
| 0 | NOG | Non-gesture | 8 | TRT | Throw right |
| 1 | P1F | Pointing with one finger | 9 | OP2 | Open twice |
| 2 | P2F | Pointing with two fingers | 10 | 2C1 | Double click with one finger |
| 3 | C1F | Click with one finger | 11 | 2C2 | Double click with two fingers |
| 4 | C2F | Click with two fingers | 12 | ZIN | Zoom in |
| 5 | TUP | Throw up | 13 | ZOT | Zoom out |
| 6 | TDN | Throw down | **14** | **GRB** | **Grab** |
| 7 | TLT | Throw left | | | |

The same list is in `classes.txt`. GRB is new in IPN HandS, and some segments were relabeled with respect to the original IPN Hand annotations; [see the paper](https://doi.org/10.3390/app15116321) for details.

## Train/test split

`splits/train_list.txt` (148 videos) and `splits/test_list.txt` (52 videos) follow the official [IPN Hand split](https://gibranbenitez.github.io/IPN_Hand/).

## Checksums

```
4686a2ebf70a6036166c261bc6992d050cfba08c3b18edcdd460b3fa9f30ec71  IPN_HandS_annotations.zip
493bf53638d5b1477e28f05c41884f0cabdcb7b84f20d68f3d5d94e0b46fad4d  IPN_HandS_landmarks_GT.zip
e511bb2b902207b82a296190daadc30a057f3abc16f628579542d2b1274ce598  IPN_HandS_landmarks_MPHands.zip
4746db8afa7376ae647c4c61c58cba963aa22fbc3458edbc06fc88775ab972b8  IPN_HandS_landmarks_MPHolistic.zip
```

## License

The IPN HandS annotations and landmarks are licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode), like the IPN Hand dataset. 
The MediaPipe predictions were produced with [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache 2.0).

## Citation

If you use these data, please cite both papers:

```bibtex
@article{bega2025IPNhands,
  title={IPN HandS: Efficient Annotation Tool and Dataset for Skeleton-Based Hand Gesture Recognition},
  author={Benitez-Garcia, Gibran and Olivares-Mercado, Jesus and Sanchez-Perez, Gabriel and Takahashi, Hiroki},
  journal={Applied Sciences},
  volume={15},
  number={11},
  pages={6321},
  year={2025},
  doi={10.3390/app15116321}
}

@inproceedings{bega2020IPNhand,
  title={IPN Hand: A Video Dataset and Benchmark for Real-Time Continuous Hand Gesture Recognition},
  author={Benitez-Garcia, Gibran and Olivares-Mercado, Jesus and Sanchez-Perez, Gabriel and Yanai, Keiji},
  booktitle={25th International Conference on Pattern Recognition, {ICPR 2020}, Milan, Italy, Jan 10--15, 2021},
  pages={1--8},
  year={2021},
  organization={IEEE},
}
```
