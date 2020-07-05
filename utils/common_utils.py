import cv2
import os
import sys
import numpy as np
import torch

def id_generator(number, id_len=4):
    number = str(number)
    assert len(number) < id_len

    return "0" * (id_len - len(number)) + number

def progressbar(current, total, prefix="", gap=20):
    sys.stdout.write("\r{}:{}/{} |{}| {:.4f}%".format(prefix, current, total,
                                                      "#" * (int(gap * current / total)) + " " * (
                                                          int(gap * (1 - current / total))), 100 * current / total))
    sys.stdout.flush()
    if current == total:
        print("")


def str2seconds(time):
    try:

        h, m, s = time.split(":")[0], time.split(":")[1], time.split(":")[2]
        h, m, s = int(h), int(m), int(s)
        assert h >= 0
        assert 0 <= m < 60
        assert 0 <= s < 60
        seconds = h * 3600 + m * 60 + s
        return int(seconds)
    except:
        print("时间格式错误！")
        sys.exit(0)


def extacted_videos(video_path, save_path=None, time_start="default", time_end="default"):
    start_frame, end_frame = 0, 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(5)
    frame_nums = int(cap.get(7))
    total_seconds = int(frame_nums / fps)

    if time_start == "default":
        start_frame = 0
    else:
        start_frame = int(frame_nums * (str2seconds(time_start) / total_seconds))
    if time_end == "default":
        end_frame = frame_nums
    else:
        tmp = int(frame_nums * (str2seconds(time_end) / total_seconds))
        if tmp > frame_nums:
            end_frame = frame_nums
        else:
            end_frame = tmp

    assert start_frame <= end_frame

    iters = end_frame - start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    all_frames = []
    for count in range(iters):
        ret, frame = cap.read()
        if frame is None:
            pass
        if not ret:
            break

        if save_path is not None:
            cv2.imwrite(os.path.join(save_path, "{}.png".format(id_generator(count))), frame)
        all_frames.append(frame)
        progressbar(count + 1, iters, "extract {}".format(video_path))
    return all_frames, fps

def frames2video(frames, fps, save_path):
    base_folder = os.path.split(save_path)[0]
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    H, W = frames[0].shape[0:2]
    img_size = (W, H)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, img_size)

    num = 0
    for frame in frames:
        video_writer.write(frame)
        num += 1
        progressbar(num, len(frames), prefix="write video")

    video_writer.release()


def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()

def crop(image, center, scale, resolution=256.0):
    """Center crops an image or set of heatmaps

    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face

    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})

    Returns:
        [type] -- [description]
    """  # Crop around the center point
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    # pad = math.ceil(torch.norm((ul - br).float()) / 2.0 - (br[0] - ul[0]) / 2.0)
    if image.ndim > 2:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0],
                           image.shape[2]], dtype=np.int32)
        newImg = np.zeros(newDim, dtype=np.uint8)
    else:
        newDim = np.array([br[1] - ul[1], br[0] - ul[0]], dtype=np.int)
        newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32)
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32)
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32)
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32)
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
                        interpolation=cv2.INTER_LINEAR)
    return newImg