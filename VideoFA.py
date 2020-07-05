import warnings

warnings.filterwarnings("ignore")
import glob
from FA.FA import *
from utils.time_smooth import *
from utils.common_utils import progressbar

MODEL_FAN2D = 0
MODEL_FAN3D = 4
MODEL_ADAPT = 1
MODEL_3DDFA = 2
MODEL_PRNET = 3


class VideoFA():
    def __init__(self, detector=MODEL_3DDFA, use_smooth=False, smooth_mode=SMOOTH_METHODS_GAUSSIAN):
        self.detector = detector
        if self.detector == MODEL_FAN2D:
            self.model = BaseFaceAlignment("2D")
        if self.detector == MODEL_FAN3D:
            self.model = BaseFaceAlignment("3D")
        if self.detector == MODEL_3DDFA:
            self.model = DDFAFaceAlignment()
        if self.detector == MODEL_ADAPT:
            self.model = AdaptiveWLFaceAlignment()
        if self.detector == MODEL_PRNET:
            self.model = PRNetFaceAlignment()

        self.use_smooth = use_smooth
        self.smooth_mode = smooth_mode

        self.smooth_window = 5
        self.gaussian_miu = 0
        self.gaussian_sigma = 1

        self.names = {
            0: "FAN2D",
            4: "FAN3D",
            1: "ADAPT",
            2: "3DDFA",
            3: "PRNET",

        }

    def set_smooth_params(self, window_size, miu, sigma):
        self.smooth_window = window_size
        self.gaussian_sigma = sigma
        self.gaussian_miu = miu

    def start_align_video(self, seq_folder):
        image_files = glob.glob(os.path.join(seq_folder, "*.png"))
        image_files.sort()

        landmarks = []
        ncount = 0
        for imf in image_files:
            ncount += 1

            landmarks.append(self.model.align(imf))

            progressbar(ncount, len(image_files), prefix="detecting... {}".format(self.names[self.detector]))
        seq_len = len(landmarks)
        landmarks = np.array(landmarks)

        if self.use_smooth:
            landmarks = np.array(landmarks).reshape((seq_len, -1)).T
            landmarks = start_smooth_param(landmarks, self.smooth_mode, win_size=self.smooth_window,
                                           miu=self.gaussian_miu,
                                           sigma=self.gaussian_sigma)
            landmarks = landmarks.T.reshape((seq_len, 68, 2))

        return landmarks
