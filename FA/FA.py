import face_alignment
from skimage import io
from FA.AdaptiveWL.core.models import FAN as AdpativeWLFAN
from FA.AdaptiveWL.utils.utils import get_preds_fromhm, cv_crop
import torch
from PIL import Image
import numpy as np
import cv2
from FA.PRNet.api import PRN
import MorphableModelFitting as mmf
import dlib
from utils.common_utils import crop

class BaseFaceAlignment():
    def __init__(self, LandmarkType="2D", flip_input=False, face_detector="dlib", cuda=True):
        if cuda:
            self.device = "cuda"
        else:
            self.device = "cpu"

        LandmarkType = LandmarkType.upper()
        if LandmarkType == "2D":
            ldt = face_alignment.LandmarksType._2D
        elif LandmarkType == "3D":
            ldt = face_alignment.LandmarksType._3D
        else:
            ldt = None
            Exception("wrong landmark type, choose from [2D,3D]")
        self.ldt = ldt

        face_detector = face_detector.lower()
        if face_detector not in ["sfd", "dlib"]:
            Exception("wrong face detector, choose from [sfd, dlib]")
        self.fa = face_alignment.FaceAlignment(ldt, flip_input=flip_input, face_detector=face_detector, device=self.device)

    def align(self, image_path):
        input = io.imread(image_path)
        ldmarks = self.fa.get_landmarks_from_image(input, [[0,0,224,224]])[0]

        if ldmarks is None:
            return []
        if self.ldt == face_alignment.LandmarksType._3D:
            return ldmarks[:, 0:2]
        return ldmarks


class AdaptiveWLFaceAlignment():
    def __init__(self, cuda=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
        self.model = AdpativeWLFAN(4, "False", "False", 98)
        checkpoint = torch.load("./data/WFLW_4HG.pth")
        if 'state_dict' not in checkpoint:
            self.model.load_state_dict(checkpoint)
        else:
            pretrained_weights = checkpoint['state_dict']
            model_weights = self.model.state_dict()
            pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                                  if k in model_weights}
            model_weights.update(pretrained_weights)
            self.model.load_state_dict(model_weights)

        self.center = [450 // 2, 450 // 2 + 0]
        self.scale = 2.25
        self.model.to(self.device)

        self.ld68_config = {
            "face_coutour": [2 * i + 1 for i in range(16)] + [32],
            "eyebow" : [33, [34, 41], [35,40], [36,39], [37,38], [42, 50], [43, 49], [44,48], [45,47], 46],
            "nose":[51,52,53,54,55,56,57,58,59],
            "eyes": [60, 61,[62,63], 64,[65,66],67,68,[69,70],71,72,73,[74,75]],
            "mouse": [i for i in range(76,96)]
        }
    def transkp98_68(self, kp98):
        kp68 = []
        for k, indexs in self.ld68_config.items():
            for idx in indexs:
                if isinstance(idx, list):
                    kp68.append(np.mean(kp98[idx], axis=0))
                else:
                    kp68.append(kp98[idx])
        return np.array(kp68)
    def align(self, image_path):
        image = Image.open(image_path)
        image = image.resize((256, 256))

        image, _ = cv_crop(np.array(image), np.random.random_sample((98,2))*254, self.center, self.scale, 256, 100)

        '''
        bottom = 255
        right = 255
        while image[0, right, 0] == 0:
            right = right - 1
        while image[bottom, 0, 0] == 0:
            bottom = bottom - 1
        '''

        inputs = torch.from_numpy(np.array(image).transpose((2,0,1))).float().div(255.0).unsqueeze(0)

        inputs = inputs.to(self.device)

        with torch.no_grad():
            outputs, boundary_channels = self.model(inputs)
            pred_heatmap = outputs[-1][:, :-1, :, :][0].detach().cpu()
            pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0))
            pred_landmarks = pred_landmarks.squeeze().numpy()

        #pred_landmarks[:, 0] = pred_landmarks[:, 0] * (224.0 / )
        return self.transkp98_68(pred_landmarks * (224.0 / 34.8))
        #print(pred_landmarks)


class PRNetFaceAlignment():
    def __init__(self):
        self.prn = PRN()

    def align(self, image_path):
        image = cv2.imread(image_path)[:,:,::-1]
        image = cv2.resize(image, (256, 256)) / 255

        pos = self.prn.net_forward(image)
        kp68 = self.prn.get_landmarks(pos)[:, 0:2]
        return kp68 * (224.0 / 256)
        #print(pos)

class DDFAFaceAlignment():
    def __init__(self):
        self.model = mmf.FaceLandmarkDetector("3D")

    def align(self, image_path):
        rects = [[0, 0, 224, 224]]
        kp68 = self.model.detect_face_landmark(image_path, rects)
        return kp68[0]


