import cv2
import random

def draw_landmarks(image, landmarks, plot_index=False, colors=None, identity_info=None):
    image1 = image.copy()
    if colors is not None:
        color_num = len(colors)
    else:
        color_num = 1

    if identity_info is not None:
        assert len(identity_info) == len(landmarks)

    for i in range(len(landmarks)):
        lm_num = len(landmarks[i])
        if colors is None:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        else:
            if identity_info is not None:
                color = colors[i % identity_info[i]]
            else:
                color = colors[i % color_num]
        for j in range(lm_num):
            x, y = int(landmarks[i][j][0]), int(landmarks[i][j][1])
            image1 = cv2.circle(image1, (x, y), radius=3, thickness=2, color=color)
            if plot_index:
                image1 = cv2.putText(image1, str(j), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                                            color,
                                            thickness=1)
    return image1