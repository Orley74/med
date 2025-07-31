import random
import cv2


model_path = './pose_landmarker_lite.task'

class BodyParts:
    """
    Returns:
        _type_: _description_
    """
    heart_img = cv2.imread("./heart.png", cv2.IMREAD_UNCHANGED)

    RIGHT_ARM = [12,14,16]
    LEFT_ARM = [11,13,15]
    RIGHT_LEG = [24,26,28]
    LEFT_LEG = [23,25,27]
    HEAD = list(range(11))
    BELLY = [11,12,23,24]
    
    PARTS = {
        "RIGHT_ARM": [12,14,16],
        "LEFT_ARM": [11,13,15],
        "RIGHT_LEG": [24,26,28],
        "LEFT_LEG": [23,25,27],
        "HEAD": list(range(11)),
        "BELLY": [11,12,23,24]
    }

    all_parts = set(PARTS['RIGHT_ARM'] + PARTS['LEFT_ARM'] + ['RIGHT_LEG'] + ['LEFT_LEG'] + ['HEAD'] + ['BELLY'])

    @classmethod
    def getRightArm(cls):
        """ Metoda zwracajaca kordy prawego ramienia

        Returns:
            Lista kordow
        """
        return cls.PARTS['RIGHT_ARM']
    
    @classmethod
    def getLeftArm(cls):
        return cls.PARTS['LEFT_ARM']
    @classmethod
    def getRightLeg(cls):
        return cls.PARTS['RIGHT_LEG']
    
    @classmethod
    def getLeftLeg(cls):
        return cls.PARTS['LEFT_LEG']
    @classmethod
    def getHead(cls):
        return cls.PARTS['HEAD']
    
    @classmethod
    def getBelly(cls):
        return cls.PARTS['BELLY']

    @classmethod
    def randomSinglePart(cls):
        """
        Losowo wybrana jedna czesc ciala (jeden punkt na ciele)
        w funkcji do rysowania potrzebna jest lista dlatego na jana jest zrobiona
        lista jednoelementowa z wyiku losowania
        """
        target = random.choice(list(cls.all_parts))
        return [target]
    
    @classmethod
    def randomPart(cls):
        """
        Zwraca losowo wybraną część ciała (nazwa i indeksy).

        Returns:
            tuple[str, list[int]]: Nazwa części i jej indeksy.
        """
        name = random.choice(list(cls.PARTS.keys()))
        return cls.PARTS[name]
    

    @staticmethod
    def _overlay_image(bg, fg, x, y):
        """
        Nakłada fg (z kanałem alpha) na tło bg w punkcie (x, y).
        """
        h, w = fg.shape[:2]

        if y + h > bg.shape[0] or x + w > bg.shape[1] or x < 0 or y < 0:
            return

        alpha_fg = fg[:, :, 3] / 255.0
        alpha_bg = 1.0 - alpha_fg

        for c in range(3):
            bg[y:y+h, x:x+w, c] = (alpha_fg * fg[:, :, c] +
                                   alpha_bg * bg[y:y+h, x:x+w, c])
               
    @classmethod
    def getHeartCoords(cls, pose_landmarks, image_shape):
        h, w = image_shape[:2]
        try:
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
        except IndexError:
            return None

        shoulder_cx = (left_shoulder.x + right_shoulder.x) / 2 * w
        shoulder_cy = (left_shoulder.y + right_shoulder.y) / 2 * h
        hip_cx = (left_hip.x + right_hip.x) / 2 * w
        hip_cy = (left_hip.y + right_hip.y) / 2 * h

        cx = int((shoulder_cx + hip_cx) / 2)
        cy = int(shoulder_cy + 0.7 * (hip_cy - shoulder_cy))
        cx += int(0.07 * w)

        return cx, cy