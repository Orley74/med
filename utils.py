import random
import cv2
import numpy as np


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
               
   
class ImageUtils:

    @classmethod
    def getHeartCoords(cls, pose_landmarks, image_shape):
        """_summary_

        Args:
            pose_landmarks (list): wykryte czesci ciala
            image_shape (list): wielkosc obrazu

        Returns:
            Zwraca wyliczone wspolrzedne 
            cx - lewy dolny rog serca, cy - prawy gorny rog serca
        """
        h, w = image_shape[:2]
        distanceShoulderHip = 0
        hip_y = 0
        # sprawdzam czy wykryto barki i uda
        try:
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
        except IndexError:
            return None
        try:
            left_hip = pose_landmarks[23]
            right_hip = pose_landmarks[24]
            # domyslie zawsze znajdzie jakis punkt (np w tle który oceni ze jest to udo na 5%)
            # wiec sprawdzam czy pewosc wykrycia jest > 50% 
            if left_hip.presence > 0.5:
                hip_y = left_hip.y

            elif right_hip.presence > 0.5:
                hip_y = right_hip.y
            
            else:
                hip_y = 0
        except:
            hip_y = 0 

        # ustawiam lewa krawedz mniej więcej na środku barkow, lekko przesuwajac go w lewa strone     
        cx = ((left_shoulder.x + (0.9 * right_shoulder.x)) / 2) * w
        shoulder_line = ((left_shoulder.y + right_shoulder.y) / 2) 

        # gorna krawdedz lekko ponizej barkow
        if hip_y != 0:
            cy = int((shoulder_line + 0.7*(shoulder_line - hip_y)) * h)
            
        else:
            cy = int((shoulder_line * h) + 50)
        

        return int(cx), cy
    

    def draw_rgba(mask, img, x_position, y_position, size=(50, 50)):
        """    Rysuje obrazek z kanałem alpha (img) na masce RGBA (mask).


        Args:
            mask (_type_): maska wejsciowa 
            img (_type_): wklejane zdjecie
            x_position (_type_): wspolrzedna x, na ktorej ma znalezc sie zdjecie
            y_position (_type_): wspolrzedna y, na ktorej ma znalezc sie zdjecie
            size (tuple, optional): wielkosc zdjecia (x,y)
        """
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        h, w, _ = img_resized.shape

        # najpierw sprawdzam czy obrazek nie wychodzi poza okno i czy poprawnie podane zostaly jego wspolrzedne
        if x_position < 0 or y_position < 0 or x_position + w > mask.shape[1] or y_position + h > mask.shape[0]:
            return
        
        # oddzielam kanaly rgb od alfy (alfa odpowiada za przezroczystosc)
        img_rgb = img_resized[..., :3].astype(float)
        img_alpha = img_resized[..., 3].astype(float) / 255.0  

        roi = mask[y_position:y_position+h, x_position:x_position+w, :3].astype(float)
        roi_alpha = mask[y_position:y_position+h, x_position:x_position+w, 3].astype(float) / 255.0

        out_alpha = img_alpha + roi_alpha * (1 - img_alpha)

        for c in range(3):
            roi[..., c] = (img_rgb[..., c] * img_alpha + roi[..., c] * roi_alpha * (1 - img_alpha)) / np.maximum(out_alpha, 1e-5)

        mask[y_position:y_position+h, x_position:x_position+w, :3] = np.clip(roi, 0, 255).astype(np.uint8)
        mask[y_position:y_position+h, x_position:x_position+w, 3] = np.clip(out_alpha * 255, 0, 255).astype(np.uint8)


    def blend_rgba_over_bgr(background_bgr, overlay_rgba):
        """
        Nakłada overlay_rgba (z kanałem alpha) na tło BGR.
        Zwraca wynik w BGR.

        Args:
            background_bgr (_type_): obraz wejsciowy (najlepiej frame.copy() zeby nie miec bledow z oryginalna klatka)
            overlay_rgba (_type_): maska na obraz (maska z dodanymi zdjeciami za pomoca funkcji draw_rgba())

        Returns:
            _type_: zwraca gotowy do wyswietlenia obraz z maska
        """
        overlay_rgb = overlay_rgba[..., :3]
        alpha = overlay_rgba[..., 3:] / 255.0

        blended = background_bgr * (1 - alpha) + overlay_rgb * alpha
        return blended.astype(np.uint8)