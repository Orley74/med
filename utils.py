import random
import cv2
import numpy as np
import load_images

model_path = './pose_landmarker_lite.task'

class BodyParts:
    """
    Returns:
        _type_: _description_
    """
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

    all_parts = set(
    PARTS['RIGHT_ARM'] +
    PARTS['LEFT_ARM'] +
    PARTS['RIGHT_LEG'] +
    PARTS['LEFT_LEG'] +
    PARTS['HEAD'] +
    PARTS['BELLY']
)
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
    
class Injures:
    @classmethod
    def noArm_inpaint(cls, frame, mask, arm, image_shape, pose_landmarks):
        """
        Usuwa kończynę (ramię) poprzez zamaskowanie i inpainting.
        Strazsnie laguje, obecnie korzystamy z noPart_Simple
        """
        h, w = image_shape[:2]

        # Wybierz odpowiednie punkty dla ramienia
        if arm == 0:  # prawa ręka
            x1, y1 = int(pose_landmarks[12].x * w), int(pose_landmarks[12].y * h)
            x2, y2 = int(pose_landmarks[14].x * w), int(pose_landmarks[14].y * h)
            x3, y3 = int(pose_landmarks[16].x * w), int(pose_landmarks[16].y * h)
        else:  # lewa ręka
            x1, y1 = int(pose_landmarks[11].x * w), int(pose_landmarks[11].y * h)
            x2, y2 = int(pose_landmarks[13].x * w), int(pose_landmarks[13].y * h)
            x3, y3 = int(pose_landmarks[15].x * w), int(pose_landmarks[15].y * h)

        # Rysowanie maski ramienia
        cv2.line(mask, (x1, y1), (x2, y2), 255, 40)
        cv2.line(mask, (x2, y2), (x3, y3), 255, 40)

        # Upewnij się, że maska jest w odcieniach szarości (jednokanałowa)
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        # Inpainting - symulacja zniknięcia przedmiotu
        inpainted_frame = cv2.inpaint(frame, mask_gray, inpaintRadius=15, flags=cv2.INPAINT_TELEA)

        return inpainted_frame

    @classmethod
    def noPart_simple(cls, frame, mask, chosen_part, place, pose_landmarks):
        """
        Usuwa kończynę poprzez zwykle zamazanie jej.

        chosen_part - zmienna odpowiadajaca za wybor konczyny
        0 - prawa reka
        1 - lewa reka
        2 - prawa noga
        3 - lewa noga
        4 - losowo

        place - miejsce "urwania" 
        0 - gora (przy brzuchu)
        1 - srodek
        2 - dol (dlon lub stopa)
        """
        h, w = frame.shape[:2]
        part = []
        #zmienna na dlon albo stope
        part_end = 0
        belly = BodyParts.getBelly()
        # Wybierz odpowiednie punkty dla ramienia
        if chosen_part == 0:  # prawa ręka
            part = BodyParts.getRightArm()
            part_end = 20

        elif chosen_part == 1:  # lewa ręka
            part_end = 19
            part = BodyParts.getLeftArm()

        elif chosen_part == 2:
            part_end = 32
            part = BodyParts.getRightLeg()

        elif chosen_part == 3:
            part_end = 31
            part = BodyParts.getLeftLeg()
        elif chosen_part == 4:
            part_end = 4
            part = BodyParts.randomPart()
        else:
            raise Exception
        
        #biore wykryty punkt i dopasowuje go do indeksu czesci ciala dlatego wykryty_punkt[czesc_ciala[index]]
        x1, y1 = int(pose_landmarks[part[0]].x * w), int(pose_landmarks[part[0]].y * h)
        x2, y2 = int(pose_landmarks[part[1]].x * w), int(pose_landmarks[part[1]].y * h)
        x3, y3 = int(pose_landmarks[part[2]].x * w), int(pose_landmarks[part[2]].y * h)
        x4, y4 = int(pose_landmarks[part_end].x * w), int(pose_landmarks[part_end].y * h)
        
        # Rysowanie maski ramienia
        try:
            size = w*(abs(pose_landmarks[belly[0]].x-pose_landmarks[belly[1]].x))
            if(place <= 0):
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), int(0.25 * size))
            if(place <= 1):
                cv2.line(frame, (x2, y2), (x3, y3), (0, 0, 255), int(0.20 * size))
            if(place <= 2):
                cv2.line(frame, (x3, y3), (x4, y4), (0, 0, 255), int(0.15 * size))
        except:
            print("linia bylaby za mala // TODO poprawic to z  0 < thickness && thickness <= MAX_THICKNESS in function 'cv::line'")

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
        cx = int(((left_shoulder.x + (0.9 * right_shoulder.x)) / 2) * w)
        shoulder_line = ((left_shoulder.y + right_shoulder.y) / 2) 

        # gorna krawdedz lekko ponizej barkow
        if hip_y != 0:
            cy = int((hip_y+0.85*(shoulder_line - hip_y)) * h)
            
        else:
            cy = int((shoulder_line * h) + 50)

        return cx, cy
    
    @classmethod
    def getHeadCoords(cls, pose_landmarks, image_shape):
        """_summary_

        Args:
            pose_landmarks (list): wykryte czesci ciala
            image_shape (list): wielkosc obrazu

        Returns:
            Zwraca wyliczone wspolrzedne 
            cx - lewy dolny rog serca, cy - prawy gorny rog serca
        """
        h, w = image_shape[:2]
        
        # sprawdzam czy wykryto barki i uda
        try:
            middle_head = pose_landmarks[0]
            right_head = pose_landmarks[7]
            left_head = pose_landmarks[8]
            shoulder = pose_landmarks[11]
        except IndexError:
            return None
        helmet_x = int(((((right_head.x + left_head.x)) / 2) - (abs(right_head.x - left_head.x))) * w)
        helmet_y = int((middle_head.y - (abs(middle_head.y - shoulder.y))) * h)

        return helmet_x, helmet_y
   
    @staticmethod
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
        # img_resized[..., :3] - ten zapis bierze pierwsze 3 elemety img
        # (wszystkie 3 kanaly RGB to te kropki ze bierze wszystko)
        # img_resized[..., 3] - ten bierze 4 element 
        img_rgb = img_resized[..., :3].astype(float)
        img_alpha = img_resized[..., 3].astype(float) / 255.0  

        roi = mask[y_position:y_position+h, x_position:x_position+w, :3].astype(float)
        roi_alpha = mask[y_position:y_position+h, x_position:x_position+w, 3].astype(float) / 255.0

        out_alpha = img_alpha + roi_alpha * (1 - img_alpha)

        for c in range(3):
            roi[..., c] = (img_rgb[..., c] * img_alpha + roi[..., c] * roi_alpha * (1 - img_alpha)) / np.maximum(out_alpha, 1e-5)

        # w razie gdyby wartosci byly wyzsze niz 255, zamieniam na wartosc maks (255)
        mask[y_position:y_position+h, x_position:x_position+w, :3] = np.clip(roi, 0, 255).astype(np.uint8)
        mask[y_position:y_position+h, x_position:x_position+w, 3] = np.clip(out_alpha * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
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
    
    @staticmethod
    def draw_gallery_background(mask, selected_index, main_images, variant_images, image_size=(100, 100), margin=10):
        # Spróbuj załadować logo, jeśli istnieje
        logo = cv2.imread("./images/logo.png", cv2.IMREAD_UNCHANGED)

        h, w = mask.shape[:2]
        main_w, main_h = 100, 100
        variant_w, variant_h = 75, 75

        y_main = h - main_h - margin

        for i, img in enumerate(main_images):
            x_main = margin + i * (main_w + margin)

            # Rysuj warianty nad klikniętym
            if i == selected_index:
                for j, var in enumerate(variant_images):
                    vx = x_main + j * (variant_w + margin)
                    vy = y_main - variant_h - margin

                    # Sprawdź poprawność kanałów
                    if var.shape[2] == 3:
                        var = cv2.cvtColor(var, cv2.COLOR_BGR2BGRA)

                    ImageUtils.draw_rgba(mask, var, vx, vy, size=(variant_w, variant_h))

            # Główne obrazki
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            ImageUtils.draw_rgba(mask, img, x_main, y_main, size=(main_w, main_h))

        if logo is not None:
            ImageUtils.draw_rgba(mask, logo, 0, 0, (logo.shape[1], logo.shape[0]))

    @staticmethod
    def draw_timer(mask, remaining_seconds):
        text = "czas do wykrwawienia"
        minutes = remaining_seconds // 60
        seconds = remaining_seconds % 60
        timer_text = f"{minutes:02}:{seconds:02}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        color = (0, 0, 255, 255)  

        x, y = mask.shape[1] - 250, 40

        cv2.putText(mask, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(mask, timer_text, (x, y + 30), font, scale + 0.2, color, thickness + 1, cv2.LINE_AA)

    @staticmethod
    def draw_Success(mask):
        text = "ZALICZONE"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 3
        color = (0, 255, 0, 255)  

        x, y = mask.shape[1] - 250, 40

        cv2.putText(mask, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
