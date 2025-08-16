import cv2
import numpy as np
import os
import glob


inputs = "dataset"
results = "results"

if not os.path.exists(results):
    os.makedirs(results)


image_paths = glob.glob(os.path.join(inputs, "*.jpg")) + glob.glob(os.path.join(inputs, "*.png"))
print(f"{len(image_paths)} görsel bulundu. İşlem başlatılıyor...")


for path in image_paths:
    img1 = cv2.imread(path)
    if img1 is None:
        print(f"[HATA] Görsel yüklenemedi: {path}")
        continue

    
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    mean_v = np.mean(img[:, :, 2])

    
    if mean_v < 80:
        img1 = cv2.convertScaleAbs(img1, alpha=2.0, beta=60)
        print("Karanlık görsel tespit edildi, aydınlatıldı.")
        img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)  

    
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(img, lower_red1, upper_red1)
    mask2 = cv2.inRange(img, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            
            cx = x + w // 2
            cy = y + h // 2
            if cy > img1.shape[0] * 0.75:
                print(f"[ALT] {os.path.basename(path)} → Alt kısımda bulundu, atlandı ({cy})")
                continue

            if 0.8 < aspect_ratio < 1.2:
                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

                if 6 <= len(approx) <= 10:
                    cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(img1, (cx, cy), 5, (255, 0, 0), -1)
                    print(f"[OK] {os.path.basename(path)} → STOP tespit edildi ({cx},{cy})")
                    detected = True

    if detected:
        out_path = os.path.join(results, os.path.basename(path))
        cv2.imwrite(out_path, img1)
    else:
        print(f"[BOŞ] {os.path.basename(path)} → STOP bulunamadı.")

print("✅ Tüm görseller işlendi.")
