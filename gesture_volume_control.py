import cv2      # Untuk akses webcam
import mediapipe as mp   # Untuk pelacakan tangan
import numpy as np       # Untuk operasi matematika
import os       # Untuk menjalankan perintah sistem di macOS

# Inisialisasi modul pelacakan tangan dari Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Kita hanya melacak satu tangan
mp_drawing = mp.solutions.drawing_utils

# Fungsi untuk mengatur volume sistem (skala 0 hingga 100 di macOS)
def set_volume(vol):
    os.system(f"osascript -e 'set volume output volume {int(vol)}'")

# Mulai menangkap video dari webcam
cap = cv2.VideoCapture(0)

# Inisialisasi variabel untuk kontrol volume
current_volume = 50  # Mulai dengan volume default
smoothing_factor = 0.1  # Faktor penghalusan untuk perubahan volume
gesture_threshold = 0.1  # Ambang untuk posisi jempol menentukan perubahan volume

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Ubah frame ke RGB (Mediapipe menggunakan RGB, tetapi OpenCV menggunakan BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses frame dengan Mediapipe untuk mendeteksi tangan
    results = hands.process(rgb_frame)

    # Periksa apakah ada landmark tangan yang terdeteksi
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan pada frame untuk visualisasi
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Dapatkan landmark untuk ujung jempol dan pergelangan tangan
            thumb_tip = hand_landmarks.landmark[4]
            wrist = hand_landmarks.landmark[0]

            # Dapatkan tinggi dan lebar frame video
            h, w, c = frame.shape

            # Ubah landmark tangan menjadi koordinat piksel
            thumb_tip_y = int(thumb_tip.y * h)
            wrist_y = int(wrist.y * h)

            # Hitung posisi vertikal jempol relatif terhadap pergelangan tangan
            thumb_position = thumb_tip_y - wrist_y

            # Tentukan penyesuaian volume berdasarkan posisi jempol
            if thumb_position < -gesture_threshold * h:  # Gestur jempol ke bawah
                current_volume -= 5  # Mengurangi volume
            elif thumb_position > gesture_threshold * h:  # Gestur jempol ke atas
                current_volume += 5  # Menaikkan volume

            # Batasi volume dalam rentang 0 hingga 100
            current_volume = np.clip(current_volume, 0, 100)

            # Setel volume sistem berdasarkan volume saat ini
            set_volume(current_volume)

            # Tampilkan volume saat ini di layar
            cv2.putText(frame, f'Volume: {int(current_volume)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Gambar lingkaran di ujung jempol
            thumb_tip_x = int(thumb_tip.x * w)
            cv2.circle(frame, (thumb_tip_x, thumb_tip_y), 10, (255, 0, 255), cv2.FILLED)

    # Tampilkan frame dalam jendela
    cv2.imshow('Gesture Volume Control', frame)

    # Hentikan loop jika 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan webcam dan tutup jendela
cap.release()
cv2.destroyAllWindows()
