"""
Gesture Based Media Controller
--------------------------------
A touchless, webcam-based controller built with OpenCV, MediaPipe Hands,
PyAutoGUI, scikit-learn, pyttsx3 and pycaw.

Modes (console menu):
  1 Media (play/pause, mute, volume)   2 Enroll gesture password
  3 Authenticate                       4 Touchless web navigation
  5 Virtual mouse                      6 Collect sign data
  7 Train sign model                   8 Sign -> text/speech
  9 Ready-made signs                   0 Exit
"""
import csv
import os
import pickle
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ---------- optional libraries (the program still runs without them) ----------
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
except ImportError:
    RandomForestClassifier = accuracy_score = train_test_split = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:  # Windows master-volume control
    from ctypes import POINTER, cast
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    PYCAW_AVAILABLE = True
except Exception:
    PYCAW_AVAILABLE = False

# ---------- MediaPipe set-up ----------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

pyautogui.FAILSAFE = True
SW, SH = pyautogui.size()                      # screen width and height

PASSWORD_FILE = "gesture_password.pkl"
DATA_FILE = "sign_language_data.csv"
MODEL_FILE = "sign_language_model.pkl"


# =====================================================================
# 4.1  Hand detection and feature extraction
# =====================================================================
def get_landmarks(h):
    """Wrist-relative, scale-normalised vector of 42 values (21 x, y pairs)."""
    p = []
    bx, by = h.landmark[0].x, h.landmark[0].y
    for lm in h.landmark:
        p += [lm.x - bx, lm.y - by]
    m = max(abs(x) for x in p) if p else 0
    return [x / m for x in p] if m else p


def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def fingers_up(h):
    """Returns [thumb, index, middle, ring, little] as 1 (up) or 0 (down)."""
    lm = h.landmark
    f = [1 if lm[4].x < lm[3].x else 0]                      # thumb (x-axis)
    f += [1 if lm[t].y < lm[q].y else 0                      # fingertip above
          for t, q in zip([8, 12, 16, 20], [6, 10, 14, 18])]  # middle joint
    return f


def gesture(h):
    f = fingers_up(h)
    return {(0, 0, 0, 0, 0): "FIST", (0, 1, 1, 0, 0): "PEACE",
            (1, 1, 1, 1, 1): "OPEN_PALM", (1, 0, 0, 0, 0): "THUMB",
            (0, 1, 0, 0, 0): "INDEX"}.get(tuple(f), "UNKNOWN")


def txt(frame, s, pos=(20, 50), scale=0.8, color=(0, 255, 0)):
    cv2.putText(frame, s, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


# =====================================================================
# 4.2  Gesture media and volume control
# =====================================================================
def _endpoint():
    """Return the Windows master-volume endpoint (works across pycaw versions)."""
    d = AudioUtilities.GetSpeakers()
    if hasattr(d, "EndpointVolume"):                         # newer pycaw
        return d.EndpointVolume
    i = d.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(i, POINTER(IAudioEndpointVolume))


def getvol():
    """Current Windows master volume (0-100); 50 if unavailable."""
    if not PYCAW_AVAILABLE:
        return 50
    try:
        return int(round(_endpoint().GetMasterVolumeLevelScalar() * 100))
    except Exception:
        return 50


def volume(v):
    """Set the Windows master volume (0-100) using pycaw."""
    if not PYCAW_AVAILABLE:
        return
    try:
        _endpoint().SetMasterVolumeLevelScalar(max(0, min(100, v)) / 100, None)
    except Exception:
        pass


def media_control():
    cap = cv2.VideoCapture(0)
    v = getvol(); last = 0; action = ""
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.flip(f, 1)
        r = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if r.multi_hand_landmarks:
            h = r.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(f, h, mp_hands.HAND_CONNECTIONS)
            g = gesture(h); lm = h.landmark
            if g == "PEACE" and time.time() - last > 1:
                pyautogui.press("playpause")
                action = "PLAY/PAUSE"; last = time.time()
            if g == "FIST" and time.time() - last > 1:
                pyautogui.press("volumemute")
                action = "MUTE/UNMUTE"; last = time.time()
            # thumb-tip to index-tip distance controls the volume
            d = np.linalg.norm(np.array([lm[4].x, lm[4].y]) -
                               np.array([lm[8].x, lm[8].y]))
            if d > 0.15:
                v = min(100, v + 2); volume(v)
            elif d < 0.05:
                v = max(0, v - 2); volume(v)
        txt(f, f"Volume: {v}%", (20, 450))
        if time.time() - last < 1.5:
            txt(f, action, (200, 100), 1, (0, 0, 255))
        cv2.imshow("Gesture Media Controller", f)
        if cv2.waitKey(1) & 255 == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


# =====================================================================
# 4.3  Gesture password: enrol and authenticate
# =====================================================================
def capture(cap, n):
    """Wait until the user presses S to save the n-th gesture (Q = quit)."""
    while True:
        ok, f = cap.read()
        if not ok:
            return None
        f = cv2.flip(f, 1)
        r = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        txt(f, f"Gesture {n}/3 | S=save Q=quit", (20, 50), 0.7, (0, 255, 255))
        if r.multi_hand_landmarks:
            h = r.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(f, h, mp_hands.HAND_CONNECTIONS)
            cv2.imshow("Gesture Login", f)
            k = cv2.waitKey(1) & 255
            if k == ord("s"):
                return get_landmarks(h)
            if k == ord("q"):
                return None
        else:
            cv2.imshow("Gesture Login", f)
            if cv2.waitKey(1) & 255 == ord("q"):
                return None


def enroll():
    cap = cv2.VideoCapture(0); p = []
    for i in range(1, 4):                      # record three gestures
        x = capture(cap, i)
        if x is None:
            cap.release(); cv2.destroyAllWindows(); return
        p.append(x)
    with open(PASSWORD_FILE, "wb") as f:
        pickle.dump(p, f)
    cap.release(); cv2.destroyAllWindows()
    print("Gesture password saved.")


def auth():
    if not os.path.exists(PASSWORD_FILE):
        print("Enroll first."); return
    with open(PASSWORD_FILE, "rb") as f:
        stored = pickle.load(f)
    cap = cv2.VideoCapture(0); cur = []
    for i in range(1, 4):
        x = capture(cap, i)
        if x is None:
            cap.release(); cv2.destroyAllWindows(); return
        cur.append(x)
    cap.release(); cv2.destroyAllWindows()
    d = sum(distance(a, b) for a, b in zip(cur, stored)) / 3
    print("ACCESS GRANTED" if d < 2.5 else "ACCESS DENIED",
          f"(distance {d:.3f})")


# =====================================================================
# 4.4  Touchless web navigation and virtual mouse
# =====================================================================
def web():
    cap = cv2.VideoCapture(0); last = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.flip(f, 1)
        r = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if r.multi_hand_landmarks and time.time() - last > 0.8:
            h = r.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(f, h, mp_hands.HAND_CONNECTIONS)
            g = gesture(h)
            if g == "OPEN_PALM":
                pyautogui.scroll(3); txt(f, "SCROLLING"); last = time.time()
            elif g == "INDEX":
                pyautogui.hotkey("ctrl", "tab"); txt(f, "NEXT TAB"); last = time.time()
            elif g == "PEACE":
                pyautogui.hotkey("ctrl", "+"); txt(f, "ZOOM IN"); last = time.time()
        cv2.imshow("Touchless Web Navigation", f)
        if cv2.waitKey(1) & 255 == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


def mouse():
    cap = cv2.VideoCapture(0); last = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.flip(f, 1); h, w, _ = f.shape
        r = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        cv2.rectangle(f, (100, 80), (w - 100, h - 80), (255, 0, 255), 2)
        if r.multi_hand_landmarks:
            hand = r.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(f, hand, mp_hands.HAND_CONNECTIONS)
            lm = hand.landmark
            x, y = int(lm[8].x * w), int(lm[8].y * h)       # index fingertip
            pyautogui.moveTo(np.interp(x, [100, w - 100], [0, SW]),
                             np.interp(y, [80, h - 80], [0, SH]), duration=0.05)
            p = np.linalg.norm(np.array([lm[4].x, lm[4].y]) -
                               np.array([lm[8].x, lm[8].y]))
            if p < 0.05 and time.time() - last > 0.7:
                pyautogui.click(); txt(f, "LEFT CLICK"); last = time.time()
            elif gesture(hand) == "FIST" and time.time() - last > 0.7:
                pyautogui.rightClick(); txt(f, "RIGHT CLICK"); last = time.time()
        cv2.imshow("Virtual Mouse", f)
        if cv2.waitKey(1) & 255 == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


# =====================================================================
# 4.5  Sign data collection and model training
# =====================================================================
def collect():
    label = input("Sign label: ").strip()
    if not label:
        return
    cap = cv2.VideoCapture(0); n = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.flip(f, 1)
        r = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        k = cv2.waitKey(1) & 255
        if r.multi_hand_landmarks:
            h = r.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(f, h, mp_hands.HAND_CONNECTIONS)
            if k == ord("s"):                       # save one labelled sample
                new = not os.path.exists(DATA_FILE)
                with open(DATA_FILE, "a", newline="") as z:
                    w = csv.writer(z)
                    if new:
                        w.writerow(["label"] + [f"f{i}" for i in range(42)])
                    w.writerow([label] + get_landmarks(h))
                n += 1
        txt(f, f"{label} samples:{n}  (S=save Q=quit)")
        cv2.imshow("Sign Data Collection", f)
        if k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


def train():
    if RandomForestClassifier is None:
        print("Install scikit-learn."); return
    if not os.path.exists(DATA_FILE):
        print("Collect data first."); return
    X = []; y = []
    with open(DATA_FILE) as f:
        r = csv.reader(f); next(r, None)
        for row in r:
            if len(row) >= 43:
                y.append(row[0]); X.append([float(x) for x in row[1:43]])
    if len(X) < 2:
        print("Not enough samples."); return
    X = np.array(X); y = np.array(y)
    try:
        a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=42,
                                      stratify=y if len(set(y)) > 1 else None)
    except ValueError:      # too few samples per label to stratify
        a, b, c, d = train_test_split(X, y, test_size=0.2, random_state=42)
    m = RandomForestClassifier(n_estimators=100, random_state=42)
    m.fit(a, c)
    print("Accuracy:", accuracy_score(d, m.predict(b)) * 100)
    with open(MODEL_FILE, "wb") as z:
        pickle.dump(m, z)
    print("Model saved to", MODEL_FILE)


# =====================================================================
# 4.6  Sign-to-speech, ready-made signs and main menu
# =====================================================================
def speak(s):
    if pyttsx3 and s.strip():
        e = pyttsx3.init(); e.say(s); e.runAndWait()


def sign_text():
    if not os.path.exists(MODEL_FILE):
        print("Train model first."); return
    with open(MODEL_FILE, "rb") as z:
        m = pickle.load(z)
    cap = cv2.VideoCapture(0); sentence = ""; last = ""; t = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.flip(f, 1)
        r = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); pred = ""
        if r.multi_hand_landmarks:
            h = r.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(f, h, mp_hands.HAND_CONNECTIONS)
            pred = m.predict([get_landmarks(h)])[0]
            if pred != last and time.time() - t > 1:
                sentence += str(pred) + " "; last = pred; t = time.time()
        txt(f, f"Sign: {pred}")
        txt(f, f"Sentence: {sentence}", (20, 100), 0.7, (255, 255, 255))
        cv2.imshow("Sign Language", f); k = cv2.waitKey(1) & 255
        if k == ord(" "):
            speak(sentence)          # SPACE = speak
        elif k == ord("c"):
            sentence = ""            # C = clear
        elif k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


READY_SIGNS = {(1, 0, 0, 0, 0): "A", (1, 1, 1, 1, 1): "B", (1, 1, 0, 0, 0): "L",
               (1, 0, 0, 0, 1): "Y", (1, 1, 0, 0, 1): "I LOVE YOU",
               (0, 1, 1, 0, 0): "PEACE", (0, 0, 0, 0, 0): "GOOD / YES"}


def ready():
    """Recognise a fixed set of signs from finger-state patterns (no training)."""
    cap = cv2.VideoCapture(0); sentence = ""; last = ""; t = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.flip(f, 1)
        r = hands.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); pred = ""
        if r.multi_hand_landmarks:
            h = r.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(f, h, mp_hands.HAND_CONNECTIONS)
            pred = READY_SIGNS.get(tuple(fingers_up(h)), "")
            if pred and pred != last and time.time() - t > 1:
                sentence += pred + " "; last = pred; t = time.time()
        txt(f, f"Sign: {pred}")
        txt(f, f"Sentence: {sentence}", (20, 100), 0.7, (255, 255, 255))
        cv2.imshow("Ready-Made Signs", f); k = cv2.waitKey(1) & 255
        if k == ord(" "):
            speak(sentence)
        elif k == ord("c"):
            sentence = ""; last = ""
        elif k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


def main():
    funcs = {"1": media_control, "2": enroll, "3": auth, "4": web, "5": mouse,
             "6": collect, "7": train, "8": sign_text, "9": ready}
    while True:
        print("\nGESTURE-BASED MEDIA CONTROLLER")
        print("1 Media  2 Enroll  3 Authenticate  4 Web  5 Mouse  6 Collect")
        print("7 Train  8 Sign Text/Speech  9 Ready Signs  0 Exit")
        c = input("Choice: ").strip()
        if c == "0":
            break
        if c in funcs:
            try:
                funcs[c]()
            except Exception as e:
                print("Error:", e)


if __name__ == "__main__":
    main()
