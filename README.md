# Gesture Based Media Controller

Touchless, webcam-based controller for media, volume, web browsing and the mouse,
with gesture-password login and a trainable sign-language recogniser (text + speech).
Built with Python, OpenCV, MediaPipe Hands, PyAutoGUI, scikit-learn, pyttsx3 and pycaw.

## Setup
```
python -m venv venv
venv\Scripts\activate          # Windows  (source venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
python main.py
```
Python 3.9 - 3.11 recommended. Volume control (pycaw) is Windows-only; other
features also run elsewhere but media keys may vary.

## Menu
| Option | Mode |
|---|---|
| 1 | Media control |
| 2 | Enroll gesture password (3 gestures, press S to save each) |
| 3 | Authenticate (average distance < 2.5 grants access) |
| 4 | Touchless web navigation |
| 5 | Virtual mouse |
| 6 | Collect sign data -> sign_language_data.csv |
| 7 | Train Random Forest model -> sign_language_model.pkl |
| 8 | Sign to text/speech (SPACE = speak, C = clear) |
| 9 | Ready-made signs (no training needed) |
| 0 | Exit |

Press **Q** in any camera window to return to the menu.

## Gesture mapping
| Mode | Gesture | Action |
|---|---|---|
| Media | Peace sign | Play / Pause |
| Media | Closed fist | Mute / Unmute |
| Media | Thumb-index distance > 0.15 / < 0.05 | Volume +2% / -2% |
| Web | Open palm | Scroll |
| Web | Index finger | Next tab (Ctrl+Tab) |
| Web | Peace sign | Zoom in (Ctrl +) |
| Mouse | Index fingertip | Move cursor |
| Mouse | Thumb-index pinch | Left click |
| Mouse | Closed fist | Right click |

Ready-made signs (thumb, index, middle, ring, little): A 10000, B 11111, L 11000,
Y 10001, I LOVE YOU 11001, PEACE 01100, GOOD/YES 00000.

Tip: PyAutoGUI fail-safe is on - move the mouse to a screen corner to abort.
