import time
import threading

import mss
import numpy as np
import pyautogui
from pynput import keyboard
from ultralytics import YOLO


model = YOLO("runs/detect/cookie_detector-4/weights/best.onnx")
CONFIDENCE = 0.7

sct = mss.mss()
monitor = sct.monitors[1]

# Пауза PyAutoGUI после действия — 5 мс.
pyautogui.PAUSE = 0.005

# Установлено — программа работает; снято — пауза.
active = threading.Event()
active.set()

space_is_down = False


def on_press(key):
    global space_is_down

    if key == keyboard.Key.space and not space_is_down:
        space_is_down = True

        if active.is_set():
            active.clear()
            print("Пауза. Нажмите пробел, чтобы продолжить.")
        else:
            active.set()
            print("Продолжаю работу.")


def on_release(key):
    global space_is_down

    if key == keyboard.Key.space:
        space_is_down = False


def find_and_click(class_name):
    active.wait()

    # Скриншот с основного монитора; первые три канала — BGR для модели.
    screenshot = np.array(sct.grab(monitor))[:, :, :3]

    results = model(screenshot, verbose=False, device="cpu")

    for result in results:
        for box in result.boxes:
            if not active.is_set():
                return False

            detected_class = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])

            if detected_class == class_name and confidence >= CONFIDENCE:
                x1, y1, x2, y2 = box.xyxy[0]
                x = int((x1 + x2) / 2) + monitor["left"]
                y = int((y1 + y2) / 2) + monitor["top"]

                pyautogui.click(x, y)
                print(f"Клик по {class_name}: ({x}, {y})")
                return True

    return False


def click_with_timeout(class_name, timeout=1):
    deadline = time.time() + timeout

    while time.time() < deadline:
        active.wait()

        if find_and_click(class_name):
            return True

        time.sleep(0.05)

    print(f"{class_name} не найден за {timeout} секунду — перехожу дальше.")
    return False


if __name__ == "__main__":
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )
    listener.start()

    print("Программа запущена. Пробел — пауза/продолжение.")

    try:
        while True:
            # Пытается нажать game до 4 раз.
            for _ in range(10):
                if not click_with_timeout("game", timeout=1):
                    break

            # Ищет taper до секунды, кликает один раз и сразу переходит к reklama.
            click_with_timeout("taper", timeout=1)
            click_with_timeout("reklama", timeout=1)

    except KeyboardInterrupt:
        print("\nПрограмма завершена.")
    finally:
        listener.stop()
        sct.close()
