"""Face Recognition Attendance System (refactored).

Requires: pip install opencv-contrib-python numpy pillow
"""
import csv
import hashlib
import hmac
import json
import os
import time
import tkinter as tk
import tkinter.simpledialog as sd
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image

# ----------------------------------------------------------------- config
BASE = Path(__file__).resolve().parent
TRAIN_DIR = BASE / "TrainingImage"
MODEL_DIR = BASE / "TrainingImageLabel"
ATTEND_DIR = BASE / "Attendance"
STUDENTS_CSV = BASE / "StudentDetails" / "students.csv"
MODEL_FILE = MODEL_DIR / "trainer.yml"
PASSWORD_FILE = MODEL_DIR / "password.json"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # ships with OpenCV

SAMPLES_PER_PERSON = 50
CONFIDENCE_THRESHOLD = 50  # LBPH: lower = better match

for d in (TRAIN_DIR, MODEL_DIR, ATTEND_DIR, STUDENTS_CSV.parent):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- password (hashed, not plaintext)
def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()


def set_password(password: str) -> None:
    salt = os.urandom(16)
    PASSWORD_FILE.write_text(json.dumps({"salt": salt.hex(), "hash": _hash(password, salt)}))


def check_password(password: str) -> bool:
    data = json.loads(PASSWORD_FILE.read_text())
    expected = _hash(password, bytes.fromhex(data["salt"]))
    return hmac.compare_digest(expected, data["hash"])


# ----------------------------------------------------------------- student storage
def load_students() -> dict[int, str]:
    if not STUDENTS_CSV.exists():
        return {}
    with open(STUDENTS_CSV, newline="") as f:
        return {int(r["id"]): r["name"] for r in csv.DictReader(f)}


def add_student(sid: int, name: str) -> None:
    new_file = not STUDENTS_CSV.exists()
    with open(STUDENTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["id", "name"])
        w.writerow([sid, name])


# ----------------------------------------------------------------- attendance storage
def today_file() -> Path:
    return ATTEND_DIR / f"Attendance_{datetime.now():%d-%m-%Y}.csv"


def read_attendance() -> list[list[str]]:
    f = today_file()
    if not f.exists():
        return []
    with open(f, newline="") as fh:
        return list(csv.reader(fh))[1:]


def mark_attendance(sid: int, name: str) -> bool:
    """Write a row once per person per day. Returns True if newly marked."""
    if any(r[0] == str(sid) for r in read_attendance()):
        return False
    f = today_file()
    new_file = not f.exists()
    now = datetime.now()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["Id", "Name", "Date", "Time"])
        w.writerow([sid, name, now.strftime("%d-%m-%Y"), now.strftime("%H:%M:%S")])
    return True


# ----------------------------------------------------------------- app
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Attendance System")
        root.geometry("1000x620")
        root.configure(bg="#2d420a")

        self.detector = cv2.CascadeClassifier(CASCADE)
        if self.detector.empty():
            messagebox.showerror("Error", "Could not load face detector.")
            root.destroy()
            return

        self._build_ui()
        self._ensure_password()
        self.refresh_table()
        self.tick()

    # ---------- UI
    def _build_ui(self):
        r = self.root
        tk.Label(r, text="Face Recognition Attendance System", fg="white", bg="#2d420a",
                 font=("Arial", 24, "bold")).pack(pady=10)
        self.clock = tk.Label(r, fg="#ff61e5", bg="#2d420a", font=("Arial", 18, "bold"))
        self.clock.pack()

        body = tk.Frame(r, bg="#2d420a")
        body.pack(fill="both", expand=True, padx=20, pady=10)

        # left: attendance
        left = tk.LabelFrame(body, text="Attendance", bg="#c79cff", font=("Arial", 13, "bold"))
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        tk.Button(left, text="Take Attendance", bg="#3ffc00", font=("Arial", 13, "bold"),
                  command=self.take_attendance).pack(fill="x", padx=15, pady=10)
        self.tv = ttk.Treeview(left, columns=("name", "date", "time"), height=14)
        for col, text, w in (("#0", "ID", 60), ("name", "NAME", 130), ("date", "DATE", 100), ("time", "TIME", 90)):
            self.tv.heading(col, text=text)
            self.tv.column(col, width=w)
        sb = ttk.Scrollbar(left, orient="vertical", command=self.tv.yview)
        self.tv.configure(yscrollcommand=sb.set)
        self.tv.pack(side="left", fill="both", expand=True, padx=(15, 0), pady=10)
        sb.pack(side="left", fill="y", pady=10, padx=(0, 10))

        # right: registration
        right = tk.LabelFrame(body, text="New Registration", bg="#c79cff", font=("Arial", 13, "bold"))
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="ID (number)", bg="#c79cff", font=("Arial", 12, "bold")).pack(pady=(15, 0))
        self.id_entry = tk.Entry(right, font=("Arial", 13), width=30)
        self.id_entry.pack()
        tk.Label(right, text="Name", bg="#c79cff", font=("Arial", 12, "bold")).pack(pady=(15, 0))
        self.name_entry = tk.Entry(right, font=("Arial", 13), width=30)
        self.name_entry.pack()
        tk.Button(right, text="Clear", bg="#ff7221", command=self.clear_form).pack(pady=8)
        tk.Button(right, text="1) Take Images", bg="#6d00fc", fg="white", font=("Arial", 13, "bold"),
                  command=self.take_images).pack(fill="x", padx=20, pady=5)
        tk.Button(right, text="2) Save Profile (train)", bg="#6d00fc", fg="white", font=("Arial", 13, "bold"),
                  command=self.train_with_password).pack(fill="x", padx=20, pady=5)
        self.status = tk.Label(right, bg="#c79cff", font=("Arial", 11, "bold"), wraplength=380)
        self.status.pack(pady=15)
        self.update_status()

        menu = tk.Menu(r)
        help_menu = tk.Menu(menu, tearoff=0)
        help_menu.add_command(label="Change Password", command=self.change_password)
        help_menu.add_command(label="Exit", command=r.destroy)
        menu.add_cascade(label="Help", menu=help_menu)
        r.config(menu=menu)

    def tick(self):
        self.clock.config(text=time.strftime("%d-%b-%Y  |  %H:%M:%S"))
        self.root.after(500, self.tick)

    def update_status(self, text: str | None = None):
        self.status.config(text=text or f"Total registrations: {len(load_students())}")

    def clear_form(self):
        self.id_entry.delete(0, "end")
        self.name_entry.delete(0, "end")
        self.update_status()

    def refresh_table(self):
        self.tv.delete(*self.tv.get_children())
        for sid, name, date, t in reversed(read_attendance()):
            self.tv.insert("", "end", text=sid, values=(name, date, t))

    # ---------- password
    def _ensure_password(self):
        if PASSWORD_FILE.exists():
            return
        pw = sd.askstring("Set password", "No password found. Create one:", show="*", parent=self.root)
        if pw:
            set_password(pw)
            messagebox.showinfo("Password", "Password saved.")

    def change_password(self):
        if not PASSWORD_FILE.exists():
            return self._ensure_password()
        old = sd.askstring("Change password", "Old password:", show="*", parent=self.root)
        if old is None:
            return
        if not check_password(old):
            return messagebox.showerror("Wrong password", "Old password is incorrect.")
        new = sd.askstring("Change password", "New password:", show="*", parent=self.root)
        if not new:
            return
        if sd.askstring("Change password", "Confirm new password:", show="*", parent=self.root) != new:
            return messagebox.showerror("Error", "Passwords do not match.")
        set_password(new)
        messagebox.showinfo("Done", "Password changed.")

    # ---------- registration
    def take_images(self):
        sid_text, name = self.id_entry.get().strip(), self.name_entry.get().strip()
        if not sid_text.isdigit():
            return messagebox.showerror("Invalid ID", "ID must be a number.")
        if not name.replace(" ", "").isalpha():
            return messagebox.showerror("Invalid name", "Name may contain only letters and spaces.")
        sid = int(sid_text)
        if sid in load_students():
            return messagebox.showerror("Duplicate", f"ID {sid} is already registered.")

        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return messagebox.showerror("Camera", "Could not open the camera.")

        count = 0
        try:
            while count < SAMPLES_PER_PERSON:
                ok, frame = cam.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(gray, 1.3, 5)
                if len(faces):
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face only
                    count += 1
                    cv2.imwrite(str(TRAIN_DIR / f"{sid}_{count}.jpg"), gray[y:y + h, x:x + w])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{count}/{SAMPLES_PER_PERSON}  (q = quit)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Taking Images", frame)
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()

        if count == 0:
            return messagebox.showwarning("No face", "No face was captured. Please try again.")
        add_student(sid, name)
        self.update_status(f"Captured {count} images for ID {sid}. Now click 'Save Profile'.")

    # ---------- training
    def train_with_password(self):
        pw = sd.askstring("Password", "Enter password:", show="*", parent=self.root)
        if pw is None:
            return
        if not PASSWORD_FILE.exists() or not check_password(pw):
            return messagebox.showerror("Wrong password", "Incorrect password.")
        self.train()

    def train(self):
        faces, ids = [], []
        for p in TRAIN_DIR.glob("*.jpg"):
            try:
                ids.append(int(p.stem.split("_")[0]))
                faces.append(np.array(Image.open(p).convert("L"), "uint8"))
            except (ValueError, OSError):
                continue
        if not faces:
            return messagebox.showwarning("No data", "Please register someone first.")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.write(str(MODEL_FILE))
        self.update_status(f"Profile saved. {len(set(ids))} people trained.")

    # ---------- attendance
    def take_attendance(self):
        if not MODEL_FILE.exists():
            return messagebox.showwarning("No model", "Please click 'Save Profile' first.")
        students = load_students()
        if not students:
            return messagebox.showwarning("No students", "No registered students found.")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(MODEL_FILE))
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            return messagebox.showerror("Camera", "Could not open the camera.")

        try:
            while True:
                ok, frame = cam.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for x, y, w, h in self.detector.detectMultiScale(gray, 1.2, 5):
                    sid, conf = recognizer.predict(gray[y:y + h, x:x + w])
                    if conf < CONFIDENCE_THRESHOLD and sid in students:
                        label = students[sid]
                        if mark_attendance(sid, label):
                            self.refresh_table()
                        color = (0, 255, 0)
                    else:
                        label, color = "Unknown", (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                cv2.imshow("Taking Attendance (q = quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cam.release()
            cv2.destroyAllWindows()
        self.refresh_table()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
