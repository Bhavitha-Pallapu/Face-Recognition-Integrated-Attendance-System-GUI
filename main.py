"""Face Recognition Attendance System.

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

try:
    for d in (TRAIN_DIR, MODEL_DIR, ATTEND_DIR, STUDENTS_CSV.parent):
        d.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    raise RuntimeError(f"Could not create application data directories: {exc}") from exc


# ----------------------------------------------------------------- password (hashed, not plaintext)
def _hash(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000).hex()


def set_password(password: str) -> None:
    salt = os.urandom(16)
    PASSWORD_FILE.write_text(json.dumps({"salt": salt.hex(), "hash": _hash(password, salt)}))


def check_password(password: str) -> bool:
    """Return False for invalid or unreadable password data instead of crashing the UI."""
    try:
        data = json.loads(PASSWORD_FILE.read_text())
        salt = bytes.fromhex(data["salt"])
        stored_hash = data["hash"]
        if not isinstance(stored_hash, str):
            return False
        expected = _hash(password, salt)
        return hmac.compare_digest(expected, stored_hash)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


# ----------------------------------------------------------------- student storage
def load_students() -> dict[int, str]:
    if not STUDENTS_CSV.exists():
        return {}
    try:
        with open(STUDENTS_CSV, newline="", encoding="utf-8") as f:
            students = {}
            for row in csv.DictReader(f):
                sid = int(row["id"])
                name = row["name"].strip()
                if not name:
                    raise ValueError(f"Student {sid} has an empty name.")
                students[sid] = name
            return students
    except (OSError, csv.Error, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Could not read student records in {STUDENTS_CSV.name}: {exc}") from exc


def add_student(sid: int, name: str) -> None:
    new_file = not STUDENTS_CSV.exists()
    with open(STUDENTS_CSV, "a", newline="", encoding="utf-8") as f:
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
    try:
        with open(f, newline="", encoding="utf-8") as fh:
            rows = list(csv.reader(fh))
        if rows and rows[0][:4] != ["Id", "Name", "Date", "Time"]:
            raise ValueError("unexpected attendance CSV header")
        return [row for row in rows[1:] if len(row) >= 4]
    except (OSError, csv.Error, UnicodeError, ValueError) as exc:
        raise ValueError(f"Could not read today's attendance file: {exc}") from exc


def mark_attendance(sid: int, name: str) -> bool:
    """Write a row once per person per day. Returns True if newly marked."""
    if any(r[0] == str(sid) for r in read_attendance()):
        return False
    f = today_file()
    new_file = not f.exists()
    now = datetime.now()
    with open(f, "a", newline="", encoding="utf-8") as fh:
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

        try:
            self.detector = cv2.CascadeClassifier(CASCADE)
            if self.detector.empty():
                raise cv2.error("Could not load the face detector cascade.")
        except cv2.error as exc:
            messagebox.showerror("Face detector error", f"Could not load face detector: {exc}")
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
        try:
            total = len(load_students())
        except ValueError as exc:
            self.status.config(text="Student records could not be loaded.")
            messagebox.showerror("Student data error", str(exc))
            return
        self.status.config(text=text or f"Total registrations: {total}")

    def clear_form(self):
        self.id_entry.delete(0, "end")
        self.name_entry.delete(0, "end")
        self.update_status()

    def refresh_table(self):
        try:
            rows = read_attendance()
        except ValueError as exc:
            messagebox.showerror("Attendance data error", str(exc))
            return
        self.tv.delete(*self.tv.get_children())
        for sid, name, date, t in reversed(rows):
            self.tv.insert("", "end", text=sid, values=(name, date, t))

    # ---------- password
    def _ensure_password(self):
        if PASSWORD_FILE.exists():
            return
        pw = sd.askstring("Set password", "No password found. Create one:", show="*", parent=self.root)
        if pw:
            try:
                set_password(pw)
            except OSError as exc:
                return messagebox.showerror("Password error", f"Could not save password: {exc}")
            messagebox.showinfo("Password", "Password saved.")

    def change_password(self):
        if not PASSWORD_FILE.exists():
            return self._ensure_password()
        old = sd.askstring("Change password", "Old password:", show="*", parent=self.root)
        if old is None:
            return
        if not check_password(old):
            return messagebox.showerror("Wrong password", "Old password is incorrect or password data is unreadable.")
        new = sd.askstring("Change password", "New password:", show="*", parent=self.root)
        if not new:
            return
        if sd.askstring("Change password", "Confirm new password:", show="*", parent=self.root) != new:
            return messagebox.showerror("Error", "Passwords do not match.")
        try:
            set_password(new)
        except OSError as exc:
            return messagebox.showerror("Password error", f"Could not save password: {exc}")
        messagebox.showinfo("Done", "Password changed.")

    # ---------- registration
    def take_images(self):
        sid_text, name = self.id_entry.get().strip(), self.name_entry.get().strip()
        if not sid_text.isdigit():
            return messagebox.showerror("Invalid ID", "ID must be a number.")
        if not name.replace(" ", "").isalpha():
            return messagebox.showerror("Invalid name", "Name may contain only letters and spaces.")
        sid = int(sid_text)
        try:
            if sid in load_students():
                return messagebox.showerror("Duplicate", f"ID {sid} is already registered.")
        except ValueError as exc:
            return messagebox.showerror("Student data error", str(exc))

        try:
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                cam.release()
                return messagebox.showerror("Camera", "Could not open the camera.")
        except cv2.error as exc:
            return messagebox.showerror("Camera error", f"Could not initialize the camera: {exc}")

        count = 0
        capture_error = None
        try:
            while count < SAMPLES_PER_PERSON:
                ok, frame = cam.read()
                if not ok or frame is None:
                    capture_error = "The camera stopped returning frames."
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector.detectMultiScale(gray, 1.3, 5)
                if len(faces):
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face only
                    image_path = TRAIN_DIR / f"{sid}_{count + 1}.jpg"
                    if not cv2.imwrite(str(image_path), gray[y:y + h, x:x + w]):
                        capture_error = f"Could not save captured image: {image_path.name}"
                        break
                    count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{count}/{SAMPLES_PER_PERSON}  (q = quit)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Taking Images", frame)
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break
        except (cv2.error, OSError) as exc:
            capture_error = str(exc)
        finally:
            cam.release()
            cv2.destroyAllWindows()

        if capture_error:
            messagebox.showerror("Capture error", capture_error)
            if count == 0:
                return
        if count == 0:
            return messagebox.showwarning("No face", "No face was captured. Please try again.")
        try:
            add_student(sid, name)
        except OSError as exc:
            return messagebox.showerror("Student data error", f"Images were captured, but the student record could not be saved: {exc}")
        self.update_status(f"Captured {count} images for ID {sid}. Now click 'Save Profile'.")

    # ---------- training
    def train_with_password(self):
        pw = sd.askstring("Password", "Enter password:", show="*", parent=self.root)
        if pw is None:
            return
        if not PASSWORD_FILE.exists() or not check_password(pw):
            return messagebox.showerror("Wrong password", "Incorrect password or password data is unreadable.")
        self.train()

    def train(self):
        faces, ids = [], []
        try:
            for p in TRAIN_DIR.glob("*.jpg"):
                try:
                    sid = int(p.stem.split("_")[0])
                    with Image.open(p) as image:
                        face = np.array(image.convert("L"), "uint8")
                    if face.size == 0:
                        continue
                    ids.append(sid)
                    faces.append(face)
                except (ValueError, OSError):
                    continue
            if not faces:
                return messagebox.showwarning("No data", "No valid training images found. Please register someone first.")
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(ids))
            recognizer.write(str(MODEL_FILE))
        except (AttributeError, cv2.error, OSError, RuntimeError) as exc:
            return messagebox.showerror("Training error", f"Could not train or save the face model: {exc}")
        self.update_status(f"Profile saved. {len(set(ids))} people trained.")

    # ---------- attendance
    def take_attendance(self):
        if not MODEL_FILE.exists():
            return messagebox.showwarning("No model", "Please click 'Save Profile' first.")
        try:
            students = load_students()
        except ValueError as exc:
            return messagebox.showerror("Student data error", str(exc))
        if not students:
            return messagebox.showwarning("No students", "No registered students found.")

        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(str(MODEL_FILE))
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                cam.release()
                return messagebox.showerror("Camera", "Could not open the camera.")
        except (AttributeError, cv2.error, OSError, RuntimeError) as exc:
            return messagebox.showerror("Attendance setup error", f"Could not load the model or camera: {exc}")

        attendance_error = None
        try:
            while True:
                ok, frame = cam.read()
                if not ok or frame is None:
                    attendance_error = "The camera stopped returning frames."
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
        except (cv2.error, OSError, ValueError, RuntimeError) as exc:
            attendance_error = str(exc)
        finally:
            cam.release()
            cv2.destroyAllWindows()
        if attendance_error:
            messagebox.showerror("Attendance error", attendance_error)
        self.refresh_table()


if __name__ == "__main__":
    try:
        root = tk.Tk()
        App(root)
        root.mainloop()
    except (OSError, RuntimeError, tk.TclError) as exc:
        messagebox.showerror("Startup error", f"The attendance system could not start: {exc}")
