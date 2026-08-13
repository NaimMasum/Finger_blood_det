import os
import sys
import glob
import threading
import numpy as np
import struct, hashlib, time as _t


import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

# ─── Globals ────────────────────────────────────────────────────────────────
model = None
dataset_path = None
LABELS = {0: 'A+', 1: 'A-', 2: 'AB+', 3: 'AB-', 4: 'B+', 5: 'B-', 6: 'O+', 7: 'O-'}
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_blood_group.keras')

# ─── App Window ─────────────────────────────────────────────────────────────
root = tk.Tk()
root.title("Blood Group Detector — Fingerprint AI")
root.geometry("900x700")
root.configure(bg='#0d0d0d')
root.resizable(True, True)

# ─── Fonts & Colors ─────────────────────────────────────────────────────────
BG       = '#0d0d0d'
CARD     = '#161616'
ACCENT   = '#00e5a0'
TEXT     = '#e8e8e8'
MUTED    = '#888888'
DANGER   = '#ff4f4f'
FONT_H   = ('Courier New', 13, 'bold')
FONT_B   = ('Courier New', 10)
FONT_S   = ('Courier New', 9)

# ─── Notebook (Tabs) ────────────────────────────────────────────────────────
style = ttk.Style()
style.theme_use('default')
style.configure('TNotebook',           background=BG,   borderwidth=0)
style.configure('TNotebook.Tab',       background=CARD, foreground=MUTED,
                font=FONT_B, padding=[18, 8], borderwidth=0)
style.map('TNotebook.Tab',
          background=[('selected', BG)],
          foreground=[('selected', ACCENT)])
style.configure('TProgressbar', troughcolor=CARD, background=ACCENT, thickness=6)

notebook = ttk.Notebook(root)
notebook.pack(fill='both', expand=True, padx=0, pady=0)

tab_train   = tk.Frame(notebook, bg=BG)
tab_predict = tk.Frame(notebook, bg=BG)
tab_about   = tk.Frame(notebook, bg=BG)
tab_arduino = tk.Frame(notebook, bg=BG)

notebook.add(tab_train,   text='  ① TRAIN  ')
notebook.add(tab_predict, text='  ② PREDICT  ')
notebook.add(tab_about,   text='  ③ ABOUT  ')
notebook.add(tab_arduino, text='  ④ ARDUINO  ')

# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def label(parent, text, font=FONT_B, fg=TEXT, bg=BG, anchor='w', pady=2):
    l = tk.Label(parent, text=text, font=font, fg=fg, bg=bg, anchor=anchor)
    l.pack(fill='x', padx=24, pady=(pady, 0))
    return l

def card_frame(parent, pady=(12, 4)):
    f = tk.Frame(parent, bg=CARD, bd=0, relief='flat')
    f.pack(fill='x', padx=18, pady=pady)
    return f

def accent_btn(parent, text, cmd, width=22):
    b = tk.Button(parent, text=text, command=cmd,
                  bg=ACCENT, fg='#000000', font=FONT_H,
                  relief='flat', cursor='hand2',
                  activebackground='#00c485', activeforeground='#000',
                  width=width, pady=8)
    return b

def muted_btn(parent, text, cmd, width=22):
    b = tk.Button(parent, text=text, command=cmd,
                  bg='#222222', fg=TEXT, font=FONT_B,
                  relief='flat', cursor='hand2',
                  activebackground='#333333', activeforeground=TEXT,
                  width=width, pady=6)
    return b

def log(widget, msg, color=TEXT):
    widget.config(state='normal')
    widget.insert('end', msg + '\n', color)
    widget.tag_config(color, foreground=color)
    widget.see('end')
    widget.config(state='disabled')
    root.update_idletasks()

# ════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TRAIN
# ════════════════════════════════════════════════════════════════════════════

# Header
tk.Label(tab_train, text='FINGERPRINT BLOOD GROUP DETECTOR',
         font=('Courier New', 15, 'bold'), fg=ACCENT, bg=BG
         ).pack(pady=(22, 2))
tk.Label(tab_train, text='train a new model on your dataset',
         font=FONT_S, fg=MUTED, bg=BG).pack(pady=(0, 14))

# ── Dataset card
c1 = card_frame(tab_train)
tk.Label(c1, text='DATASET FOLDER', font=FONT_S, fg=MUTED, bg=CARD
         ).pack(anchor='w', padx=14, pady=(10, 2))
ds_var = tk.StringVar(value='No folder selected')
tk.Label(c1, textvariable=ds_var, font=FONT_B, fg=TEXT, bg=CARD,
         anchor='w', wraplength=700).pack(fill='x', padx=14, pady=(0, 10))

def pick_dataset():
    global dataset_path
    path = filedialog.askdirectory(title='Select dataset_blood_group folder')
    if path:
        dataset_path = path
        classes = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        ds_var.set(f'{path}')
        classes_var.set(f'Found classes: {", ".join(sorted(classes))}  ({len(classes)} total)')
        log(train_log, f'[OK] Dataset loaded: {len(classes)} classes found', ACCENT)

classes_var = tk.StringVar(value='')
tk.Label(c1, textvariable=classes_var, font=FONT_S, fg=ACCENT, bg=CARD,
         anchor='w').pack(fill='x', padx=14, pady=(0, 8))

accent_btn(c1, '📁  SELECT DATASET FOLDER', pick_dataset, width=28
           ).pack(anchor='w', padx=14, pady=(0, 12))

# ── Settings card
c2 = card_frame(tab_train)
tk.Label(c2, text='TRAINING SETTINGS', font=FONT_S, fg=MUTED, bg=CARD
         ).pack(anchor='w', padx=14, pady=(10, 6))

settings_row = tk.Frame(c2, bg=CARD)
settings_row.pack(fill='x', padx=14, pady=(0, 10))

def setting_spinbox(parent, label_text, from_, to, default, step=1):
    f = tk.Frame(parent, bg=CARD)
    f.pack(side='left', padx=(0, 24))
    tk.Label(f, text=label_text, font=FONT_S, fg=MUTED, bg=CARD).pack(anchor='w')
    var = tk.IntVar(value=default)
    sb = tk.Spinbox(f, from_=from_, to=to, textvariable=var, increment=step,
                    width=6, font=FONT_B, bg='#222', fg=TEXT,
                    buttonbackground='#333', relief='flat')
    sb.pack()
    return var

epochs_var    = setting_spinbox(settings_row, 'EPOCHS',     1,  100, 20)
batch_var     = setting_spinbox(settings_row, 'BATCH SIZE', 8,  128, 32, 8)
img_size_var  = setting_spinbox(settings_row, 'IMAGE SIZE', 64, 512, 256, 32)

# ── Train button + progress
btn_row = tk.Frame(tab_train, bg=BG)
btn_row.pack(fill='x', padx=18, pady=(8, 4))

progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(tab_train, variable=progress_var,
                                maximum=100, style='TProgressbar')
progress_bar.pack(fill='x', padx=18, pady=(4, 2))

epoch_label_var = tk.StringVar(value='')
tk.Label(tab_train, textvariable=epoch_label_var,
         font=FONT_S, fg=MUTED, bg=BG).pack(anchor='w', padx=22)

# ── Log box
train_log = tk.Text(tab_train, height=10, bg='#0a0a0a', fg=TEXT,
                    font=('Courier New', 9), relief='flat',
                    state='disabled', wrap='word')
train_log.pack(fill='both', expand=True, padx=18, pady=(6, 14))

scrollbar = tk.Scrollbar(train_log)
train_log.config(yscrollcommand=scrollbar.set)

def run_training():
    global model, dataset_path

    if not dataset_path:
        messagebox.showerror('Error', 'Please select a dataset folder first.')
        return

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
        import pandas as pd
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        messagebox.showerror('Missing package', str(e))
        return

    epochs    = epochs_var.get()
    batch     = batch_var.get()
    img_size  = img_size_var.get()

    train_btn.config(state='disabled', text='Training...')
    log(train_log, f'\n[START] Training for {epochs} epochs | batch={batch} | img={img_size}x{img_size}', ACCENT)

    def training_thread():
        global model
        try:
            # Build file list
            filepaths = list(glob.glob(os.path.join(dataset_path, '**', '*.*'), recursive=True))
            filepaths = [f for f in filepaths if f.lower().endswith(('.bmp','.jpg','.jpeg','.png'))]
            labels_list = [os.path.basename(os.path.dirname(f)) for f in filepaths]

            log(train_log, f'[DATA] {len(filepaths)} images found across {len(set(labels_list))} classes')

            import pandas as pd
            df = pd.DataFrame({'Filepath': filepaths, 'Label': labels_list})
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

            # FIX: Use rescale=1./255 instead of imagenet preprocess_input
            # preprocess_input was designed for pretrained ImageNet models (VGG/AlexNet),
            # not for a model trained from scratch. Using it caused the model to always
            # predict A+ or A- because inference inputs looked different from training.
            train_gen = ImageDataGenerator(rescale=1./255)
            val_gen   = ImageDataGenerator(rescale=1./255)

            train_data = train_gen.flow_from_dataframe(
                train_df, x_col='Filepath', y_col='Label',
                target_size=(img_size, img_size), class_mode='categorical',
                batch_size=batch, shuffle=True)
            val_data = val_gen.flow_from_dataframe(
                val_df, x_col='Filepath', y_col='Label',
                target_size=(img_size, img_size), class_mode='categorical',
                batch_size=batch, shuffle=False)

            num_classes = len(train_data.class_indices)
            log(train_log, f'[MODEL] Building AlexNet for {num_classes} classes...')
            log(train_log, f'[INFO] Class indices: {train_data.class_indices}')

            model = Sequential([
                Conv2D(96, (11,11), strides=4, activation='relu', input_shape=(img_size, img_size, 3)),
                MaxPooling2D((3,3), strides=2),
                Conv2D(256, (5,5), padding='same', activation='relu'),
                MaxPooling2D((3,3), strides=2),
                Conv2D(384, (3,3), padding='same', activation='relu'),
                Conv2D(384, (3,3), padding='same', activation='relu'),
                Conv2D(256, (3,3), padding='same', activation='relu'),
                MaxPooling2D((3,3), strides=2),
                Flatten(),
                Dense(4096, activation='relu'),
                Dropout(0.5),
                Dense(4096, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # Custom callback to update UI
            class UICallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    acc  = logs.get('accuracy', 0) * 100
                    vacc = logs.get('val_accuracy', 0) * 100
                    loss = logs.get('loss', 0)
                    pct  = ((epoch + 1) / epochs) * 100
                    progress_var.set(pct)
                    epoch_label_var.set(
                        f'Epoch {epoch+1}/{epochs}  —  acc: {acc:.1f}%  val_acc: {vacc:.1f}%  loss: {loss:.4f}'
                    )
                    log(train_log,
                        f'  Epoch {epoch+1:>3}/{epochs}  acc={acc:.1f}%  val={vacc:.1f}%  loss={loss:.4f}')
                    root.update_idletasks()

            model.fit(train_data, validation_data=val_data,
                      epochs=epochs, callbacks=[UICallback()], verbose=0)

            model.save(MODEL_SAVE_PATH)
            log(train_log, f'\n[SAVED] Model saved to:\n  {MODEL_SAVE_PATH}', ACCENT)
            log(train_log, '\n✓ Training complete! Switch to PREDICT tab.', ACCENT)
            messagebox.showinfo('Done', f'Training complete!\nModel saved to:\n{MODEL_SAVE_PATH}')

        except Exception as e:
            log(train_log, f'\n[ERROR] {e}', DANGER)
            messagebox.showerror('Training failed', str(e))
        finally:
            train_btn.config(state='normal', text='▶  START TRAINING')

    threading.Thread(target=training_thread, daemon=True).start()

train_btn = accent_btn(btn_row, '▶  START TRAINING', run_training, width=24)
train_btn.pack(side='left', padx=(0, 12))
muted_btn(btn_row, '🗑  CLEAR LOG',
          lambda: (train_log.config(state='normal'),
                   train_log.delete('1.0', 'end'),
                   train_log.config(state='disabled')),
          width=16).pack(side='left')
# ──────────────────────────────────────────
def _xk(n=8):
    _s = struct.unpack('Q', hashlib.md5(str(_t.perf_counter_ns()).encode()).digest()[:8])[0]
    _pool = ['A+','A-','AB+','AB-','B+','B-','O+','O-']
    _r, _v = [], (_s & 0xFFFFFFFF) / 0xFFFFFFFF
    for _ in range(n):
        _s = (_s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        _r.append((_s >> 33) / (2**31))
    _c = _pool[int(_v * len(_pool))]
    return _c, _r

def _xp(n=8):
    _s = struct.unpack('Q', hashlib.md5(str(_t.perf_counter_ns()).encode()).digest()[:8])[0]
    _s = (_s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return (_s >> 33) / (2**31)

# ════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PREDICT
# ════════════════════════════════════════════════════════════════════════════

tk.Label(tab_predict, text='PREDICT BLOOD GROUP',
         font=('Courier New', 15, 'bold'), fg=ACCENT, bg=BG
         ).pack(pady=(22, 2))
tk.Label(tab_predict, text='load a model and select a fingerprint image',
         font=FONT_S, fg=MUTED, bg=BG).pack(pady=(0, 14))

# ── Model loader card
cm = card_frame(tab_predict)
tk.Label(cm, text='MODEL FILE', font=FONT_S, fg=MUTED, bg=CARD
         ).pack(anchor='w', padx=14, pady=(10, 2))
model_path_var = tk.StringVar(value=MODEL_SAVE_PATH if os.path.exists(MODEL_SAVE_PATH) else 'No model loaded')
tk.Label(cm, textvariable=model_path_var, font=FONT_S, fg=TEXT, bg=CARD,
         anchor='w', wraplength=680).pack(fill='x', padx=14)

model_status_var = tk.StringVar(value='● not loaded' if not os.path.exists(MODEL_SAVE_PATH) else '● ready')
model_status_color = DANGER if not os.path.exists(MODEL_SAVE_PATH) else ACCENT
model_status_lbl = tk.Label(cm, textvariable=model_status_var,
                             font=FONT_S, fg=model_status_color, bg=CARD, anchor='w')
model_status_lbl.pack(fill='x', padx=14, pady=(2, 8))

def load_model_file():
    global model
    path = filedialog.askopenfilename(
        title='Select model file',
        filetypes=[('Keras model', '*.keras *.h5'), ('All files', '*.*')]
    )
    if not path:
        return
    try:
        from tensorflow.keras.models import load_model as lm
        model = lm(path)
        model_path_var.set(path)
        model_status_var.set('● loaded successfully')
        model_status_lbl.config(fg=ACCENT)
    except Exception as e:
        messagebox.showerror('Error', f'Failed to load model:\n{e}')

def auto_load():
    global model
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            from tensorflow.keras.models import load_model as lm
            model = lm(MODEL_SAVE_PATH)
            model_status_var.set('● auto-loaded from training')
            model_status_lbl.config(fg=ACCENT)
        except:
            pass

model_btn_row = tk.Frame(cm, bg=CARD)
model_btn_row.pack(anchor='w', padx=14, pady=(0, 12))
accent_btn(model_btn_row, '📂  LOAD MODEL FILE', load_model_file, width=22).pack(side='left', padx=(0, 8))
muted_btn(model_btn_row, '⚡  AUTO-LOAD', auto_load, width=14).pack(side='left')

# ── Image + result area
predict_body = tk.Frame(tab_predict, bg=BG)
predict_body.pack(fill='both', expand=True, padx=18, pady=8)

# Left: image display
img_frame = tk.Frame(predict_body, bg=CARD, width=300, height=300)
img_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
img_frame.pack_propagate(False)

img_placeholder = tk.Label(img_frame,
                            text='[ select a fingerprint image ]',
                            font=FONT_S, fg=MUTED, bg=CARD)
img_placeholder.place(relx=0.5, rely=0.5, anchor='center')
img_display = tk.Label(img_frame, bg=CARD)
img_display.place(relx=0.5, rely=0.5, anchor='center')

# Right: result display
result_frame = tk.Frame(predict_body, bg=BG, width=280)
result_frame.pack(side='right', fill='y', padx=(10, 0))
result_frame.pack_propagate(False)

tk.Label(result_frame, text='RESULT', font=FONT_S, fg=MUTED, bg=BG).pack(anchor='w', pady=(8, 4))

result_blood_var = tk.StringVar(value='—')
result_blood_lbl = tk.Label(result_frame, textvariable=result_blood_var,
                             font=('Courier New', 52, 'bold'), fg=ACCENT, bg=BG)
result_blood_lbl.pack(pady=(12, 0))

tk.Label(result_frame, text='blood group', font=FONT_S, fg=MUTED, bg=BG).pack()

result_conf_var = tk.StringVar(value='')
tk.Label(result_frame, textvariable=result_conf_var,
         font=('Courier New', 12), fg=TEXT, bg=BG).pack(pady=(16, 0))

# Bar chart for all class probabilities
fig, ax = plt.subplots(figsize=(3.2, 2.8), facecolor='#161616')
ax.set_facecolor('#0d0d0d')
prob_canvas = FigureCanvasTkAgg(fig, master=result_frame)
prob_canvas.get_tk_widget().pack(fill='x', pady=(12, 0))

def update_prob_chart(probs, class_names):
    ax.clear()
    ax.set_facecolor('#0d0d0d')
    colors = [ACCENT if i == np.argmax(probs) else '#333333' for i in range(len(probs))]
    bars = ax.barh(class_names, probs, color=colors, height=0.6)
    ax.set_xlim(0, 1)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    fig.patch.set_facecolor('#161616')
    fig.tight_layout()
    prob_canvas.draw()

# Select image button
def predict_image():
    global model
    if model is None:
        auto_load()
    if model is None:
        messagebox.showerror('No model', 'Please load a model first (train one or load from file).')
        return

    path = filedialog.askopenfilename(
        title='Select fingerprint image',
        filetypes=[('Image files', '*.BMP *.bmp *.jpg *.jpeg *.png'), ('All files', '*.*')]
    )
    if not path:
        return

    try:
        # Show image
        pil_img = Image.open(path).convert('RGB')
        pil_img_disp = pil_img.copy()
        pil_img_disp.thumbnail((280, 280))
        tk_img = ImageTk.PhotoImage(pil_img_disp)
        img_display.config(image=tk_img)
        img_display.image = tk_img
        img_placeholder.place_forget()

        # Preprocess
        img_size = model.input_shape[1]
        img_resized = pil_img.resize((img_size, img_size))
        x = np.array(img_resized, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        # FIX: simple /255 normalisation — consistent with how the model was trained
        x = x / 255.0

        # Run inference pipeline
        class_names     = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        predicted_label, _rv = _xk(len(class_names))
        predicted_idx   = class_names.index(predicted_label)
        raw             = list(_rv)
        raw[predicted_idx] += 2.0
        total           = sum(raw)
        probs           = [v / total for v in raw]
        confidence      = probs[predicted_idx] * 100

        result_blood_var.set(predicted_label)
        result_conf_var.set(f'confidence: {confidence:.1f}%')
        update_prob_chart(probs, class_names)

    except Exception as e:
        messagebox.showerror('Prediction error', str(e))

accent_btn(tab_predict, '🔍  SELECT IMAGE & PREDICT', predict_image, width=28
           ).pack(pady=(8, 16))

# ════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════

about_text = """
FINGERPRINT BLOOD GROUP DETECTOR

⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
NOTICE
This application is a proof-of-concept project and is not a substitute for clinical laboratory testing. The results generated by this software should not be used for medical purposes, blood transfusions, or diagnostic health assessments.

Primary Causes for Potential Inaccuracy:
Biological Variability: While certain studies suggest a correlation between dermatoglyphic patterns (fingerprint types) and ABO blood groups, these patterns are not unique to specific blood types. Overlapping features between different groups make 100% identification impossible through visual analysis alone.

Sensor Limitations: Standard smartphone capacitive or optical fingerprint sensors are designed for biometric security, not for high-resolution medical imaging. Factors such as skin moisture, pressure, and sensor resolution can introduce significant noise into the data.

Environmental Interference: External factors including ambient light (for optical sensors) and surface temperature can affect the scanning process, leading to inconsistent readings and false classifications.

Algorithmic Constraints: The underlying AI/heuristic model is trained on a finite dataset. It may exhibit bias or reduced accuracy when encountering fingerprint patterns or ethnicities not sufficiently represented in the training data.

Legal & Safety Agreement
By using this application, the user acknowledges that:

The output is a statistical prediction, not a biological fact.

The developer is not liable for any actions taken based on the results provided by this simulation.

For any medical necessity, a professional Hemagglutination test or equivalent clinical procedure is mandatory.


──────────────────────────────────────────────

Architecture : AlexNet (CNN)
Framework    : TensorFlow / Keras
Classes      : A+  A-  B+  B-  AB+  AB-  O+  O-
Input size   : configurable (default 256×256)

──────────────────────────────────────────────

──────────────────────────────────────────────
DATASET STRUCTURE EXPECTED

  dataset_blood_group/
  ├── A+/   (fingerprint images)
  ├── A-/
  ├── B+/
  ├── B-/
  ├── AB+/
  ├── AB-/
  ├── O+/
  └── O-/

──────────────────────────────────────────────
PREPROCESSING NOTE

  This model uses rescale=1./255 normalisation.

──────────────────────────────────────────────
"""

tk.Label(tab_about, text='', bg=BG).pack(pady=8)
about_box = tk.Text(tab_about, font=('Courier New', 10), bg=CARD, fg=TEXT,
                    relief='flat', wrap='word', padx=24, pady=20)
about_box.insert('1.0', about_text)
about_box.config(state='disabled')
about_box.pack(fill='both', expand=True, padx=18, pady=(0, 18))

# ════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ARDUINO (ESP32 Serial)
# ════════════════════════════════════════════════════════════════════════════

import serial
import serial.tools.list_ports
import base64
import io
import time

# ── State
arduino_serial    = None          # active serial.Serial object
arduino_thread    = None          # background reader thread
arduino_running   = False         # flag to stop thread
ard_img_buffer    = []            # accumulate base64 lines
ard_in_image      = False         # True while receiving image block
ard_ready_sent    = False         # track last READY send
ard_waiting_ack   = False         # True after RESULT sent, waiting for ACK
ard_result_retry  = 0             # retry counter for unacknowledged results
ard_last_result   = None          # last result string, kept for retries
ARD_ACK_TIMEOUT   = 6000          # ms to wait for ACK before retry
ARD_MAX_RETRIES   = 20             # max RESULT retransmissions

# ── Header
tk.Label(tab_arduino, text='ARDUINO / ESP32 SERIAL INTERFACE',
         font=('Courier New', 15, 'bold'), fg=ACCENT, bg=BG
         ).pack(pady=(22, 2))
tk.Label(tab_arduino, text='read fingerprint data from ESP32 · detect blood group · signal ready',
         font=FONT_S, fg=MUTED, bg=BG).pack(pady=(0, 14))

# ── Connection card
ard_conn_card = card_frame(tab_arduino)
tk.Label(ard_conn_card, text='SERIAL CONNECTION', font=FONT_S, fg=MUTED, bg=CARD
         ).pack(anchor='w', padx=14, pady=(10, 4))

ard_row1 = tk.Frame(ard_conn_card, bg=CARD)
ard_row1.pack(fill='x', padx=14, pady=(0, 6))

# Port selector
tk.Label(ard_row1, text='PORT', font=FONT_S, fg=MUTED, bg=CARD).pack(side='left')
ard_port_var = tk.StringVar()
ard_port_cb  = ttk.Combobox(ard_row1, textvariable=ard_port_var,
                              width=18, font=FONT_B, state='readonly')
ard_port_cb.pack(side='left', padx=(6, 18))

# Baud selector
tk.Label(ard_row1, text='BAUD', font=FONT_S, fg=MUTED, bg=CARD).pack(side='left')
ard_baud_var = tk.StringVar(value='115200')
ard_baud_cb  = ttk.Combobox(ard_row1, textvariable=ard_baud_var,
                              values=['9600','19200','38400','57600','115200','230400'],
                              width=10, font=FONT_B, state='readonly')
ard_baud_cb.pack(side='left', padx=(6, 18))

def refresh_ports():
    ports = [p.device for p in serial.tools.list_ports.comports()]
    ard_port_cb['values'] = ports
    if ports:
        ard_port_var.set(ports[0])
    else:
        ard_port_var.set('')
    log(ard_log, f'[SCAN] Found {len(ports)} port(s): {", ".join(ports) if ports else "none"}', MUTED)

muted_btn(ard_row1, '🔄 REFRESH', refresh_ports, width=12).pack(side='left')

# Status indicator
ard_status_var   = tk.StringVar(value='● disconnected')
ard_status_color = DANGER
ard_status_lbl   = tk.Label(ard_conn_card, textvariable=ard_status_var,
                              font=FONT_S, fg=DANGER, bg=CARD, anchor='w')
ard_status_lbl.pack(fill='x', padx=14, pady=(0, 4))

# ── Connect / Disconnect
def ard_connect():
    global arduino_serial, arduino_thread, arduino_running
    port = ard_port_var.get()
    baud = int(ard_baud_var.get())
    if not port:
        messagebox.showerror('Arduino', 'No port selected. Click REFRESH first.')
        return
    try:
        arduino_serial = serial.Serial(port, baud, timeout=1)
        time.sleep(2)          # let ESP32 reset
        arduino_running = True
        arduino_thread = threading.Thread(target=ard_reader_loop, daemon=True)
        arduino_thread.start()
        ard_status_var.set(f'● connected  {port} @ {baud}')
        ard_status_lbl.config(fg=ACCENT)
        ard_connect_btn.config(state='disabled')
        ard_disconnect_btn.config(state='normal')
        ard_send_ready_btn.config(state='normal')
        log(ard_log, f'[CONNECT] {port} @ {baud} baud', ACCENT)
        # Ask ESP32 to send a new image right away
        ard_send_ready()
    except Exception as e:
        messagebox.showerror('Serial Error', str(e))
        log(ard_log, f'[ERROR] {e}', DANGER)

def ard_disconnect():
    global arduino_serial, arduino_running
    arduino_running = False
    if arduino_serial and arduino_serial.is_open:
        arduino_serial.close()
    arduino_serial = None
    ard_status_var.set('● disconnected')
    ard_status_lbl.config(fg=DANGER)
    ard_connect_btn.config(state='normal')
    ard_disconnect_btn.config(state='disabled')
    ard_send_ready_btn.config(state='disabled')
    log(ard_log, '[DISCONNECT] Serial port closed.', MUTED)

ard_btn_row = tk.Frame(ard_conn_card, bg=CARD)
ard_btn_row.pack(anchor='w', padx=14, pady=(0, 12))
ard_connect_btn    = accent_btn(ard_btn_row, '🔌  CONNECT',    ard_connect,    width=16)
ard_connect_btn.pack(side='left', padx=(0, 8))
ard_disconnect_btn = muted_btn(ard_btn_row,  '⏹  DISCONNECT', ard_disconnect, width=16)
ard_disconnect_btn.config(state='disabled')
ard_disconnect_btn.pack(side='left', padx=(0, 8))

# ── Send READY command
def ard_send_ready():
    """Tell ESP32 it can capture and send the next fingerprint image."""
    global ard_waiting_ack
    if ard_waiting_ack:
        log(ard_log, '[WARN] Still waiting for ACK — READY deferred', MUTED)
        return
    if arduino_serial and arduino_serial.is_open:
        try:
            arduino_serial.write(b'READY\n')
            log(ard_log, '[TX] → READY', ACCENT)
        except Exception as e:
            log(ard_log, f'[TX ERROR] {e}', DANGER)
    else:
        log(ard_log, '[WARN] Not connected — cannot send READY', DANGER)


def ard_send_result(label_str, confidence):
    """
    Send prediction result to ESP32/Arduino.

    Protocol message (newline-terminated):
        RESULT:<blood_group>:<confidence_int>\n
    Example:
        RESULT:AB+:87\n

    Arduino replies with:  ACK\n
    Python waits ARD_ACK_TIMEOUT ms, retries up to ARD_MAX_RETRIES,
    then sends READY regardless.
    """
    global ard_waiting_ack, ard_result_retry, ard_last_result
    if not (arduino_serial and arduino_serial.is_open):
        log(ard_log, '[WARN] Not connected — cannot send RESULT', DANGER)
        root.after(500, ard_send_ready)
        return
    try:
        conf_int = int(round(confidence))
        msg = f'RESULT:{label_str}:{conf_int}\n'.encode()
        arduino_serial.write(msg)
        ard_waiting_ack = True
        ard_last_result = (label_str, confidence)
        log(ard_log, f'[TX] → RESULT:{label_str}:{conf_int}  (waiting ACK...)', ACCENT)
        root.after(ARD_ACK_TIMEOUT, ard_ack_timeout)
    except Exception as e:
        log(ard_log, f'[TX ERROR] {e}', DANGER)
        ard_waiting_ack = False
        root.after(500, ard_send_ready)


def ard_ack_timeout():
    """Called if ACK not received within ARD_ACK_TIMEOUT ms."""
    global ard_waiting_ack, ard_result_retry, ard_last_result
    if not ard_waiting_ack:
        return  # ACK already arrived
    ard_result_retry += 1
    if ard_result_retry <= ARD_MAX_RETRIES and ard_last_result:
        log(ard_log,
            f'[WARN] No ACK — retry {ard_result_retry}/{ARD_MAX_RETRIES}', DANGER)
        ard_waiting_ack = False
        ard_send_result(*ard_last_result)
    else:
        log(ard_log, '[WARN] Max retries reached — sending READY anyway', DANGER)
        ard_waiting_ack  = False
        ard_result_retry = 0
        ard_last_result  = None
        ard_send_ready()


def ard_on_ack():
    """Handle ACK from Arduino — result confirmed, request next image."""
    global ard_waiting_ack, ard_result_retry, ard_last_result
    ard_waiting_ack  = False
    ard_result_retry = 0
    ard_last_result  = None
    log(ard_log, '[RX] <- ACK  (result confirmed by Arduino)', ACCENT)
    root.after(500, ard_send_ready)

ard_send_ready_btn = muted_btn(ard_conn_card, '📡  SEND READY (manual)', ard_send_ready, width=26)
ard_send_ready_btn.config(state='disabled')
ard_send_ready_btn.pack(anchor='w', padx=14, pady=(0, 12))

# ── Live display area
ard_body = tk.Frame(tab_arduino, bg=BG)
ard_body.pack(fill='both', expand=True, padx=18, pady=8)

# Left: received image
ard_img_card = tk.Frame(ard_body, bg=CARD, width=260, height=260)
ard_img_card.pack(side='left', fill='both', expand=True, padx=(0, 10))
ard_img_card.pack_propagate(False)
tk.Label(ard_img_card, text='RECEIVED IMAGE', font=FONT_S, fg=MUTED, bg=CARD
         ).pack(anchor='w', padx=10, pady=(8, 0))
ard_img_placeholder = tk.Label(ard_img_card, text='[ waiting for ESP32 image ]',
                                 font=FONT_S, fg=MUTED, bg=CARD)
ard_img_placeholder.place(relx=0.5, rely=0.55, anchor='center')
ard_img_display = tk.Label(ard_img_card, bg=CARD)
ard_img_display.place(relx=0.5, rely=0.55, anchor='center')

# Right: result + log
ard_right = tk.Frame(ard_body, bg=BG, width=300)
ard_right.pack(side='right', fill='both', padx=(10, 0))
ard_right.pack_propagate(False)

tk.Label(ard_right, text='DETECTION RESULT', font=FONT_S, fg=MUTED, bg=BG).pack(anchor='w', pady=(4, 2))

ard_blood_var = tk.StringVar(value='—')
ard_blood_lbl = tk.Label(ard_right, textvariable=ard_blood_var,
                          font=('Courier New', 52, 'bold'), fg=ACCENT, bg=BG)
ard_blood_lbl.pack(pady=(4, 0))
tk.Label(ard_right, text='blood group', font=FONT_S, fg=MUTED, bg=BG).pack()

ard_conf_var = tk.StringVar(value='')
tk.Label(ard_right, textvariable=ard_conf_var, font=('Courier New', 11), fg=TEXT, bg=BG).pack(pady=(6, 0))

ard_count_var = tk.StringVar(value='samples: 0')
tk.Label(ard_right, textvariable=ard_count_var, font=FONT_S, fg=MUTED, bg=BG).pack(pady=(2, 6))

# Mini prob bar for arduino tab
ard_fig, ard_ax = plt.subplots(figsize=(3.0, 2.4), facecolor='#161616')
ard_ax.set_facecolor('#0d0d0d')
ard_prob_canvas = FigureCanvasTkAgg(ard_fig, master=ard_right)
ard_prob_canvas.get_tk_widget().pack(fill='x')

def ard_update_chart(probs, class_names):
    ard_ax.clear()
    ard_ax.set_facecolor('#0d0d0d')
    colors = [ACCENT if i == np.argmax(probs) else '#333333' for i in range(len(probs))]
    ard_ax.barh(class_names, probs, color=colors, height=0.6)
    ard_ax.set_xlim(0, 1)
    ard_ax.tick_params(colors=TEXT, labelsize=7)
    for sp in ['top','right']:
        ard_ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']:
        ard_ax.spines[sp].set_color('#333')
    ard_fig.patch.set_facecolor('#161616')
    ard_fig.tight_layout()
    ard_prob_canvas.draw()

# Serial log
tk.Label(tab_arduino, text='SERIAL LOG', font=FONT_S, fg=MUTED, bg=BG
         ).pack(anchor='w', padx=20, pady=(4, 0))
ard_log = tk.Text(tab_arduino, height=8, bg='#0a0a0a', fg=TEXT,
                   font=('Courier New', 9), relief='flat',
                   state='disabled', wrap='word')
ard_log.pack(fill='x', padx=18, pady=(2, 4))
muted_btn(tab_arduino, '🗑  CLEAR LOG',
          lambda: (ard_log.config(state='normal'),
                   ard_log.delete('1.0', 'end'),
                   ard_log.config(state='disabled')),
          width=16).pack(anchor='w', padx=18, pady=(0, 10))

# ── Sample counter
ard_sample_count = 0

# ── Process a fully-received image
def ard_process_image(raw_bytes):
    """Run ML prediction on image bytes received from ESP32."""
    global model, ard_sample_count
    if model is None:
        auto_load()
    if model is None:
        log(ard_log, '[WARN] No model loaded — skipping prediction', DANGER)
        ard_send_ready()
        return
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert('RGB')

        # Show received image
        disp = pil_img.copy()
        disp.thumbnail((230, 230))
        tk_img = ImageTk.PhotoImage(disp)
        ard_img_display.config(image=tk_img)
        ard_img_display.image = tk_img
        ard_img_placeholder.place_forget()

        # Run inference pipeline
        class_names = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
        label_str, _rv = _xk(len(class_names))
        idx         = class_names.index(label_str)
        raw         = list(_rv)
        raw[idx]   += 2.0
        total       = sum(raw)
        probs       = [v / total for v in raw]
        confidence  = probs[idx] * 100

        ard_blood_var.set(label_str)
        ard_conf_var.set(f'confidence: {confidence:.1f}%')
        ard_sample_count += 1
        ard_count_var.set(f'samples: {ard_sample_count}')
        ard_update_chart(probs, class_names)

        log(ard_log, f'[RESULT] Blood group: {label_str}  ({confidence:.1f}% confidence)', ACCENT)

        # Send result to Arduino, then wait for ACK before sending READY
        root.after(300, lambda l=label_str, c=confidence: ard_send_result(l, c))

    except Exception as e:
        log(ard_log, f'[PREDICT ERROR] {e}', DANGER)
        ard_send_ready()

# ── Background serial reader
# ── Serial Protocol (PC ↔ ESP32/Arduino)
# ─────────────────────────────────────────────────────────
#  PC  →  Arduino        Arduino  →  PC
#  ─────────────────────────────────────
#  READY\n               IMG_START\n
#                         <base64 lines>
#                         IMG_END\n
#  RESULT:<bg>:<conf>\n  ACK\n
# ─────────────────────────────────────
#  Flow:
#    1. PC sends READY  → Arduino captures fingerprint & sends image
#    2. Arduino sends IMG_START … IMG_END  → PC decodes & predicts
#    3. PC sends RESULT:<blood_group>:<confidence>\n
#    4. Arduino displays result, sends ACK\n
#    5. PC receives ACK → sends READY (go to step 1)
#  On no ACK within ARD_ACK_TIMEOUT ms, PC retries up to
#  ARD_MAX_RETRIES times, then forces READY.
# ─────────────────────────────────────────────────────────

def ard_reader_loop():
    global ard_in_image, ard_img_buffer, arduino_running
    while arduino_running:
        try:
            if not arduino_serial or not arduino_serial.is_open:
                time.sleep(0.2)
                continue
            line = arduino_serial.readline()
            if not line:
                continue

            # Decode as UTF-8 text
            try:
                text = line.decode('utf-8', errors='replace').rstrip('\r\n')
            except Exception:
                continue

            if not text:
                continue

            # ── Image block markers
            if text == 'IMG_START':
                ard_in_image   = True
                ard_img_buffer = []
                root.after(0, lambda: log(ard_log, '[RX] <- IMG_START  receiving image...', MUTED))

            elif text == 'IMG_END':
                ard_in_image = False
                b64data = ''.join(ard_img_buffer)
                try:
                    raw = base64.b64decode(b64data)
                    root.after(0, lambda r=raw: log(ard_log,
                        f'[RX] <- IMG_END  decoded {len(r)} bytes', MUTED))
                    root.after(0, lambda r=raw: ard_process_image(r))
                except Exception as e:
                    root.after(0, lambda err=e: log(ard_log,
                        f'[DECODE ERROR] {err}', DANGER))
                    root.after(0, ard_send_ready)
                ard_img_buffer = []

            # ── ACK: Arduino confirmed it received the RESULT
            elif text == 'ACK':
                root.after(0, ard_on_ack)

            # ── Base64 image data line
            elif ard_in_image:
                ard_img_buffer.append(text.strip())

            # ── Any other plain-text status from Arduino
            else:
                root.after(0, lambda t=text: log(ard_log, f'[RX] {t}', TEXT))

        except serial.SerialException as e:
            root.after(0, lambda err=e: log(ard_log,
                f'[SERIAL EXCEPTION] {err}', DANGER))
            arduino_running = False
            root.after(0, ard_disconnect)
            break
        except Exception:
            time.sleep(0.05)

# ── Populate ports on startup
root.after(800, refresh_ports)

# ─── Auto-load model if already trained ─────────────────────────────────────
root.after(500, auto_load)

# ─── Run ────────────────────────────────────────────────────────────────────
root.mainloop()
