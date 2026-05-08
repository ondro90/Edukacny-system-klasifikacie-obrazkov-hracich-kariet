from __future__ import annotations # Moderná syntax typov (napr. X | None) aj na staršom Pythone

import os
import io
import sys
import json
import queue
import threading
import traceback
import numpy as np
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk
from tkinter import messagebox, ttk
from tkinter import filedialog

# trochu stlmíme TensorFlow výpisy
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.metrics import precision_score, recall_score, f1_score

# Tento helper rieši problém s cestami po zabalení programu do EXE
def get_resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = Path(__file__).resolve().parent

    return Path(base_path) / relative_path

# ============================================================
# 1. ZÁKLADNÉ NASTAVENIA
# ============================================================
# Tu sú všetky hlavné cesty a čísla.
# ============================================================

# priečinok kde je tento .py súbor
ROOT = Path(__file__).resolve().parent

# dataset
DATASET = ROOT / "dataset"

# priečinok kde sa bude ukladať model a grafy
MODELS = ROOT / "models"

# súbor s modelom
MODEL_FILE = MODELS / "model_best.keras"

# súbor s názvami tried
CLASSES_FILE = MODELS / "class_names.json"

# súbory s grafmi
ACC_PLOT_FILE = MODELS / "accuracy_plot.png"
LOSS_PLOT_FILE = MODELS / "loss_plot.png"
CM_PLOT_FILE = MODELS / "confusion_matrix.png"
METRICS_PLOT_FILE = MODELS / "metrics_plot.png"

# veľkosť obrázka pre model
IMG_H = 224
IMG_W = 224

# veľkosť preview obrázka v GUI
PREVIEW_SIZE = (340, 340)

# koľko top výsledkov chceme vypísať
TOP_K = 3

# seed kvôli tomu aby to bolo opakovateľné
SEED = 42

# defaultné hodnoty
DEFAULT_EPOCHS = 10
DEFAULT_BATCH = 32
DEFAULT_DROPOUT = 0.2

DROPOUT_VALUES = [0.1, 0.2, 0.3, 0.4]

# povolené prípony obrázkov
OK_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ak priečinok neexistuje, vytvorí sa
MODELS.mkdir(exist_ok=True)


# ============================================================
# 2. POMOCNÉ FUNKCIE
# ============================================================
# Tu sú menšie pomocné veci.
# ============================================================

def log_write(log_q: queue.Queue[str] | None, text: str) -> None:
    """
    Zapíše text do GUI logu.
    Keď GUI fronta nie je, vypíše text len do konzoly.
    """
    if log_q is not None:
        log_q.put(text)
    else:
        print(text)


class TrainCallback(keras.callbacks.Callback):
    """
    Toto je náš vlastný callback počas tréningu.

    Na čo je dobrý:
    - píše info do GUI
    - sleduje STOP tlačidlo
    - po epoche vypíše accuracy a loss
    """
    def __init__(
        self,
        log_q: queue.Queue[str],
        all_epochs: int,
        stop_event: threading.Event | None = None,
    ):
        super().__init__()

        # sem budeme písať logy
        self.log_q = log_q

        # celkový počet epoch
        self.all_epochs = all_epochs

        # event pre stop tréningu
        self.stop_event = stop_event

        # aby sa správa o stopnutí neposielala stále dokola
        self.stop_msg_sent = False

    def on_train_begin(self, logs=None):
        # toto sa spustí na začiatku tréningu
        self.log_q.put("=== MODEL: začína tréning ===")

    def on_epoch_begin(self, epoch, logs=None):
        # toto sa spustí na začiatku každej epochy
        self.log_q.put(f"Epoch {epoch + 1}/{self.all_epochs} - začiatok")

    def check_stop(self) -> None:
        """
        Tu sa kontroluje, či user stlačil STOP.
        Keď áno, nastavíme model.stop_training = True
        """
        if self.stop_event is not None and self.stop_event.is_set():
            self.model.stop_training = True

            if not self.stop_msg_sent:
                self.log_q.put("=== MODEL: prišla požiadavka na STOP ===")
                self.stop_msg_sent = True

    def on_train_batch_end(self, batch, logs=None):
        # po každom batchi skontrolujeme stop
        self.check_stop()

    def on_epoch_end(self, epoch, logs=None):
        # po každej epoche si vytiahneme hlavné metriky
        logs = logs or {}

        acc = logs.get("accuracy", 0.0)
        loss = logs.get("loss", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)
        val_loss = logs.get("val_loss", 0.0)

        self.log_q.put(
            f"Epoch {epoch + 1}/{self.all_epochs} - "
            f"accuracy={acc:.4f} | "
            f"loss={loss:.4f} | "
            f"val_accuracy={val_acc:.4f} | "
            f"val_loss={val_loss:.4f}"
        )

        # a znova check stop
        self.check_stop()

    def on_train_end(self, logs=None):
        # keď tréning končí
        if self.stop_event is not None and self.stop_event.is_set():
            self.log_q.put("=== MODEL: tréning bol zastavený ===")
        else:
            self.log_q.put("=== MODEL: tréning skončil ===")


def save_classes(class_names: list[str]) -> None:
    """
    Uloží názvy tried do JSON súboru.
    Je to dôležité, aby sme po načítaní modelu vedeli čo je index 0, 1, 2...
    """
    CLASSES_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLASSES_FILE.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_classes() -> list[str]:
    """
    Načíta názvy tried zo súboru.
    """
    if not CLASSES_FILE.exists():
        raise FileNotFoundError("Súbor class_names.json neexistuje. Najprv natrénuj model.")

    return json.loads(CLASSES_FILE.read_text(encoding="utf-8"))


def normalize_cm(cm: np.ndarray) -> np.ndarray:
    """
    Spraví normalizovanú confusion matrix.
    Teda každý riadok prepočítame na pomer.
    """
    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float32)

    # aby sme nedelili nulou
    row_sums[row_sums == 0] = 1.0

    return cm.astype(np.float32) / row_sums


def save_acc_plot(history_data: dict) -> None:
    """
    Uloží accuracy graf do png.
    """
    train_acc = history_data.get("accuracy", [])
    val_acc = history_data.get("val_accuracy", [])
    xs = list(range(1, len(train_acc) + 1))

    fig = Figure(figsize=(8, 4.5), dpi=120)
    ax = fig.add_subplot(111)

    if train_acc:
        ax.plot(xs, train_acc, marker="o", label="Train accuracy")

    if val_acc:
        ax.plot(xs, val_acc, marker="o", label="Validation accuracy")

    ax.set_title("Accuracy počas tréningu")
    ax.set_xlabel("Epocha")
    ax.set_ylabel("Accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(ACC_PLOT_FILE)
    fig.clear()


def save_loss_plot(history_data: dict) -> None:
    """
    Uloží loss graf do png.
    """
    train_loss = history_data.get("loss", [])
    val_loss = history_data.get("val_loss", [])
    xs = list(range(1, len(train_loss) + 1))

    fig = Figure(figsize=(8, 4.5), dpi=120)
    ax = fig.add_subplot(111)

    if train_loss:
        ax.plot(xs, train_loss, marker="o", label="Train loss")

    if val_loss:
        ax.plot(xs, val_loss, marker="o", label="Validation loss")

    ax.set_title("Loss počas tréningu")
    ax.set_xlabel("Epocha")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(LOSS_PLOT_FILE)
    fig.clear()


def save_cm_plot(cm: np.ndarray, class_names: list[str]) -> None:
    """
    Uloží confusion matrix do png.
    """
    cm_norm = normalize_cm(cm)

    fig = Figure(figsize=(10, 9), dpi=120)
    ax = fig.add_subplot(111)

    img = ax.imshow(cm_norm, cmap="Blues")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    class_count = len(class_names)

    # keď je veľa tried, ukážeme iba každú druhú / štvrtú atď.
    step = 1
    if class_count > 20:
        step = 2
    if class_count > 40:
        step = 4
    if class_count > 80:
        step = 5

    tick_pos = np.arange(0, class_count, step)
    tick_labels = [class_names[i] for i in tick_pos]

    ax.set_title("Confusion matrix (normalizovaná)")
    ax.set_xlabel("Predikovaná trieda")
    ax.set_ylabel("Skutočná trieda")
    ax.set_xticks(tick_pos)
    ax.set_yticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(tick_labels, fontsize=6)

    fig.tight_layout()
    fig.savefig(CM_PLOT_FILE)
    fig.clear()


def save_metrics_plot(precision: float, recall: float, f1: float) -> None:
    """
    Uloží metrics (Precision/Recall/F1) bar chart do PNG.
    """
    values = [precision * 100, recall * 100, f1 * 100]
    labels = ["Precision", "Recall", "F1-score"]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    fig = Figure(figsize=(5.6, 4.2), dpi=100)
    ax = fig.add_subplot(111)

    bars = ax.bar(labels, values, color=colors)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.0f}%",
            ha='center',
            fontsize=10,
        )

    ax.set_title("Precision, Recall a F1-score")
    ax.set_ylabel("Hodnota v %")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(METRICS_PLOT_FILE)
    fig.clear()

def get_true_pred(model: keras.Model, test_ds) -> tuple[np.ndarray, np.ndarray]:
    """
    Z test datasetu zoberie:
    - skutočné labely
    - predikované labely
    """
    true_parts = []

    # tu zbierame skutočné labely po batchoch
    for _, labels in test_ds:
        true_parts.append(labels.numpy())

    true_labels = np.concatenate(true_parts, axis=0)

    # tu model predikuje celý test dataset
    preds = model.predict(test_ds, verbose=0)

    # zoberieme index najvyššej hodnoty
    pred_labels = np.argmax(preds, axis=1)

    return true_labels, pred_labels


# ============================================================
# 3. DATASET
# ============================================================
# Tu sa načítava train / valid / test dataset.
# Očakávaná štruktúra:
#
# dataset/
#   train/
#   valid/
#   test/
#
# a v nich priečinky jednotlivých tried.
# ============================================================

def load_data(batch_size: int, log_q: queue.Queue[str] | None = None):
    # cesty ku priečinkom
    train_dir = DATASET / "train"
    valid_dir = DATASET / "valid"
    test_dir = DATASET / "test"

    # toto len vypíšeme do logu, aby bolo vidieť kde sa hľadá dataset
    if log_q is not None:
        log_q.put(f"Kontrola datasetu: {train_dir}")
        log_q.put(f"Kontrola datasetu: {valid_dir}")
        log_q.put(f"Kontrola datasetu: {test_dir}")

    # ak niečo chýba, vyhodíme chybu
    if not train_dir.exists() or not valid_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Dataset nie je v správnej štruktúre. "
            "Skontroluj dataset/train, dataset/valid a dataset/test."
        )

    # train dáta
    # tu je shuffle=True, lebo počas tréningu chceme obrázky miešať
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_H, IMG_W),
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
    )

    # valid dáta
    # tu shuffle netreba
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        image_size=(IMG_H, IMG_W),
        batch_size=batch_size,
        shuffle=False,
    )

    # test dáta
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(IMG_H, IMG_W),
        batch_size=batch_size,
        shuffle=False,
    )

    # názvy tried vezmeme z train datasetu
    class_names = train_ds.class_names

    # prefetch trochu zrýchli načítanie batchov
    auto = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=auto)
    valid_ds = valid_ds.prefetch(buffer_size=auto)
    test_ds = test_ds.prefetch(buffer_size=auto)

    return train_ds, valid_ds, test_ds, class_names


# ============================================================
# 4. MODEL
# ============================================================
# Tu sa skladá CNN model.
# ============================================================

def make_model(class_count: int, dropout_rate: float = 0.2) -> keras.Model:
    # augmentácia
    # tu sa obrázky pri tréningu mierne menia
    aug = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.05),
        ],
        name="data_augmentation",
    )

    # samotný model
    model = keras.Sequential(
        [
            # vstup
            layers.Input(shape=(IMG_H, IMG_W, 3)),

            # augmentácia
            aug,

            # preškálovanie pixelov na 0-1
            layers.Rescaling(1.0 / 255.0),

            # prvý conv blok
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            # druhý conv blok
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            # tretí conv blok
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),

            # sploštenie
            layers.Flatten(),

            # dense vrstva
            layers.Dense(128, activation="relu"),

            # dropout
            layers.Dropout(dropout_rate),

            # výstup
            layers.Dense(class_count, activation="softmax"),
        ],
        name="cards_cnn",
    )

    # kompilácia modelu
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# 5. TRÉNING
# ============================================================
# Toto je hlavná časť:
# - načítanie dát
# - vytvorenie modelu
# - tréning
# - uloženie grafov
# - test
# - confusion matrix
# ============================================================

def train_net(
    epochs: int,
    batch_size: int,
    dropout_rate: float = 0.2,
    log_q: queue.Queue[str] | None = None,
    stop_event: threading.Event | None = None,
) -> dict:
    # tu len vypíšeme základné info
    log_write(log_q, "=== ŠTART TRÉNINGU MODELU ===")
    log_write(log_q, f"Nastavené epochs: {epochs}")
    log_write(log_q, f"Nastavený batch size: {batch_size}")
    log_write(log_q, f"Nastavený dropout: {dropout_rate}")

    # načítanie dát
    train_ds, valid_ds, test_ds, class_names = load_data(
        batch_size=batch_size,
        log_q=log_q,
    )

    # uložíme názvy tried
    save_classes(class_names)

    log_write(log_q, f"Počet tried: {len(class_names)}")
    log_write(log_q, f"Triedy: {class_names}")

    # vytvorenie modelu
    model = make_model(class_count=len(class_names), dropout_rate=dropout_rate)

    # model summary do GUI
    # tu si summary uložíme do bufferu a potom vypíšeme do logu
    buf = io.StringIO()
    model.summary(print_fn=lambda line: buf.write(line + "\n"))

    for line in buf.getvalue().splitlines():
        log_write(log_q, line)

    # callbacky
    cbs = [
        # uloží sa najlepší model podľa val_accuracy
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_FILE),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),

        # keď sa val_loss nezlepšuje, tréning sa zastaví skôr
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # náš callback na log a stop
    if log_q is not None:
        cbs.append(
            TrainCallback(
                log_q=log_q,
                all_epochs=epochs,
                stop_event=stop_event,
            )
        )

    # samotný tréning
    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        callbacks=cbs,
        verbose=0,
    )

    # history.history má accuracy, loss, val_accuracy, val_loss
    history_data = history.history

    train_acc = history_data.get("accuracy", [])
    train_loss = history_data.get("loss", [])
    val_acc = history_data.get("val_accuracy", [])
    val_loss = history_data.get("val_loss", [])

    # najlepšia epocha
    best_epoch = None
    best_train_acc = None
    best_train_loss = None
    best_val_acc = None
    best_val_loss = None

    if val_acc:
        best_epoch = int(np.argmax(val_acc))
        best_val_acc = float(val_acc[best_epoch])
        best_val_loss = float(val_loss[best_epoch]) if best_epoch < len(val_loss) else 0.0
        best_train_acc = float(train_acc[best_epoch]) if best_epoch < len(train_acc) else 0.0
        best_train_loss = float(train_loss[best_epoch]) if best_epoch < len(train_loss) else 0.0

        log_write(log_q, "")
        log_write(log_q, "=== NAJLEPŠÍ VÝSLEDOK TRÉNINGU ===")
        log_write(log_q, f"Najlepšia epocha: {best_epoch + 1}")
        log_write(log_q, f"Train accuracy: {best_train_acc * 100:.2f} %")
        log_write(log_q, f"Train loss: {best_train_loss:.4f}")
        log_write(log_q, f"Val accuracy: {best_val_acc * 100:.2f} %")
        log_write(log_q, f"Val loss: {best_val_loss:.4f}")

    # uloženie grafov accuracy a loss
    # tieto grafy chceme uložiť aj keď user dá STOP
    save_acc_plot(history_data)
    save_loss_plot(history_data)

    log_write(log_q, f"Uložený accuracy graf: {ACC_PLOT_FILE}")
    log_write(log_q, f"Uložený loss graf: {LOSS_PLOT_FILE}")

    # pripravíme výsledný slovník
    out = {
        "history": history_data,
        "class_names": class_names,
        "stopped": False,
        "best_epoch": best_epoch,
        "best_train_acc": best_train_acc,
        "best_train_loss": best_train_loss,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_accuracy": None,
        "test_loss": None,
        "cm": None,
    }

    # ak bol stop, test už nerobíme
    if stop_event is not None and stop_event.is_set():
        log_write(log_q, "Test dataset sa preskočil, lebo tréning bol zastavený.")
        log_write(log_q, "=== TRÉNING ZASTAVENÝ ===")
        out["stopped"] = True
        return out

    # test modelu
    test_loss_value, test_acc_value = model.evaluate(test_ds, verbose=0)

    log_write(log_q, f"Test accuracy: {test_acc_value * 100:.2f} %")
    log_write(log_q, f"Test loss: {test_loss_value:.4f}")

    # confusion matrix
    true_labels, pred_labels = get_true_pred(model, test_ds)

    cm = tf.math.confusion_matrix(
        true_labels,
        pred_labels,
        num_classes=len(class_names),
    ).numpy()

    save_cm_plot(cm, class_names)

    # výpočet precision, recall a f1 z test datasetu
    precision = precision_score(true_labels, pred_labels, average="macro", zero_division=0)
    recall = recall_score(true_labels, pred_labels, average="macro", zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

    log_write(log_q, f"Precision: {precision:.4f}")
    log_write(log_q, f"Recall: {recall:.4f}")
    log_write(log_q, f"F1-score: {f1:.4f}")

    out["precision"] = float(precision)
    out["recall"] = float(recall)
    out["f1"] = float(f1)

    save_metrics_plot(precision, recall, f1)
    log_write(log_q, f"Uložený metrics graf: {METRICS_PLOT_FILE}")

    log_write(log_q, f"Uložená confusion matrix: {CM_PLOT_FILE}")
    log_write(log_q, "=== TRÉNING DOKONČENÝ ===")

    out["test_accuracy"] = float(test_acc_value)
    out["test_loss"] = float(test_loss_value)
    out["cm"] = cm

    return out


# ============================================================
# 6. MODEL + OBRÁZKY PRE PREDIKCIU
# ============================================================
# Toto používame keď už model existuje a chceme s ním robiť.
# ============================================================

def load_model_and_classes() -> tuple[keras.Model, list[str]]:
    """
    Načíta model a názvy tried.
    """
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model neexistuje. Najprv ho natrénuj.")

    model = tf.keras.models.load_model(MODEL_FILE)
    class_names = load_classes()

    return model, class_names


def prepare_img(img_path: Path) -> np.ndarray:
    """
    Pripraví obrázok pre model.
    Tu nedelíme 255, lebo v modeli je Rescaling vrstva.
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_W, IMG_H))

    arr = np.asarray(img, dtype=np.float32)

    # pridáme batch dimenziu => (1, H, W, 3)
    arr = np.expand_dims(arr, axis=0)

    return arr


def load_test_images() -> list[Path]:
    """
    Nájde všetky obrázky v dataset/test.
    """
    test_dir = DATASET / "test"

    if not test_dir.exists():
        raise FileNotFoundError("Priečinok dataset/test neexistuje.")

    imgs: list[Path] = []

    # prejdeme všetky súbory vo vnútri test priečinka
    for p in test_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in OK_EXTENSIONS:
            imgs.append(p)

    imgs.sort()
    return imgs


# ============================================================
# 7. GUI TRIEDA
# ============================================================
# Tu je celé hlavné okno programu.
# Tu sa deje skoro všetko:
# - tlačidlá
# - tréning
# - načítanie obrázkov
# - klasifikácia
# - grafy
# ============================================================

class App:
    def __init__(self, root: tk.Tk) -> None:
        # základné nastavenie okna
        self.root = root
        self.root.title("Klasifikácia hracích kariet")
        self.root.geometry("1200x920")
        self.root.minsize(1100, 860)

        # tu si budeme držať model a triedy
        self.model: keras.Model | None = None
        self.class_names: list[str] = []

        # obrázok na preview
        self.preview_photo = None

        # sem sa ukladajú načítané test obrázky
        self.img_paths: list[Path] = []
        self.now_img_idx = -1

        # premenné pre tréning
        self.train_thread: threading.Thread | None = None
        self.train_error: str | None = None
        self.train_result: dict | None = None

        self.is_training = False
        self.stop_event = threading.Event()
        self.was_stopped = False

        # fronta na log
        self.log_q: queue.Queue[str] = queue.Queue()

        # posledné dobré hodnoty
        self.last_epochs = DEFAULT_EPOCHS
        self.last_batch = DEFAULT_BATCH
        self.last_dropout = DEFAULT_DROPOUT

        # tkinter premenné
        self.epochs_var = tk.StringVar(value=str(self.last_epochs))
        self.batch_var = tk.StringVar(value=str(self.last_batch))
        self.dropout_var = tk.StringVar(value=str(self.last_dropout))
        self.status_var = tk.StringVar(value="Hotovo, môžeš pokračovať.")
        self.img_info_var = tk.StringVar(value="/ 0 | -")
        self.jump_var = tk.StringVar(value="1")

        # postavíme GUI
        self.make_gui()

        # skúsime načítať model
        self.try_load_model()

        # skúsime načítať grafy
        self.load_saved_plots()

        # stále čítame log frontu
        self.read_log_q()

        # šípky na klávesnici = posun medzi obrázkami
        self.root.bind("<Left>", lambda event: self.prev_img())
        self.root.bind("<Right>", lambda event: self.next_img())

    def make_gui(self) -> None:

        # ============================================================
        # MENU APLIKÁCIE
        # ============================================================

        menubar = tk.Menu(self.root)

        # --- FILE MENU ---
        file_menu = tk.Menu(menubar, tearoff=0)

        file_menu.add_command(label="Nahrať obrázok (Upload)", command=self.upload_image_gui)
        file_menu.add_separator()
        file_menu.add_command(label="Zatvoriť", command=self.root.quit)

        menubar.add_cascade(label="Súbor", menu=file_menu)

        # --- HELP MENU ---
        help_menu = tk.Menu(menubar, tearoff=0)

        help_menu.add_command(
            label="Dokumentácia (GitHub)",
            command=lambda: self._open_url("https://github.com/ondro90/Edukacny-system-klasifikacie-obrazkov-hracich-kariet"),
        )

        help_menu.add_separator()

        help_menu.add_command(
            label="O aplikácii",
            command=self._show_about,
        )

        menubar.add_cascade(label="Pomocník", menu=help_menu)

        self.root.config(menu=menubar)

        # hlavný frame
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill="both", expand=True)

        # nadpis
        ttk.Label(
            main,
            text="Edukačný systém klasifikácie obrázkov hracích kariet",
            font=("Arial", 18, "bold"),
        ).pack(pady=(0, 6))

        # rámik na obrázok
        img_frame = ttk.LabelFrame(main, text="Ukážka obrázka", padding=12)
        img_frame.pack(fill="x", pady=(0, 12))

        self.img_label = ttk.Label(
            img_frame,
            text="Tu sa ukáže načítaný obrázok",
            anchor="center",
            relief="solid",
            padding=10,
            width=42,
        )
        self.img_label.pack(pady=(0, 10))

        # info riadok o obrázku
        info_row = ttk.Frame(img_frame)
        info_row.pack(pady=(0, 8))

        ttk.Label(info_row, text="Obrázok").pack(side="left", padx=(0, 4))

        self.jump_entry = ttk.Entry(
            info_row,
            textvariable=self.jump_var,
            width=4,
            justify="center",
        )
        self.jump_entry.pack(side="left")
        self.jump_entry.bind("<Return>", self.jump_img_event)

        ttk.Label(info_row, textvariable=self.img_info_var).pack(side="left", padx=(4, 0))

        # tlačidlá doľava / doprava
        nav = ttk.Frame(img_frame)
        nav.pack()

        self.prev_btn = ttk.Button(nav, text="← Doľava", command=self.prev_img)
        self.prev_btn.pack(side="left", padx=6)

        self.next_btn = ttk.Button(nav, text="Doprava →", command=self.next_img)
        self.next_btn.pack(side="left", padx=6)

        # riadok s nastaveniami a hlavnými tlačidlami
        btn_row = ttk.Frame(main)
        btn_row.pack(pady=(0, 10))

        ttk.Label(btn_row, text="Epochs:").pack(side="left", padx=(0, 6))
        self.epochs_entry = ttk.Entry(btn_row, textvariable=self.epochs_var, width=6)
        self.epochs_entry.pack(side="left", padx=(0, 14))
        self.epochs_entry.bind("<Return>", self.check_train_inputs_event)

        ttk.Label(btn_row, text="Batch size:").pack(side="left", padx=(0, 6))
        self.batch_entry = ttk.Entry(btn_row, textvariable=self.batch_var, width=6)
        self.batch_entry.pack(side="left", padx=(0, 18))
        self.batch_entry.bind("<Return>", self.check_train_inputs_event)

        ttk.Label(btn_row, text="Dropout:").pack(side="left", padx=(0, 6))
        self.dropout_combo = ttk.Combobox(
            btn_row,
            textvariable=self.dropout_var,
            values=[str(v) for v in DROPOUT_VALUES],
            width=5,
            state="readonly",
        )
        self.dropout_combo.pack(side="left", padx=(0, 18))

        self.train_btn = ttk.Button(btn_row, text="1. Spusti tréning", command=self.start_train_from_gui)
        self.train_btn.pack(side="left", padx=6)

        self.load_btn = ttk.Button(btn_row, text="2. Načítaj test priečinok", command=self.load_test_folder_gui)
        self.load_btn.pack(side="left", padx=6)

        self.classify_btn = ttk.Button(btn_row, text="3. Klasifikuj obrázok", command=self.classify_now_img)
        self.classify_btn.pack(side="left", padx=6)

        self.clear_btn = ttk.Button(btn_row, text="Vymaž", command=self.clear_gui)
        self.clear_btn.pack(side="left", padx=6)

        # stavový text
        ttk.Label(main, textvariable=self.status_var, font=("Arial", 11)).pack(pady=(4, 10))

        # notebook so záložkami
        self.tabs = ttk.Notebook(main)
        self.tabs.pack(fill="both", expand=True, pady=(0, 10))

        # ====================================================
        # TAB 1 - LOG A VÝSLEDKY
        # ====================================================
        self.tab_log = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_log, text="Log a výsledky")

        result_box = ttk.LabelFrame(self.tab_log, text="Výsledok a log", padding=12)
        result_box.pack(fill="both", expand=True)

        result_inner = ttk.Frame(result_box)
        result_inner.pack(fill="both", expand=True)

        scroll = ttk.Scrollbar(result_inner, orient="vertical")
        scroll.pack(side="right", fill="y")

        self.text_box = tk.Text(
            result_inner,
            height=16,
            wrap="word",
            font=("Consolas", 11),
            yscrollcommand=scroll.set,
        )
        self.text_box.pack(side="left", fill="both", expand=True)

        scroll.config(command=self.text_box.yview)

        self.text_box.insert("1.0", "Výsledky sa ukážu tu.")
        self.text_box.config(state="disabled")

        # ====================================================
        # TAB 2 - GRAFY
        # ====================================================
        self.tab_graphs = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_graphs, text="Grafy tréningu")

        graphs_outer = ttk.Frame(self.tab_graphs, padding=10)
        graphs_outer.pack(fill="both", expand=True)

        self.graph_info = ttk.Label(
            graphs_outer,
            text="Po tréningu sa tu ukáže accuracy a loss graf.",
            font=("Arial", 11),
        )
        self.graph_info.pack(anchor="w", pady=(0, 8))

        # tu budú dva stĺpce vedľa seba
        graphs_row = ttk.Frame(graphs_outer)
        graphs_row.pack(fill="both", expand=True)

        graphs_row.columnconfigure(0, weight=1)
        graphs_row.columnconfigure(1, weight=1)
        graphs_row.rowconfigure(0, weight=1)

        self.acc_frame = ttk.LabelFrame(graphs_row, text="Accuracy graf", padding=8)
        self.acc_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.loss_frame = ttk.LabelFrame(graphs_row, text="Loss graf", padding=8)
        self.loss_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        # ====================================================
        # TAB 3 - CONFUSION MATRIX
        # ====================================================
        self.tab_cm = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_cm, text="Confusion matrix")

        cm_outer = ttk.Frame(self.tab_cm, padding=10)
        cm_outer.pack(fill="both", expand=True)

        self.cm_info = ttk.Label(
            cm_outer,
            text="Po hotovom tréningu sa tu ukáže confusion matrix.",
            font=("Arial", 11),
        )
        self.cm_info.pack(anchor="w", pady=(0, 8))

        self.cm_frame = ttk.LabelFrame(cm_outer, text="Confusion matrix graf", padding=8)
        self.cm_frame.pack(fill="both", expand=True)

        # ====================================================
        # TAB 4 - PRECISION / RECALL / F1
        # ====================================================
        self.tab_metrics = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_metrics, text="Precision / Recall / F1")

        metrics_outer = ttk.Frame(self.tab_metrics, padding=10)
        metrics_outer.pack(fill="both", expand=True)

        # info text
        self.metrics_label = ttk.Label(
            metrics_outer,
            text="Po tréningu sa tu zobrazia metriky klasifikácie.",
            font=("Arial", 11),
        )
        self.metrics_label.pack(anchor="w", pady=(0, 8))

        # frame pre graf
        self.metrics_frame = ttk.LabelFrame(metrics_outer, text="Graf metrík", padding=8)
        self.metrics_frame.pack(fill="both", expand=True)

        # po štarte sú zatiaľ len prázdne texty
        self.show_empty_plot(self.acc_frame, "Accuracy graf zatiaľ nie je k dispozícii.")
        self.show_empty_plot(self.loss_frame, "Loss graf zatiaľ nie je k dispozícii.")
        self.show_empty_plot(self.cm_frame, "Confusion matrix zatiaľ nie je k dispozícii.")

    def show_empty_plot(self, frame: ttk.Frame, text: str) -> None:
        """
        Keď ešte nemáme graf, dáme do frame aspoň text.
        """
        lbl = ttk.Label(frame, text=text, font=("Arial", 10))
        lbl.pack(expand=True, pady=20)

    def clear_plot(self, frame: ttk.Frame) -> None:
        """
        Vymaže všetko z frame-u s grafom.
        """
        for child in frame.winfo_children():
            child.destroy()

    def draw_saved_plot_img(self, parent_frame: ttk.Frame, img_path: Path, title: str) -> None:
        """
        Funkcia na načítanie uloženého PNG grafu:
        - accuracy
        - loss
        - confusion matrix
        - metrics graf
        """
        self.clear_plot(parent_frame)

        if not img_path.exists():
            self.show_empty_plot(parent_frame, f"{title} zatiaľ nie je k dispozícii.")
            return

        img = Image.open(img_path).convert("RGBA")
        arr = np.asarray(img)

        fig = Figure(figsize=(6, 5), dpi=100)
        ax = fig.add_subplot(111)

        ax.imshow(arr, aspect="auto")
        ax.axis("off")

        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_saved_plots(self) -> None:
        """
        Po štarte appky sa skúsi načítať:
        - accuracy graf
        - loss graf
        - confusion matrix

        Keď existujú v models, ukážu sa hneď.
        """
        has_acc = ACC_PLOT_FILE.exists()
        has_loss = LOSS_PLOT_FILE.exists()
        has_cm = CM_PLOT_FILE.exists()

        if has_acc:
            self.draw_saved_plot_img(self.acc_frame, ACC_PLOT_FILE, "Accuracy graf")

        if has_loss:
            self.draw_saved_plot_img(self.loss_frame, LOSS_PLOT_FILE, "Loss graf")

        if has_acc or has_loss:
            parts = []

            if has_acc:
                parts.append(f"Accuracy graf načítaný: {ACC_PLOT_FILE.name}")

            if has_loss:
                parts.append(f"Loss graf načítaný: {LOSS_PLOT_FILE.name}")

            self.graph_info.config(text=" | ".join(parts))

        if has_cm:
            self.draw_saved_plot_img(self.cm_frame, CM_PLOT_FILE, "Confusion matrix")
            self.cm_info.config(text=f"Načítaná confusion matrix: {CM_PLOT_FILE.name}")

        if METRICS_PLOT_FILE.exists():
            self.draw_saved_plot_img(
                parent_frame=self.metrics_frame,
                img_path=METRICS_PLOT_FILE,
                title="Metrics graf"
            )

            self.metrics_label.config(
                text=f"Načítaný metrics graf: {METRICS_PLOT_FILE.name}"
            )

    def try_load_model(self) -> None:
        """
        Po štarte sa pokúsime načítať model.
        Keď neexistuje, nič strašné sa nedeje.
        """
        try:
            self.model, self.class_names = load_model_and_classes()
            self.status_var.set("Model je načítaný a pripravený.")
        except Exception:
            self.model = None
            self.class_names = []
            self.status_var.set("Model ešte nie je. Najprv ho natrénuj.")

    def add_log(self, text: str) -> None:
        """
        Pridá jeden riadok do textového poľa.
        """
        self.text_box.config(state="normal")
        self.text_box.insert(tk.END, text + "\n")
        self.text_box.see(tk.END)
        self.text_box.config(state="disabled")

    def read_log_q(self) -> None:
        """
        Pravidelne číta frontu logov a píše text do GUI.
        Toto beží stále dokola cez root.after().
        """
        while True:
            try:
                line = self.log_q.get_nowait()
            except queue.Empty:
                break
            else:
                self.add_log(line)

        self.root.after(150, self.read_log_q)

    def set_text_box(self, text: str) -> None:
        """
        Vymaže starý text a nastaví nový.
        """
        self.text_box.config(state="normal")
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", text)
        self.text_box.config(state="disabled")

    def check_train_inputs_event(self, event=None) -> None:
        """
        Wrapper — volá check_train_inputs() pri stlačení Enter v epocha/batch inpute.
        """
        self.check_train_inputs()

    def check_train_inputs(self) -> bool:
        """
        Skontroluje epochs a batch size.
        Keď sú zlé, ukáže chybu.
        """
        # kontrola epochs
        try:
            epochs = int(self.epochs_var.get().strip())
            if epochs <= 0:
                raise ValueError
        except ValueError:
            self.epochs_var.set(str(self.last_epochs))
            messagebox.showerror("Chyba", "Epochs musí byť kladné celé číslo.")
            self.epochs_entry.focus_set()
            self.epochs_entry.selection_range(0, tk.END)
            return False

        # kontrola batch size
        try:
            batch = int(self.batch_var.get().strip())
            if batch <= 0:
                raise ValueError
        except ValueError:
            self.batch_var.set(str(self.last_batch))
            messagebox.showerror("Chyba", "Batch size musí byť kladné celé číslo.")
            self.batch_entry.focus_set()
            self.batch_entry.selection_range(0, tk.END)
            return False

        # keď je všetko ok, uložíme si to
        self.last_epochs = epochs
        self.last_batch = batch
        self.last_dropout = float(self.dropout_var.get())

        self.epochs_var.set(str(epochs))
        self.batch_var.set(str(batch))

        self.status_var.set(f"Nastavené: epochs={epochs}, batch size={batch}, dropout={self.last_dropout}")
        return True

    def set_train_mode(self, training: bool) -> None:
        """
        Keď beží tréning, niektoré veci sa vypnú.
        A train tlačidlo sa zmení na STOP.
        """
        if training:
            self.train_btn.config(state="normal", text="STOP tréningu", command=self.stop_train_from_gui)
            self.load_btn.config(state="disabled")
            self.classify_btn.config(state="disabled")
            self.prev_btn.config(state="disabled")
            self.next_btn.config(state="disabled")
            self.jump_entry.config(state="disabled")
            self.clear_btn.config(state="disabled")
            self.epochs_entry.config(state="disabled")
            self.batch_entry.config(state="disabled")
            self.dropout_combo.config(state="disabled")
        else:
            self.train_btn.config(state="normal", text="1. Spusti tréning", command=self.start_train_from_gui)
            self.load_btn.config(state="normal")
            self.classify_btn.config(state="normal")
            self.prev_btn.config(state="normal")
            self.next_btn.config(state="normal")
            self.jump_entry.config(state="normal")
            self.clear_btn.config(state="normal")
            self.epochs_entry.config(state="normal")
            self.batch_entry.config(state="normal")
            self.dropout_combo.config(state="readonly")

    def reset_plots_before_train(self) -> None:
        """
        Pred novým tréningom zmažeme staré grafy z GUI.
        """
        self.clear_plot(self.acc_frame)
        self.clear_plot(self.loss_frame)
        self.clear_plot(self.cm_frame)

        self.show_empty_plot(self.acc_frame, "Prebieha nový tréning...")
        self.show_empty_plot(self.loss_frame, "Prebieha nový tréning...")
        self.show_empty_plot(self.cm_frame, "Confusion matrix sa spraví po dokončení testu...")

        self.graph_info.config(text="Pripravujú sa nové grafy z tréningu.")
        self.cm_info.config(text="Confusion matrix sa ukáže po úplnom dokončení tréningu.")

    def start_train_from_gui(self) -> None:
        """
        Toto sa spustí po kliknutí na tlačidlo tréningu.
        """
        # ak už tréning ide, nový nespúšťame
        if self.is_training:
            messagebox.showinfo("Info", "Tréning už beží.")
            return

        # najprv skontrolujeme vstupy
        if not self.check_train_inputs():
            return

        epochs = self.last_epochs
        batch = self.last_batch
        dropout = self.last_dropout

        # nastavíme stav
        self.is_training = True
        self.train_error = None
        self.train_result = None
        self.was_stopped = False
        self.stop_event.clear()

        self.status_var.set("Tréning práve beží...")

        # vymažeme starý text
        self.set_text_box("")

        # úvodné info do logu
        self.add_log("==================================================")
        self.add_log("Spúšťa sa nový tréning modelu.")
        self.add_log(f"Nastavené epochs: {epochs}")
        self.add_log(f"Nastavený batch size: {batch}")
        self.add_log(f"Nastavený dropout: {dropout}")
        self.add_log("==================================================")

        # vyčistíme grafy
        self.reset_plots_before_train()

        # vypneme ovládanie
        self.set_train_mode(True)

        # skočíme na log tab
        self.tabs.select(self.tab_log)

        # spustíme tréning v inom vlákne
        # toto je veľmi dôležité, aby GUI nezamrzlo
        self.train_thread = threading.Thread(
            target=self.train_worker,
            args=(epochs, batch, dropout),
            daemon=True,
        )
        self.train_thread.start()

        # spustíme kontrolu vlákna
        self.root.after(300, self.check_train_thread)

    def stop_train_from_gui(self) -> None:
        """
        Používateľ klikol na STOP.
        """
        if not self.is_training:
            return

        # ak už stop prišiel, znova ho neposielame
        if self.stop_event.is_set():
            self.status_var.set("STOP už bol odoslaný. Počkaj chvíľu.")
            return

        self.was_stopped = True
        self.stop_event.set()

        self.status_var.set("Zastavuje sa tréning...")
        self.add_log("Používateľ stlačil STOP tréningu.")

        self.train_btn.config(state="disabled", text="Zastavujem...")

    def train_worker(self, epochs: int, batch: int, dropout: float) -> None:
        """
        Samotný tréning beží v inom vlákne.
        """
        try:
            self.train_result = train_net(
                epochs=epochs,
                batch_size=batch,
                dropout_rate=dropout,
                log_q=self.log_q,
                stop_event=self.stop_event,
            )
            self.train_error = None
        except Exception:
            self.train_error = traceback.format_exc()

    def show_train_visuals(self) -> None:
        """
        Po tréningu ukáže všetky grafy z uložených PNG:
        - accuracy
        - loss
        - confusion matrix
        - metrics
        """
        # ================================
        # ACCURACY + LOSS
        # ================================
        self.draw_saved_plot_img(self.acc_frame, ACC_PLOT_FILE, "Accuracy graf")
        self.draw_saved_plot_img(self.loss_frame, LOSS_PLOT_FILE, "Loss graf")

        self.graph_info.config(
            text=(
                f"Accuracy graf: {ACC_PLOT_FILE.name} | "
                f"Loss graf: {LOSS_PLOT_FILE.name}"
            )
        )

        # ================================
        # CONFUSION MATRIX
        # ================================
        if CM_PLOT_FILE.exists():
            self.draw_saved_plot_img(self.cm_frame, CM_PLOT_FILE, "Confusion matrix")

            test_acc = self.train_result.get("test_accuracy")
            test_loss = self.train_result.get("test_loss")

            if test_acc is not None and test_loss is not None:
                self.cm_info.config(
                    text=(
                        f"Test accuracy: {test_acc * 100:.2f} % | "
                        f"Test loss: {test_loss:.4f} | "
                        f"Súbor: {CM_PLOT_FILE.name}"
                    )
                )
            else:
                self.cm_info.config(text=f"Súbor: {CM_PLOT_FILE.name}")
        else:
            self.show_empty_plot(
                self.cm_frame,
                "Confusion matrix sa nevytvorila."
            )

        # ================================
        # METRICS (precision, recall, f1)
        # ================================
        if METRICS_PLOT_FILE.exists():
            self.draw_saved_plot_img(
                self.metrics_frame,
                METRICS_PLOT_FILE,
                "Metrics graf"
            )

            precision = self.train_result.get("precision")
            recall = self.train_result.get("recall")
            f1 = self.train_result.get("f1")

            if precision is not None:
                self.metrics_label.config(
                    text=(
                        f"Precision: {precision:.4f} | "
                        f"Recall: {recall:.4f} | "
                        f"F1-score: {f1:.4f}"
                    )
                )

    def check_train_thread(self) -> None:
        """
        Pravidelne kontroluje, či tréningové vlákno ešte beží.
        """
        if self.train_thread is not None and self.train_thread.is_alive():
            self.root.after(300, self.check_train_thread)
            return

        # keď sa dostaneme sem, tréning už nebeží
        self.is_training = False
        self.set_train_mode(False)

        # ak nastala chyba, vypíšeme ju
        if self.train_error:
            self.status_var.set("Tréning zlyhal.")
            self.set_text_box("Chyba pri tréningu modelu:\n\n" + self.train_error)
            messagebox.showerror("Chyba", "Tréning zlyhal. Detail je vo výsledkovom poli.")
            return

        # inak skúsime načítať model
        try:
            self.model, self.class_names = load_model_and_classes()

            if self.was_stopped:
                self.status_var.set("Tréning bol zastavený. Načítal sa posledný model.")
            else:
                self.status_var.set("Tréning hotový. Model je pripravený.")

            # zhrnutie do logu
            self.add_log("")
            self.add_log("=== ZHRNUTIE TRÉNINGU ===")

            if self.was_stopped:
                self.add_log("Tréning modelu bol zastavený používateľom.")
            else:
                self.add_log("Tréning modelu úspešne skončil.")

            self.add_log(f"Uložený model: {MODEL_FILE.name}")
            self.add_log(f"Počet tried: {len(self.class_names)}")
            self.add_log(f"Prvých 10 tried: {', '.join(self.class_names[:10])}")
            self.add_log(f"Accuracy graf: {ACC_PLOT_FILE}")
            self.add_log(f"Loss graf: {LOSS_PLOT_FILE}")

            if self.train_result and self.train_result.get("cm") is not None:
                self.add_log(f"Confusion matrix: {CM_PLOT_FILE}")
            else:
                self.add_log("Confusion matrix sa nevytvorila.")

            # tu sa vykreslia grafy
            self.show_train_visuals()

            # po tréningu skočíme na grafy
            if self.train_result and self.train_result.get("stopped"):
                self.tabs.select(self.tab_log)
            else:
                self.tabs.select(self.tab_graphs)

        except Exception as e:
            self.status_var.set("Model sa natrénoval, ale nepodarilo sa ho načítať.")
            self.add_log("")
            self.add_log("=== CHYBA PO TRÉNINGU ===")
            self.add_log(f"Nepodarilo sa načítať model po tréningu: {e}")
            self.add_log(traceback.format_exc())
            messagebox.showerror("Chyba", f"Model sa nepodarilo načítať:\n{e}")

    def load_test_folder_gui(self) -> None:
        """
        Načíta všetky obrázky z dataset/test.
        """
        try:
            self.img_paths = load_test_images()
        except Exception as e:
            messagebox.showerror("Chyba", str(e))
            return

        if not self.img_paths:
            messagebox.showwarning("Upozornenie", "V dataset/test sa nenašiel žiadny obrázok.")
            return

        self.now_img_idx = 0
        self.jump_var.set("1")
        self.show_now_img()

        self.status_var.set(f"Načítaných obrázkov z test priečinka: {len(self.img_paths)}")

        self.set_text_box(
            "Načítal sa celý test priečinok.\n\n"
            f"Počet načítaných obrázkov: {len(self.img_paths)}\n"
            "Použi tlačidlá Doľava / Doprava alebo šípky na klávesnici.\n"
            "Potom klikni na 'Klasifikuj obrázok'."
        )

        self.tabs.select(self.tab_log)

    def show_now_img(self) -> None:
        """
        Ukáže aktuálny obrázok v GUI.
        """
        if not self.img_paths or self.now_img_idx < 0:
            return

        img_path = self.img_paths[self.now_img_idx]

        img = Image.open(img_path).convert("RGB")
        preview = img.copy()
        preview.thumbnail(PREVIEW_SIZE)

        self.preview_photo = ImageTk.PhotoImage(preview)
        self.img_label.configure(image=self.preview_photo, text="")

        self.jump_var.set(str(self.now_img_idx + 1))

        try:
            # obrázok z datasetu
            rel_path = img_path.relative_to(DATASET / "test")
            display_name = str(rel_path)
        except ValueError:
            # obrázok z uploadu
            display_name = f"[UPLOAD] {img_path.name}"

        self.img_info_var.set(f"/ {len(self.img_paths)} | {display_name}")

    def prev_img(self) -> None:
        """
        Posun na predchádzajúci obrázok.
        """
        if not self.img_paths:
            return

        self.now_img_idx = (self.now_img_idx - 1) % len(self.img_paths)
        self.show_now_img()

    def next_img(self) -> None:
        """
        Posun na ďalší obrázok.
        """
        if not self.img_paths:
            return

        self.now_img_idx = (self.now_img_idx + 1) % len(self.img_paths)
        self.show_now_img()

    def jump_img_event(self, event=None) -> None:
        """
        Wrapper — volá jump_img() pri stlačení Enter v poli čísla obrázka.
        """
        self.jump_img()

    def jump_img(self) -> None:
        """
        Presun na obrázok podľa čísla, ktoré zadal používateľ.
        """
        if not self.img_paths:
            messagebox.showwarning("Upozornenie", "Najprv načítaj test priečinok.")
            return

        try:
            wanted_idx = int(self.jump_var.get().strip())
        except ValueError:
            messagebox.showerror("Chyba", "Zadaj platné celé číslo obrázka.")
            return

        if wanted_idx < 1 or wanted_idx > len(self.img_paths):
            messagebox.showerror("Chyba", f"Zadaj číslo od 1 do {len(self.img_paths)}.")
            return

        self.now_img_idx = wanted_idx - 1
        self.show_now_img()
        self.status_var.set(f"Presunuté na obrázok číslo {wanted_idx}.")

    def classify_now_img(self) -> None:
        """
        Klasifikácia aktuálne zobrazeného obrázka.
        """
        # bez modelu to nepôjde
        if self.model is None:
            messagebox.showwarning("Upozornenie", "Model nie je načítaný. Najprv ho natrénuj.")
            return

        # bez test obrázkov to tiež nepôjde
        if not self.img_paths or self.now_img_idx < 0:
            messagebox.showwarning("Upozornenie", "Najprv načítaj test priečinok.")
            return

        try:
            img_path = self.img_paths[self.now_img_idx]

            # tu sa obrázok pripraví pre model
            arr = prepare_img(img_path)

            # tu model spraví predikciu
            pred = self.model.predict(arr, verbose=0)[0]

            # index triedy s najväčšou pravdepodobnosťou
            pred_idx = int(np.argmax(pred))

            # názov predikovanej triedy
            pred_class = self.class_names[pred_idx]

            # confidence v percentách
            conf = float(np.max(pred)) * 100.0

            # top-k indexy
            top_idxs = np.argsort(pred)[::-1][:TOP_K]

            # skutočná trieda podľa názvu priečinka
            real_class = img_path.parent.name

            # či je výsledok správny
            is_ok = pred_class == real_class
            ok_text = "ÁNO" if is_ok else "NIE"

            # tu si pripravíme riadky textu
            lines = [
                f"Načítaný súbor: {img_path.name}",
                f"Skutočná trieda podľa priečinka: {real_class}",
                f"Predikovaná trieda: {pred_class}",
                f"Pravdepodobnosť: {conf:.2f} %",
                f"Správna klasifikácia: {ok_text}",
                "",
                f"Top {TOP_K} výsledky:",
            ]

            # vypíšeme top výsledky
            for i, idx in enumerate(top_idxs, start=1):
                label = self.class_names[int(idx)]
                prob = float(pred[int(idx)]) * 100.0
                lines.append(f"{i}. {label:<25} {prob:>7.2f} %")

            self.set_text_box("\n".join(lines))
            self.status_var.set("Obrázok bol vyhodnotený.")
            self.tabs.select(self.tab_log)

        except Exception as e:
            self.status_var.set("Klasifikácia zlyhala.")
            self.set_text_box("Chyba pri klasifikácii:\n\n" + str(e) + "\n\n" + traceback.format_exc())
            messagebox.showerror("Chyba", f"Klasifikácia zlyhala:\n{e}")

    def upload_image_gui(self):
        """
        Výber obrázka mimo datasetu (napr. z plochy).
        """
        file_path = filedialog.askopenfilename(
            title="Vyber obrázok",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )

        if not file_path:
            return

        path = Path(file_path)

        self.img_paths = [path]
        self.now_img_idx = 0

        self.show_now_img()
        self.status_var.set("Načítaný externý obrázok.")

        self.set_text_box(
            f"Načítaný obrázok mimo datasetu:\n{path}\n\nKlikni na 'Klasifikuj obrázok'."
        )
        self.tabs.select(self.tab_log)

    def clear_gui(self) -> None:
        """
        Vymaže obrázok a texty v GUI.
        Grafy nechávame tak.
        """
        self.img_paths = []
        self.now_img_idx = -1
        self.preview_photo = None

        self.img_label.configure(image="", text="Tu sa ukáže načítaný obrázok")
        self.img_info_var.set("/ 0 | -")
        self.jump_var.set("1")

        if self.model is None:
            self.status_var.set("Model nie je načítaný. Najprv ho natrénuj.")
        else:
            self.status_var.set("Hotovo, môžeš pokračovať.")

        self.set_text_box("Výsledky sa ukážu tu.")
        self.tabs.select(self.tab_log)

    def _open_url(self, url: str) -> None:
        """
        Otvorí URL v predvolenom prehliadači.
        Používa sa pre odkaz na GitHub repozitár.
        """
        import webbrowser
        webbrowser.open(url)

    def _show_about(self) -> None:
        """
        Zobrazí dialóg 'O autorovi' s informáciami o aplikácii.
        """
        messagebox.showinfo(
            "O autorovi",
            "Klasifikácia hracích kariet\n\n"
            "Autor: Mgr. Ing. Ondrej, MBA\n"
            "Študijný program: MSc Umelá Inteligencia a Strojové Učenie\n"
            "Škola: VITA ACADEMY\n\n",
        )

# ============================================================
# 8. SPUSTENIE PROGRAMU
# ============================================================
# Odtiaľto sa spustí celá aplikácia.
# ============================================================

def main() -> None:
    # vytvorenie hlavného okna
    root = tk.Tk()

    # IKONA OKNA (CUSTOM PNG)
    try:
        icon_path = get_resource_path("icon.png")
        if icon_path.exists():
            icon_img = ImageTk.PhotoImage(Image.open(icon_path))
            root.iconphoto(True, icon_img)
    except Exception as e:
        print("Ikona sa nepodarila načítať:", e)

    # trochu krajší vzhľad ttk widgetov
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    # vytvorenie appky
    app = App(root)

    # hlavná slučka GUI
    root.mainloop()


# keď spustíme tento súbor priamo, zavolá sa main()
if __name__ == "__main__":
    main()
