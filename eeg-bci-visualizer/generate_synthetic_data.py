"""
generate_synthetic_data.py
============================
Generates a SYNTHETIC placeholder dataset with the same shape/format
that train_model.py produces from real PhysioNet data. This exists
because physionet.org is unreachable from this sandbox -- it lets you
deploy and test the full app pipeline (UI, plotting, CSP+LDA
classification logic) immediately.

The synthetic signal is built to be physiologically *plausible*:
- 1/f ("pink") background noise, like real EEG
- baseline mu (8-13Hz) and beta (13-30Hz) rhythms on all channels
- event-related desynchronization (ERD): imagining RIGHT fist movement
  suppresses mu/beta power over LEFT motor cortex channels (C3, C1, C5),
  and imagining LEFT fist movement suppresses it over RIGHT motor
  cortex channels (C4, C2, C6) -- this is the real physiological
  effect CSP is designed to detect.

IMPORTANT: This is NOT real EEG data. Do not present results trained
on this data as reflecting real brain signals. Run train_model.py
locally (which downloads real PhysioNet recordings) to replace these
files with a model trained on genuine EEG before using this for
anything beyond pipeline testing.
"""

import os
import numpy as np
import joblib
from scipy.signal import butter, filtfilt
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

# ---- Configuration (mirrors train_model.py) ----
SFREQ = 80.0
TMIN, TMAX = -1.0, 3.0
N_TIMES = int((TMAX - TMIN) * SFREQ)  # 320 samples

# Same 64-channel layout as the standardized PhysioNet EEGBCI montage
CH_NAMES = [
    "Fc5", "Fc3", "Fc1", "Fcz", "Fc2", "Fc4", "Fc6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "Cp5", "Cp3", "Cp1", "Cpz", "Cp2", "Cp4", "Cp6",
    "Fp1", "Fpz", "Fp2", "Af7", "Af3", "Afz", "Af4", "Af8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "Ft7", "Ft8", "T7", "T8", "T9", "T10", "Tp7", "Tp8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "Po7", "Po3", "Poz", "Po4", "Po8", "O1", "Oz", "O2", "Iz",
]
N_CHANNELS = len(CH_NAMES)

LEFT_MOTOR_CH = ["C3", "C1", "C5"]    # controls RIGHT hand -> ERD during right-fist imagery
RIGHT_MOTOR_CH = ["C4", "C2", "C6"]   # controls LEFT hand -> ERD during left-fist imagery

N_TRAIN_SUBJECTS = 9
N_TRAIN_EPOCHS_PER_CLASS = 15
N_DEMO_SUBJECTS = 5
N_DEMO_EPOCHS_PER_SUBJECT = 6

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def pink_noise(n_samples, rng, amplitude=1.0):
    """Generate 1/f pink noise via spectral shaping of white noise."""
    white = rng.standard_normal(n_samples)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]  # avoid divide-by-zero at DC
    spectrum = spectrum / np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n_samples)
    pink = pink / (np.std(pink) + 1e-9) * amplitude
    return pink


_BP_B, _BP_A = butter(4, [8 / (SFREQ / 2), 30 / (SFREQ / 2)], btype="band")


def make_epoch(rng, label):
    """
    Build one synthetic (n_channels, n_times) EEG epoch.
    label: 0 = imagined left fist, 1 = imagined right fist

    Signal is bandpass-filtered 8-30Hz after synthesis, matching the
    preprocessing step train_model.py applies to real EEG, so the
    saved data mirrors the actual deployed pipeline's input distribution.
    """
    t = np.linspace(TMIN, TMAX, N_TIMES)
    imagery_mask = (t >= 0)  # imagery/movement period starts at t=0

    epoch = np.zeros((N_CHANNELS, N_TIMES))
    suppressed_channels = LEFT_MOTOR_CH if label == 1 else RIGHT_MOTOR_CH

    for i, ch in enumerate(CH_NAMES):
        # Background 1/f noise, amplitude in a plausible EEG range (uV)
        background = pink_noise(N_TIMES, rng, amplitude=4.0)

        # Baseline mu (10Hz) + beta (20Hz) rhythms present on all channels
        mu_amp = 14.0
        beta_amp = 7.0
        if ch in suppressed_channels:
            # ERD: rhythm power drops sharply during the imagery window
            suppression = np.where(imagery_mask, 0.25, 1.0)
        else:
            suppression = np.ones(N_TIMES)

        phase_mu = rng.uniform(0, 2 * np.pi)
        phase_beta = rng.uniform(0, 2 * np.pi)
        mu_wave = mu_amp * suppression * np.sin(2 * np.pi * 10 * t + phase_mu)
        beta_wave = beta_amp * suppression * np.sin(2 * np.pi * 20 * t + phase_beta)

        sensor_noise = rng.standard_normal(N_TIMES) * 1.0

        epoch[i] = background + mu_wave + beta_wave + sensor_noise

    epoch = filtfilt(_BP_B, _BP_A, epoch, axis=1)
    return epoch.astype(np.float32)


def build_dataset(n_subjects, n_epochs_per_class, seed_offset=0):
    X, y, subject_ids = [], [], []
    for subj in range(1, n_subjects + 1):
        rng = np.random.default_rng(seed_offset + subj)
        for label in (0, 1):
            for _ in range(n_epochs_per_class):
                X.append(make_epoch(rng, label))
                y.append(label)
                subject_ids.append(subj)
    return np.array(X), np.array(y), np.array(subject_ids)


def main():
    print("Generating synthetic training set...")
    X_train, y_train, _ = build_dataset(N_TRAIN_SUBJECTS, N_TRAIN_EPOCHS_PER_CLASS, seed_offset=0)
    print(f"  Train shape: {X_train.shape}")

    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("CSP", csp), ("LDA", lda)])

    print("Running 5-fold cross-validation...")
    scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=1)
    print(f"Cross-val accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    print("Fitting final model...")
    clf.fit(X_train, y_train)

    model_path = os.path.join(DATA_DIR, "csp_lda_model.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved model -> {model_path}")

    print("\nGenerating synthetic demo dataset for app.py...")
    demo_X, demo_y, demo_subject_ids = [], [], []
    for subj in range(1, N_DEMO_SUBJECTS + 1):
        rng = np.random.default_rng(1000 + subj)  # different seed range from training
        for i in range(N_DEMO_EPOCHS_PER_SUBJECT):
            label = i % 2
            demo_X.append(make_epoch(rng, label))
            demo_y.append(label)
            demo_subject_ids.append(subj)

    demo_X = np.array(demo_X)
    demo_y = np.array(demo_y)
    demo_subject_ids = np.array(demo_subject_ids)

    demo_path = os.path.join(DATA_DIR, "demo_data.npz")
    np.savez_compressed(
        demo_path,
        X=demo_X, y=demo_y, subject_ids=demo_subject_ids,
        ch_names=np.array(CH_NAMES), sfreq=SFREQ,
        tmin=TMIN, tmax=TMAX,
        synthetic=True,
    )
    print(f"Saved demo data -> {demo_path}  (shape={demo_X.shape})")
    print("\nDone. This is SYNTHETIC placeholder data -- see README before deploying publicly.")


if __name__ == "__main__":
    main()
