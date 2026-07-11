"""
train_model.py
================
Run this ONCE locally (not on Streamlit Cloud) to:
  1. Download a handful of subjects from the PhysioNet EEG Motor
     Movement/Imagery Dataset (via MNE's built-in downloader).
  2. Preprocess (filter, resample, epoch) the data.
  3. Train a CSP + LDA classifier to distinguish imagined LEFT fist
     vs imagined RIGHT fist movement.
  4. Save the trained model AND a small demo dataset to data/,
     which is what gets committed to GitHub and loaded by app.py.

Usage:
    python train_model.py

Requires internet access to physionet.org (MNE handles the download).
This step is NOT run at Streamlit Cloud runtime -- it's a one-time
local step whose *output* (small .joblib / .npz files) is deployed.
"""

import numpy as np
import mne
from mne.decoding import CSP
from mne.datasets import eegbci
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import joblib
import os

mne.set_log_level("WARNING")

# ---- Configuration ----
SUBJECTS = list(range(1, 10))       # subjects 1-9 for training
RUNS = [4, 8, 12]                   # imagined left/right fist runs
RESAMPLE_SFREQ = 80.0                # downsample to keep files small
TMIN, TMAX = -1.0, 3.0                # epoch window around each event (seconds)
DEMO_SUBJECTS = [1, 2, 3, 4, 5]     # subset saved for the Streamlit demo
DEMO_EPOCHS_PER_SUBJECT = 6
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def load_subject_epochs(subject_id):
    """Download + preprocess one subject, return an mne.Epochs object."""
    raw_fnames = eegbci.load_data(subject_id, RUNS, update_path=True)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
    raw = mne.concatenate_raws(raws)

    eegbci.standardize(raw)  # fix channel naming to standard 10-05 system
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, on_missing="ignore")

    raw.filter(8.0, 30.0, fir_design="firwin", skip_by_annotation="edge")
    raw.resample(RESAMPLE_SFREQ)

    events, event_id = mne.events_from_annotations(raw)
    # Keep only T1 (left fist imagined) and T2 (right fist imagined)
    picked_event_id = {k: v for k, v in event_id.items() if k in ("T1", "T2")}

    epochs = mne.Epochs(
        raw, events, picked_event_id,
        tmin=TMIN, tmax=TMAX,
        baseline=None, preload=True, verbose=False,
    )
    return epochs


def main():
    print("Loading and preprocessing training subjects...")
    all_epochs = []
    for subj in SUBJECTS:
        print(f"  Subject {subj:03d}...")
        try:
            ep = load_subject_epochs(subj)
            all_epochs.append(ep)
        except Exception as e:
            print(f"    Skipping subject {subj}: {e}")

    epochs = mne.concatenate_epochs(all_epochs)
    X = epochs.get_data(copy=False)          # (n_epochs, n_channels, n_times)
    y = epochs.events[:, -1]
    # Relabel to 0 (left) / 1 (right)
    unique_labels = np.unique(y)
    label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
    y = np.array([label_map[v] for v in y])

    print(f"\nTotal epochs: {X.shape[0]}, channels: {X.shape[1]}, samples/epoch: {X.shape[2]}")

    # ---- Train CSP + LDA pipeline ----
    csp = CSP(n_components=6, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("CSP", csp), ("LDA", lda)])

    print("Running 5-fold cross-validation...")
    scores = cross_val_score(clf, X, y, cv=5, n_jobs=1)
    print(f"Cross-val accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    print("Fitting final model on all training data...")
    clf.fit(X, y)

    model_path = os.path.join(DATA_DIR, "csp_lda_model.joblib")
    joblib.dump(clf, model_path)
    print(f"Saved model -> {model_path}")

    # ---- Build small demo dataset for the Streamlit app ----
    print("\nBuilding demo dataset for app.py...")
    demo_X, demo_y, demo_subject_ids = [], [], []
    for subj in DEMO_SUBJECTS:
        ep = load_subject_epochs(subj)
        Xd = ep.get_data(copy=False)[:DEMO_EPOCHS_PER_SUBJECT]
        yd_raw = ep.events[:, -1][:DEMO_EPOCHS_PER_SUBJECT]
        yd = np.array([label_map.get(v, v) for v in yd_raw])
        demo_X.append(Xd)
        demo_y.append(yd)
        demo_subject_ids += [subj] * Xd.shape[0]

    demo_X = np.concatenate(demo_X, axis=0)
    demo_y = np.concatenate(demo_y, axis=0)
    demo_subject_ids = np.array(demo_subject_ids)
    ch_names = epochs.ch_names
    sfreq = epochs.info["sfreq"]

    demo_path = os.path.join(DATA_DIR, "demo_data.npz")
    np.savez_compressed(
        demo_path,
        X=demo_X, y=demo_y, subject_ids=demo_subject_ids,
        ch_names=np.array(ch_names), sfreq=sfreq,
        tmin=TMIN, tmax=TMAX,
    )
    print(f"Saved demo data -> {demo_path}  (shape={demo_X.shape})")
    print("\nDone. Commit the contents of data/ to your GitHub repo.")


if __name__ == "__main__":
    main()
