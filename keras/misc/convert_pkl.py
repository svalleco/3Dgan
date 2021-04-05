#!/usr/bin/env python3

import pickle

with open("../results/3dgan_history_newarch_k2.pkl", "rb") as f:
    w = pickle.load(f)
pickle.dump(w, open("3dgan_history_newarch_k2_py2.pkl","wb"), protocol=2)
