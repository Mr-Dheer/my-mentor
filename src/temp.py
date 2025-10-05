import numpy as np

U = np.load("/home/kavach/Dev/NewResearch/MENTOR/src/exported_embeddings/MENTOR-baby/user_all.npy")   # [n_users, d]
V = np.load("/home/kavach/Dev/NewResearch/MENTOR/src/exported_embeddings/MENTOR-baby/item_all.npy")   # [n_items, d]

print("U:", U.shape, U.dtype, np.isfinite(U).all())
print("V:", V.shape, V.dtype, np.isfinite(V).all())
assert U.ndim == 2 and V.ndim == 2 and U.shape[1] == V.shape[1]

u_norm = np.linalg.norm(U, axis=1)
v_norm = np.linalg.norm(V, axis=1)
print("user norm: mean=%.3f std=%.3f min=%.3f max=%.3f" % (u_norm.mean(), u_norm.std(), u_norm.min(), u_norm.max()))
print("item norm: mean=%.3f std=%.3f min=%.3f max=%.3f" % (v_norm.mean(), v_norm.std(), v_norm.min(), v_norm.max()))
# Red flags: means near 0, huge std, many zeros, or NaNs/Infs.

import pandas as pd, os

# load the train split to get user->pos items (indices must match your internal IDs)
inter_path = os.path.abspath("../data/baby/baby.inter")  # adjust if needed
df = pd.read_csv(inter_path, sep="\t", usecols=["userID","itemID","x_label"])
train_df = df[df["x_label"]==0][["userID","itemID"]].copy()

# build user->set(items) mapping from train
train_pos = {}
for u, grp in train_df.groupby("userID"):
    train_pos[u] = set(grp["itemID"].values)

def topk_for_user(u, k=10):
    scores = V @ U[u]        # dot product
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return top

def hit_at_k(u, k=10):
    if u not in train_pos or len(train_pos[u])==0:
        return None
    pred = set(topk_for_user(u, k))
    return len(pred & train_pos[u]) / min(k, len(train_pos[u]))

sample_users = [0, 1, 2, 123, 456]
hits = [hit_at_k(u, 10) for u in sample_users if hit_at_k(u, 10) is not None]
print("Sanity Recall@10 (on train positives) for sample users:", hits)
# Typical values won’t be huge on train, but should be > random. If most are ~0, check checkpoint/path.
# Optional: re-use your project components for a closer match
from utils_package.configurator import Config
from utils_package.dataset import RecDataset
from utils_package.dataloader import EvalDataLoader

cfg = Config("MENTOR", "baby", {})
full_ds = RecDataset(cfg)
train_ds, valid_ds, _ = full_ds.split()

valid_loader = EvalDataLoader(cfg, valid_ds, additional_dataset=train_ds, batch_size=4096)

import torch
U_t = torch.from_numpy(U).float()
V_t = torch.from_numpy(V).float()

def full_sort_predict_batch(users_batch):
    # users_batch: tensor of user IDs
    return U_t[users_batch, :] @ V_t.T  # [batch, n_items]

# emulate Trainer.evaluate()’s core
batch_matrix_list = []
for users_batch, masked_matrix in valid_loader:
    scores = full_sort_predict_batch(users_batch)
    # mask out train positives
    scores[masked_matrix[0], masked_matrix[1]] = -1e10
    _, topk_index = torch.topk(scores, k=max(cfg['topk']), dim=-1)
    batch_matrix_list.append(topk_index)

from utils_package.topk_evaluator import TopKEvaluator
evaluator = TopKEvaluator(cfg)
metrics = evaluator.evaluate(batch_matrix_list, valid_loader, is_test=False, idx=0)
print(metrics)
# Should be reasonably close to what you saw during training. If wildly off, check that you loaded the correct CKPT.

import random
def pos_vs_neg(u):
    if u not in train_pos or len(train_pos[u])==0: return None
    pos = random.choice(list(train_pos[u]))
    neg = random.randrange(V.shape[0])
    while neg in train_pos[u]:
        neg = random.randrange(V.shape[0])
    s_pos = U[u] @ V[pos]
    s_neg = U[u] @ V[neg]
    return s_pos, s_neg

vals = [pos_vs_neg(u) for u in sample_users]
print(vals)  # Usually s_pos > s_neg for many trials; if consistently not, checkpoint mismatch or export issue.

def l2(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True); n = np.clip(n, eps, None); return x / n

u_all = l2(U)
u_v = l2(np.load("/home/kavach/Dev/NewResearch/MENTOR/src/exported_embeddings/MENTOR-baby/user_v.npy"))
u_t = l2(np.load("/home/kavach/Dev/NewResearch/MENTOR/src/exported_embeddings/MENTOR-baby/user_t.npy"))

cos_v = (u_all * u_v).sum(axis=1).mean()
cos_t = (u_all * u_t).sum(axis=1).mean()
print("avg cos(user_all, user_v) =", cos_v, " | avg cos(user_all, user_t) =", cos_t)
# Expect moderate positive values; exactly ~0 indicates something’s off.

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2, random_state=0)
pts = pca.fit_transform(U[np.random.choice(U.shape[0], size=min(3000, U.shape[0]), replace=False)])
plt.scatter(pts[:,0], pts[:,1], s=3)
plt.title("PCA of user_all")
plt.show()

print("Loading checkpoint from:", CKPT)
