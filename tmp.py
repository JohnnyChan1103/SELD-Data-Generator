import pickle

with open('database/seld_FSD50K_10_ov1_train/metadata.obj', 'rb') as f:
    meta = pickle.load(f)

# 查看第一条音频 (mix 0)
mix_0 = meta['target_classes'][0]

# 打印第 100 帧（10秒处）的情况
frame_idx = 100
print(f"--- Frame {frame_idx} (10.0s) ---")
print(f"Class IDs: {mix_0['classid'][frame_idx]}")
print(f"MIDs     : {mix_0['mid'][frame_idx]}")
print(f"Track IDs: {mix_0['trackid'][frame_idx]}")
print(f"DOA (Azi, Ele, R): {mix_0['eventdoatimetracks'][frame_idx]}")