import pickle
import numpy as np

with open("checkpoints/buffer.pkl", "rb") as f:
    data, ptr, full = pickle.load(f)

values = np.array([s[2] for s in data])
print(f"Всего позиций: {len(values)}")
print(f"\nРаспределение value-таргетов:")
print(f"  == 0.0  (ничья):       {(values == 0.0).sum():>8,}  ({100*(values==0.0).mean():.1f}%)")
print(f"  == 1.0  (победа бел):  {(values == 1.0).sum():>8,}  ({100*(values==1.0).mean():.1f}%)")
print(f"  == -1.0 (победа чёрн): {(values == -1.0).sum():>8,}  ({100*(values==-1.0).mean():.1f}%)")
print(f"\nMSE если всегда предсказывать 0: {(values**2).mean():.6f}")
print(f"Std таргетов: {values.std():.6f}")
