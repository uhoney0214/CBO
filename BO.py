import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF
from sklearn.preprocessing import StandardScaler
import os
# 1. 데이터 전처리 - 변수 scaling 및 GP학습
df = pd.read_csv("100sampledata.txt", sep="\s+", engine="python")
df.columns = [
    "h_coil", "r_plunger", "z_plunger", "h_plunger",
    "g_air", "N", "I0", "Time", "force", "B"
]
df.drop(columns=["Time"], inplace=True)
features = ["h_coil", "r_plunger", "z_plunger", "h_plunger", "g_air", "N", "I0"]
X = df[features].values
y_force = df["force"].values
y_B     = df["B"].values
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(7), length_scale_bounds=(1e-2, 1e2))
gp_force = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-3, normalize_y=True).fit(X_scaled, y_force)
gp_B     = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-3, normalize_y=True).fit(X_scaled, y_B)
B_sat=1.6
# 2. 획득함수 정의
def cei_acquisition(x_orig, gp_f, y_best_max, gp_B, scaler_X, B_sat=1.6, xi=1.0):
    x_scaled = scaler_X.transform(np.atleast_2d(x_orig))
    mu_f, sigma_f = gp_f.predict(x_scaled, return_std=True)
    mu_B, sigma_B = gp_B.predict(x_scaled, return_std=True)
    imp = mu_f - y_best_max - xi
    Zf = imp / (sigma_f + 1e-12)
    ei = imp * norm.cdf(Zf) + sigma_f * norm.pdf(Zf)
    ei = np.where(imp > 0, ei, 0.0)
    pf = norm.cdf((B_sat - mu_B) / (sigma_B + 1e-12))
    return float((ei * pf)[0])
# 3. 다음 탐색 지점 함수 정의
def propose_location(
    gp_f, gp_B, y_best_max, bounds, scaler_X, X, y_force, y_B, tol=1e-6, perturb_ratio=0.03
):
    bounds_array = np.array(bounds)
    def acq(x):
        x_fixed = x.copy()
        # N(코일 권선 수) 정수 처리
        x_fixed[5] = int(round(x_fixed[5]))
        return -cei_acquisition(x_fixed, gp_f, y_best_max, gp_B, scaler_X)
    best_cei = -np.inf
    best_x = None
    # 랜덤 스타트 여러 번 local opt
    for _ in range(10):
        x0 = []
        for i, (lo, hi) in enumerate(bounds_array):
            if features[i] == "N":
                x0.append(np.random.randint(lo, hi + 1))
            else:
                x0.append(np.random.uniform(lo, hi))
        x0 = np.array(x0, float)
        res = minimize(acq, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.success:
            x_cand = res.x.copy()
            x_cand[5] = int(round(x_cand[5]))
            # 범위 보정
            for i, (lo, hi) in enumerate(bounds_array):
                if features[i] == "N":
                    x_cand[i] = np.clip(int(round(x_cand[i])), lo, hi)
                else:
                    x_cand[i] = np.clip(x_cand[i], lo, hi)
            cei_val = -res.fun
            if cei_val > best_cei:
                best_cei = cei_val
                best_x = x_cand                
    #못 찾으면(best_cei ≤ tol) best force point 근처 3% perturbation
    if best_cei <= tol or best_x is None:
        valid_idx = np.where(y_B < B_sat)[0]
        if len(valid_idx) == 0:
            best_idx = np.argmax(y_force)
        else:
            best_idx = valid_idx[np.argmax(y_force[valid_idx])]
        best_pt = X[best_idx]
        perturbed_pt = []
        for i, ((lo, hi), val) in enumerate(zip(bounds_array, best_pt)):
            range_width = hi - lo
            delta = np.random.uniform(-perturb_ratio, perturb_ratio) * range_width
            new_val = val + delta
            if features[i] == "N":
                new_val = int(round(new_val))
                new_val = np.clip(new_val, lo, hi)
            else:
                new_val = np.clip(new_val, lo, hi)
            perturbed_pt.append(new_val)
        best_x = np.array(perturbed_pt, float)
        best_cei = cei_acquisition(best_x, gp_f, y_best_max, gp_B, scaler_X)
    return best_x, best_cei
# history 파일 세팅
history_path = "CEI_history.txt"
header = "iteration best_so_far ECI\n"
if not os.path.exists(history_path):
    with open(history_path, "w", encoding="utf-8") as f:
        f.write(header)
with open(history_path, "r", encoding="utf-8") as f:
    lines = f.readlines()
iteration = len(lines) if lines and lines[0].startswith("iteration") else len(lines) + 1
# best_so_far 계산
def get_best_so_far(history_path, force_candidate, B_candidate):
    best = -np.inf
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("iteration"): continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        val = float(parts[1])
                        if val > best:
                            best = val
                    except:
                        pass
                    best = max(best, y_best_max)
    if B_candidate < B_sat and force_candidate > best:
        best = force_candidate
    return best if best > -np.inf else "None"
valid_idx = np.where(y_B < B_sat)[0]
if len(valid_idx) > 0:
    y_best_max = y_force[valid_idx].max()
else:
    y_best_max = y_force.max()
bounds_list = [(5,75), (10,40), (11,25), (10,50), (0.2,3), (100,600), (2,10)]
x_next, cei_val = propose_location(gp_force, gp_B, y_best_max, bounds_list, scaler_X, X, y_force, y_B)
# 4)탐색 지점에서의 예측 force, B값
mu_f, _ = gp_force.predict(scaler_X.transform([x_next]), return_std=True)
mu_B, _ = gp_B.predict(scaler_X.transform([x_next]), return_std=True)
mu_f, mu_B = float(mu_f[0]), float(mu_B[0])
# 5) best_so_far(force, B<B_sat만)로 갱신
best_so_far = get_best_so_far(history_path, mu_f, mu_B)
# 6) comsol_ready.txt에 추천점 저장
cols = features
units = {c: "mm" for c in cols}
units["N"] = ""
units["I0"] = "A"
with open("comsol_ready.txt", "w") as f:
    for col, v in zip(cols, x_next):
        f.write(f'{col} "{v:.10f}" [{units[col]}]\n')
print("✅ comsol_ready.txt 파일 생성 완료.")
# 7) 기록
row = f"{iteration} {best_so_far} {cei_val}\n"
with open(history_path, "a", encoding="utf-8") as f:
    f.write(row)
print(f"[INFO] CEI_history.txt 파일 생성 완료.")
