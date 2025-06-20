import pandas as pd
import re

# 1. 기존 데이터 로드
df = pd.read_csv("GA60.txt", sep="\s+", engine="python", header=None)

# 2. 열 이름 지정 (1-3.py와 동일)
df.columns = [
    "h_coil","r_plunger","z_plunger","h_plunger",
    "g_air","N","I0","Time","force","B"
]

with open("result.txt", "r") as f:
    lines = f.readlines()
    if len(lines) < 65:
        raise ValueError("result.txt 파일 줄 수가 55줄 미만입니다.")
    for idx, data_line in enumerate(lines[5:65], start=6):
        nums = re.findall(r'[-+]?\d*\.\d+|\d+', data_line)
        vals = [float(x) for x in nums]
        assert len(vals) == 10, f"{idx}번째 줄에서 숫자 10개를 찾지 못했습니다. (실제 추출 개수: {len(vals)})"
        df.loc[len(df)] = vals

df.to_csv("GA60.txt", sep="\t", index=False, header=False)
print(f"✅ result.txt 6~55번째 줄 데이터가 추가되었습니다. (총 {len(df)}개 샘플)")
