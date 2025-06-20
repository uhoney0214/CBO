import pandas as pd
import re

# 1. 기존 데이터 로드
df = pd.read_csv("100sampledata.txt", sep="\s+", engine="python", header=None)

# 2. 열 이름 지정 (1-3.py와 동일)
df.columns = [
    "h_coil","r_plunger","z_plunger","h_plunger",
    "g_air","N","I0","Time","force","B"
]

# 3. result.txt에서 6번째 줄만 추출하여 숫자만 읽기
with open("result.txt", "r") as f:
    lines = f.readlines()
    if len(lines) < 6:
        raise ValueError("result.txt 파일 줄 수가 6줄 미만입니다.")
    data_line = lines[5]  # 0-based index이므로 6번째 줄은 lines[5]
    # 정규표현식으로 숫자만 추출 (실수/정수 모두 지원)
    nums = re.findall(r'[-+]?\d*\.\d+|\d+', data_line)
    vals = [float(x) for x in nums]

assert len(vals) == 10, f"result.txt에서 숫자 10개를 찾지 못했습니다. (실제 추출 개수: {len(vals)})"

# 4. 행 추가 및 파일 저장
df.loc[len(df)] = vals
df.to_csv("100sampledata.txt", sep="\t", index=False, header=False)

print(f"✅ result.txt 6번째 줄 데이터가 추가되었습니다. (총 {len(df)}개 샘플)")
