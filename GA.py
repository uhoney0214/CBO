import numpy as np
import random
from deap import base, creator, tools
import re

# 변수/단위 정보 (컬럼 순서 고정)
var_names = ["h_coil", "r_plunger", "z_plunger", "h_plunger", "g_air", "N", "I0"]
units = {"h_coil":"mm", "r_plunger":"mm", "z_plunger":"mm", "h_plunger":"mm", "g_air":"mm", "N":"", "I0":"A"}
B_sat = 1.6
input_filename = "result.txt"    # 숫자만 저장된 파일명
output_filename = "GA_next_generation.txt"
desired_num = 60  # 생성할 offspring 수

def parse_comsol_results(filename):
    pop = []
    fitness = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith("%"):
                continue
            parts = [float(x) for x in re.split(r"\s+", line_strip) if x]
            if len(parts) < 10:
                continue
            # 설계변수 7개, Force(8번), B(9번) 고정
            x = [parts[i] if var_names[i] != "N" else int(parts[i]) for i in range(7)]
            force = parts[8]
            B_val = parts[9]
            pop.append(x)
            fitness.append(force if B_val < B_sat else -999)
    print(f"[INFO] 읽힌 데이터 개수: {len(pop)}")
    return np.array(pop), np.array(fitness)

# DEAP GA 설정
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
bounds = [
    (5,75), (10,40), (11,25), (10,50), (0.2,3), (100,600), (2,10)
]
for i, (lo, hi) in enumerate(bounds):
    if var_names[i] == "N":
        toolbox.register(f"attr_{i}", random.randint, int(lo), int(hi))
    else:
        toolbox.register(f"attr_{i}", random.uniform, lo, hi)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2,
                  toolbox.attr_3, toolbox.attr_4, toolbox.attr_5, toolbox.attr_6), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20,
                 low=[b[0] for b in bounds], up=[b[1] for b in bounds], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# 1. 실험 결과 불러오기 및 fitness 계산
pop_array, fit_array = parse_comsol_results(input_filename)
pop = []
for i in range(len(pop_array)):
    ind = creator.Individual([pop_array[i,j] if var_names[j] != "N" else int(pop_array[i,j]) for j in range(7)])
    ind.fitness.values = (fit_array[i],)
    pop.append(ind)

if not pop:
    raise RuntimeError("실험 데이터가 한 개도 파싱되지 않았습니다. 파일 포맷/줄바꿈/공백 확인!")

# 2. 현재 세대 best_ind 출력
best_ind = max(pop, key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -np.inf)
print("\n✅ 현재 세대의 최적 force:", best_ind.fitness.values[0], "\n설계:", list(best_ind))

# 3. 다음 세대 생성 (GA 연산)
offspring = toolbox.select(pop, desired_num)
offspring = list(map(toolbox.clone, offspring))
for c1, c2 in zip(offspring[::2], offspring[1::2]):
    if random.random() < 0.7:
        toolbox.mate(c1, c2)
        del c1.fitness.values, c2.fitness.values
for mut in offspring:
    if random.random() < 0.3:
        toolbox.mutate(mut)
        del mut.fitness.values

# 4. clip/정수 보정 및 개체 수 보장
for ind in offspring:
    for i, (lo, hi) in enumerate(bounds):
        if var_names[i] != "N":
            ind[i] = float(np.clip(ind[i], lo, hi))
        else:
            ind[i] = int(np.clip(round(ind[i]), lo, hi))

# 부족하면 추가 selection/clone해서 정확히 desired_num 개로 맞춤
while len(offspring) < desired_num:
    needed = desired_num - len(offspring)
    more = toolbox.select(pop, needed)
    more = list(map(toolbox.clone, more))
    for ind in more:
        for i, (lo, hi) in enumerate(bounds):
            if var_names[i] != "N":
                ind[i] = float(np.clip(ind[i], lo, hi))
            else:
                ind[i] = int(np.clip(round(ind[i]), lo, hi))
    offspring += more

# 5. txt로 내보내기
vals_by_var = {vn: [ind[i] for ind in offspring[:desired_num]] for i, vn in enumerate(var_names)}
with open(output_filename, "w", encoding="utf-8") as f:
    for vn in var_names:
        vals = vals_by_var[vn]
        if vn == "N":
            vals_str = ", ".join([f"{int(round(v))}" for v in vals])
        else:
            vals_str = ", ".join([f"{v:.10f}" for v in vals])
        f.write(f'{vn} "{vals_str}" [{units[vn]}]\n')
print(f"✅ {output_filename} 생성 완료 (FEM 다음 입력, {desired_num}개)")
