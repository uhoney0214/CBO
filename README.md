# CBO
초기 샘플링은 LHS_sampling.ipynb 파일을 통해 실행시킨다.
BO의 경우, BO.py 실행시킨 후, COMSOL에 텍스트 파일을 import하고 시뮬레이션을 해 데이터를 얻는다.
그 후, export해 얻은 데이터를 Addsample.py를 실행시켜 데이터를 합친다.
이 과정을 수렴할 때 까지 계속 반복한다.
GA의 경우, GA.py 실행시킨 후, COMSOL에 텍스트 파일을 import하고 시뮬레이션을 해 데이터를 얻는다.
그 후, export해 얻은 데이터를 GA_addsample.py를 실행시켜 데이터를 합친다.
이 과정을 수렴할 때 까지 계속 반복한다.
