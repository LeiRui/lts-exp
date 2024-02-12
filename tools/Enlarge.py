import argparse
import csv
import pandas as pd

parser = argparse.ArgumentParser(description="enlarge",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input")
parser.add_argument("-o", "--output")
parser.add_argument("-r", "--ratio")
args = parser.parse_args()
config = vars(args)
inputFile = str(config.get('input'))
print(inputFile)
outputFile = str(config.get('output'))
ratio = int(config.get('ratio'))
# (N-1)*r

df = pd.read_csv(inputFile, header=None)

batch = 100  # 原来每batch个点扩充成batch*ratio个点之后就先写出去，避免内存消耗过大

globalT = 0

with open(outputFile, 'w', encoding='UTF8', newline='') as f:
  writer = csv.writer(f)

  V = [0] * (batch * ratio)
  batchNum = 0
  for i in range(len(df) - 1):
    v1 = df.iloc[i, 1]
    v2 = df.iloc[i + 1, 1]
    deltaV = v2 - v1
    V[batchNum] = v1
    batchNum += 1
    for j in range(ratio - 1):  # 0,1,...,ratio-2
      V[batchNum] = v1 + deltaV / ratio * (j + 1)
      batchNum += 1
    if batchNum >= batch * ratio:
      for i in range(batchNum):
        globalT += 1
        writer.writerow([globalT, V[i]])
      V = [0] * (batch * ratio)  # 准备下一个batch
      batchNum = 0  # 准备下一个batch

  # 最后一点尾巴不满一个batch的也写出去
  if batchNum > 0:
    for i in range(batchNum):  # 注意这里是实际的batchNum有可能小于batch * ratio
      globalT += 1
      writer.writerow([globalT, V[i]])

print(globalT)

# t1=t1.reshape(len(v1),1)
# v1=v1.reshape(len(v1),1)
# arr = np.hstack([t1, v1])
# pd.DataFrame(arr).to_csv(outputFile, index=False, header = False)

# with open(outputFile, 'w', encoding='UTF8', newline='') as f:
#   writer = csv.writer(f)
#   for i in range(num):
#     writer.writerow([i + 1, V[i]])
