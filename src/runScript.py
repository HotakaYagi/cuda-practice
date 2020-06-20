import os
import subprocess

matrixList = []
for root, dirs, files in os.walk("../matrix/"):
    for filename in files:
        matrixList.append(os.path.join(root, filename)) 

result = []
for matrix in matrixList:
    command = str('./spmv ') + str(matrix)
    print(command)
    try:
        res = subprocess.check_output(command, shell=True)
    except:
        print 'Error.'
    result.append(res)

with open('./result.txt', mode='w') as f:
    for res in result: 
        f.write(res)
