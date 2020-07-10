import os
import subprocess

matrixList = []
for root, dirs, files in os.walk("../matrix_large/"):
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

with open('./result.csv', mode='w') as f:
    f.write("matrix,time,time(scl),flop,flop(scl),byte,byte(scl)\n")
    for res in result: 
        f.write(res)
