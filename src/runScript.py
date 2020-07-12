import os
import subprocess

matrixList = []
for root, dirs, files in os.walk("../matrix_large/"):
    for filename in files:
        matrixList.append(os.path.join(root, filename)) 

threads = [8, 16, 32, 64, 128]
for th in threads:
    result = []
    for matrix in matrixList:
        command = str('./spmv ') + str(matrix) + " " + str(th)
        print(command)
        try:
            res = subprocess.check_output(command, shell=True)
        except:
            print 'Error.'
        result.append(res)

    fname = "./data-vs/result" + str(th) + ".csv"
    with open(fname, mode='w') as f:
        f.write("matrix,time(scl),time(vec),time(cublas),flop(vec),flop(cublas),flop(scl),byte(scl),byte(vec),byte(cublas)\n")
        for res in result:
            f.write(res)
    pvalue = "thread num. " + str(th) + " is completed."
    print pvalue
