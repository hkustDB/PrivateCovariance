from functions import parse_args, test_minist_fix_digit
from multiprocessing import Process
import time

pargs = parse_args()
pargs.delta = 1e-10
rhos = [0.0001,0.001,0.01,0.1,1,10]
pargs.mp = False
pargs.d = 784
pargs.n = 60000
digit = 0 #0, 1, 2
run_range = range(0,50)
folders = ['./results/test_mnist_digit0/test_run'+str(k)+'/' for k in run_range]
params = [['nPaths','rhos','n','N','s','beta','d','delta'],
[str(1),'|'.join([str(rhoi) for rhoi in rhos]),str(pargs.n),str(pargs.N),str(pargs.s),str(pargs.beta)],str(pargs.d),str(pargs.delta)]

if __name__ == '__main__':
    tick = time.time()
    print('Starting test_mnist_digit_zCDP for range '+str(run_range[0])+' - '+str(run_range[-1]))
    if pargs.mp:
        ps = []
        for i in run_range:
            ps.append(Process(target=test_minist_fix_digit,args=[pargs,rhos,folders[i-run_range[0]],params,digit]))
            ps[i-run_range[0]].start()
        for i in run_range:
            ps[i-run_range[0]].join()
    else:
        for i in run_range:
            print('test '+str(i))
            test_minist_fix_digit(pargs,rhos,folders[i-run_range[0]],params,digit)
    print('Finished. Time elapsed: ',time.time() - tick)