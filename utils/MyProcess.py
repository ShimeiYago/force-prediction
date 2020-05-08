import time


class MyProcess:
    def __init__(self, func, totalstep:int):
        self.totalstep = totalstep
        self.interval = totalstep // 100
        self.starttime = time.time()
        self.func = func
    
    def __call__(self, step:int, *args):
        ret = self.func(*args)

        if step%self.interval==0 or step==self.totalstep+1:
            progress = int((step+1) / self.totalstep * 100)
            elapsed_time = time.time() - self.starttime
            print(f'\rProgress: {progress}% {elapsed_time:.1f}s', end='')
                
        return ret
