print("HI")
# guvcview

from bdb import Breakpoint
import threading
import time
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

class MyThread(threading.Thread):

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, time=None):
        super(MyThread,self).__init__(group=group, target=target,name=name)
        self.args = args
        self.kwargs = kwargs
        self.time = time
        self.list = {}
        return
        
    def run(self):
        logging.debug('start running with %s and %s', self.args, self.kwargs)
        # time.sleep(self.time)
        for i in range(self.time, 0, -1):
            l = {i: i}
            self.list.update(l)
            print(f"BOOOOOOOOOO {i}{self.list}")
        logging.debug('end running with %s and %s', self.args, self.kwargs)
        return

if __name__ == '__main__':
    j = 1
    for i in range(24000):
        t = MyThread(args=(i,), kwargs={'a':1, 'b':2}, time = j)
        j += 1
        t.start()
    
    while True:
        #print(t.is_alive())
        if t.is_alive():
            continue
        else:
            print(t.is_alive())
            break
        