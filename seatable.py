import datetime
import json
import os
import sys
import random
import socket
import subprocess
import time
from collections import Counter
from copy import deepcopy
from typing import Tuple, Iterable

import colorama
from retrying import retry
from seatable_api import Base

ST_LOG_FILENAME = 'seatable_log.json'


class STLogger:
    def __init__(self, exp_dir_abs_path, exp_name):
        self.log_path = os.path.join(exp_dir_abs_path, ST_LOG_FILENAME)
        self.st_kw = dict(abs_path=exp_dir_abs_path, exp=exp_name)
    
    @retry(stop_max_attempt_number=5, wait_fixed=500)
    def log(self, **kw):
        for k, v in kw.items():
            if isinstance(v, Iterable):
                kw[k] = str(v)
        
        changed = not all(k in self.st_kw and self.st_kw[k] == v for k, v in kw.items())
        if changed:
            self.st_kw.update(kw)
            with open(self.log_path, 'w') as fp:
                json.dump(self.st_kw, fp)


class STListener:
    ST_TAGS = [
        'tune',
    ]
    
    def __init__(self, exp_dir_abs_path, api_token='e5d0b6f80aa309ae6e12f435323c105e7b24da85', sheet_name='main'):
        self.exp_dir_abs_path = exp_dir_abs_path
        self.log_path = os.path.join(exp_dir_abs_path, ST_LOG_FILENAME)
        self.term_path = exp_dir_abs_path + '.terminate'
        
        server_url = 'https://cloud.seatable.cn'
        self.base = Base(api_token, server_url)
        self.base.auth()
        
        self.sheet_name = sheet_name
    
    def keep_listen(self):
        colorama.init(autoreset=True)
        
        while not os.path.exists(self.log_path):
            time.sleep(120)
            print(colorama.Fore.GREEN + f'[STListener] waiting for the seatable log file at {self.log_path} ...')
            if os.path.exists(self.term_path):
                os.remove(self.term_path)
                print(colorama.Fore.CYAN + '[STListener] terminated.')
                exit(-1)
        
        tb_port = self.get_tensorboard_port()
        tb_ip_port = f'{socket.gethostbyname(socket.gethostname())}:{tb_port}'
        print(colorama.Fore.LIGHTBLUE_EX + f'[STListener] tensorboard ip:port  =  {tb_ip_port}')
        cmd = f'tensorboard --logdir {self.exp_dir_abs_path} --port {tb_port} --bind_all'
        sp = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, bufsize=-1)
        
        with open(self.log_path, 'r') as fp:
            last_st_kw = json.load(fp)
            last_st_kw['tb'] = tb_ip_port
        
        try:
            upd_cnt = 0
            rid = None
            while True:
                if os.path.exists(self.term_path):
                    os.remove(self.term_path)
                    print(colorama.Fore.CYAN + '[STListener] terminated; use `sh ./kill.sh` to kill the tensorboard process')
                    exit(-1)  # the subprocess will become an orphan process; use `sh ./kill.sh` to kill it
                
                time.sleep(20)
                try:
                    with open(self.log_path, 'r') as fp:
                        cur_st_kw = json.load(fp)
                        cur_st_kw['tb'] = tb_ip_port
                except json.decoder.JSONDecodeError:
                    continue
                
                if rid is not None and cur_st_kw == last_st_kw:
                    continue
                last_st_kw = cur_st_kw
                
                rid, created = self.__create_or_upd_a_line(rid=rid, st_kw=cur_st_kw)
                
                if upd_cnt % 10 == 0 or created:
                    print(colorama.Fore.LIGHTBLUE_EX + f'[STListener] {"new line created" if created else "a line updated"}')
                upd_cnt += 1
                
                if cur_st_kw['pr'] > 1 - 1e-6:
                    print(colorama.Fore.CYAN + '[STListener] finished (100%).')
                    break
        
        except Exception as e:
            sp.kill()
            raise e
        
        sp.kill()
        time.sleep(1)
        return
    
    def __create_or_upd_a_line(self, rid, st_kw) -> Tuple[str, bool]:
        """
        if `rid` is None: create a new line via `st_kw`;
        if `rid` is not None: upd an existing line via `st_kw`.
        :returns
            (str): rid
            (bool): created a new line or not
        """
        new_kw = deepcopy(st_kw)
        new_kw['last_upd'] = STListener.get_cur_time()
        new_kw['tags'] = []
        for k, v in st_kw.items():
            if k in STListener.ST_TAGS and v == True:
                new_kw['tags'].append(k)
                new_kw.pop(k)
        
        if rid is None:
            q = self.base.filter(self.sheet_name, f"abs_path='{new_kw['abs_path']}'")
            if q.exists():
                q.update(new_kw)
                return q.get()['_id'], False
            else:
                created_rid = self.base.append_row(self.sheet_name, new_kw)['_id']
                return created_rid, True
        else:
            try:
                self.base.update_row(self.sheet_name, rid, new_kw)
                ret = rid, False
            except ConnectionError:
                ret = self.base.append_row(self.sheet_name, new_kw)['_id'], True
            return ret
    
    @staticmethod
    def get_cur_time():
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    
    @staticmethod
    def get_tensorboard_port():
        used_ports = os.popen("netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'").read()
        ava_ports = set(range(10000, 20000)) - set(map(int, used_ports.split()))
        ava_ports = sorted(list(ava_ports), key=lambda x: -10 * str(x).count('0') - max(Counter(str(x)).values()))
        top = max(round(0.05 * len(ava_ports)), 1)
        return random.choice(ava_ports[:top])


if __name__ == '__main__':
    stl = STListener(sys.argv[1])
    stl.keep_listen()
