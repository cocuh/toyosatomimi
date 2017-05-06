import json
import logging
import sys

import zmq
import os

logger = logging.getLogger(__name__)


class JobQueueServer:
    def __init__(self, queue_path='queue.json', done_path='done.json', addr='tcp://127.0.0.1:5151'):
        context = zmq.Context()
        socket = context.socket(zmq.REP)

        self.socket = socket
        self.context = context
        self.addr = addr

        self.commands = {
            'put': self._put,
            'get': self._get,
            'done': self._done,
        }

        self.queue = []
        self.done = []
        self.queue_path = queue_path
        self.done_path = done_path

    def load_queue(self):
        with open(self.queue_path) as fp:
            jobs = json.load(fp)
            assert isinstance(jobs, list)
            self.queue = jobs
            job_num = len(self.queue)
            logger.info(f'server load job(s) num:{job_num}')

    def save_queue(self):
        with open(self.queue_path, 'w') as fp:
            json.dump(self.queue, fp)
            self.queue = []

    def serve(self):
        if os.path.exists(self.queue_path):
            self.load_queue()
        self._serve()
        self.save_queue()

    def _put(self, job):
        self.queue.append(job)
        return 'success', None

    def _get(self, _):
        if len(self.queue):
            data = self.queue.pop(0)
            return 'success', data
        else:
            return 'failure', None

    def _done(self, job):
        self.done.append(job)
        with open(self.done_path, 'w') as fp:
            json.dump(self.done, fp)
        return 'success', None

    def _serve(self):
        self.socket.bind(self.addr)
        logger.info(f'server listen:{self.addr}')
        try:
            while True:
                logger.info(f'server waiting...')
                req = self.socket.recv_json()
                logger.info(f'server get: {req}')
                command = req.get('command', None)
                func = self.commands.get(command, None)
                if func is None:
                    logger.error(f'server command not found: {command}')
                    res_status = 'failure'
                    res_data = None
                else:
                    req_data = req.get('data', None)
                    logger.info(f'server do command: {command} data:{req_data}')
                    res_status, res_data = func(req_data)
                    logger.info(f'server done command: {command} status:{res_status} data:{res_data}')
                res = {
                    'status': res_status,
                    'data': res_data,
                }
                logger.info(f'server send json:{res}')
                self.socket.send_json(res)

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            self.socket.close()


class JobFeeder:
    def __init__(self, addr='tcp://127.0.0.1:5151'):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)

        self.socket = socket
        self.context = context
        self.addr = addr

    def feed(self, generator):
        self.socket.connect(self.addr)
        for job in generator:
            print('send', job)
            self.socket.send_json({
                'command': 'put',
                'data': job,
            })
            self.socket.recv_json()


class Worker:
    def __init__(self, name, addr='tcp://127.0.0.1:5151'):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)

        self.socket = socket
        self.context = context
        self.addr = addr

        self.name = name

    def run(self):
        logger.info(f'worker:{self.name} start running')
        logger.info(f'worker:{self.name} listen addr: {self.addr}')
        self.socket.connect(self.addr)
        while True:
            self.socket.send_json({'command': 'get'})
            response = self.socket.recv_json()
            logger.info(f'worker:{self.name} get response:{response}')
            if not isinstance(response, dict):
                logger.info(f'worker:{self.name} get None, exiting...')
                break
            is_success = response.get('status', '') == 'success'
            if is_success:
                kwargs = response.get('data', {})
                logger.info(f'worker:{self.name} do with:{kwargs}')
                try:
                    result = self.do(**kwargs)
                except KeyboardInterrupt:
                    logger.info(f'worker:{self.name} re-put:{kwargs}')
                    self.socket.send_json({
                        'command': 'put',
                        'data': kwargs,
                    })
                    logger.info(f'worker:{self.name} re-put done')
                    break
                except Exception as e:
                    logger.info(f'worker:{self.name} re-put:{kwargs}')
                    self.socket.send_json({
                        'command': 'put',
                        'data': kwargs,
                    })
                    logger.info(f'worker:{self.name} re-put done')
                    raise e
                self.socket.send_json({
                    'command': 'done',
                    'data': kwargs,
                })
                self.socket.recv_json()
                logger.info(f'worker:{self.name} done status:{result} kwargs:{kwargs}')
        logger.info(f'worker:{self.name} end running')

    def do(self, **kwargs):
        raise NotImplementedError()


def serve():
    print('server mode')
    server = JobQueueServer()
    server.serve()


def worker():
    print('worker mode')
    worker = Worker('worker')
    worker.run()


def main():
    logging.basicConfig(level=logging.INFO)
    mode = sys.argv[1]
    func = {
        'server': serve,
        'worker': worker,
    }.get(mode)
    func()


if __name__ == '__main__':
    main()
