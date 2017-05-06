import logging

from toyosatomimi import JobQueueServer


def main():
    logging.basicConfig(level=logging.INFO)
    server = JobQueueServer()
    server.serve()


if __name__ == '__main__':
    main()
