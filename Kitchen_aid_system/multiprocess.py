from multiprocessing import Process, Queue, Event
import time

sentinel = -1


def creator(q1, q2, flg1, flg2):
    """
    Create data to be consumed and waits for the consumer
    to finish processing
    """
    print('Creating data and putting it on the queue')
    i = 0
    while True:

        print("creator is running now")
        flg1.wait()

        while True:

            data = q1.get()
            processed = data * 2
            print('creator data', processed)
            q2.put(processed)

            if data is sentinel:
                break

        print("send edited data")
        flg2.set()
        time.sleep(0.1)


def consumer(q1, q2, flg1, flg2, frame):

    while True :
        if not flg2.is_set():
            data = frame
            print('flg2 is 0')

        elif flg2.is_set() :
            data = q2.get()
            print("data receive")
            flg1.clear()
            flg2.clear()

        if not flg1.is_set() :
            for i in frame:
                q1.put(i)
            flg1.set()
            print("flg1 is set")

        for i in data :
            print('data found to be processed: {}'.format(i))
        time.sleep(0.1)


if __name__ == '__main__':

    q1 = Queue()
    q2 = Queue()
    flg1 = Event()
    flg2 = Event()

    data = [5, 10, 13, -1]

    process_one = Process(target=creator, args=(q1, q2, flg1, flg2))
    process_two = Process(target=consumer, args=(q1, q2, flg1, flg2, data))

    process_one.start()
    process_two.start()
    process_one.join()
    process_two.join()

    q1.close()
    q1.join_thread()
    q2.close()
    q2.join_thread()
