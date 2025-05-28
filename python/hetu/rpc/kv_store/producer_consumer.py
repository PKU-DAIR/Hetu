import time
import logging
import sys
import os
import multiprocessing
import hetu as ht
from concurrent.futures import ProcessPoolExecutor
from .client import KeyValueStoreClient

# Set the start method to 'spawn' at the very beginning
multiprocessing.set_start_method('spawn', force=True)

# Global variables to store client and data store in worker processes
_worker_client = None
_worker_data_store = None

def _worker_initializer(address, dict_name):
    """
    Initializer function for worker processes.
    Initializes the client and data store and stores them in global variables.
    """
    global _worker_client, _worker_data_store
    _worker_client = KeyValueStoreClient(address=address)
    _worker_data_store = _worker_client.register_dict(dict_name)
    logging.info(f"Worker process initialized with address {address} and dict {dict_name}")

def _producer_task(key, func, args, kwargs):
    """
    Function executed by worker processes.
    Uses the globally initialized client and data store in the process.
    Runs the user-provided function and stores the result in the data store.
    """
    global _worker_data_store
    try:
        # Execute the function
        result = func(*args, **kwargs)
        # Store the result
        _worker_data_store.put(key, result)
        return result
    except Exception as e:
        logging.error(f"Exception in _producer_task with key {key}: {e}")
        return e

class ProducerConsumer:
    def __init__(self, client, dict_name='prod_cons_store', max_workers=None):
        self.dict_name = dict_name
        self.client = client
        self.address = self.client.address
        self.mp_context = multiprocessing.get_context('spawn')
        self.executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=self.mp_context,
            initializer=_worker_initializer,
            initargs=(self.address, self.dict_name)
        )
        self.data_store = client.register_dict(dict_name)
        self.futures = []
        ht.setup_logging()
        logging.info("ProducerConsumer initialized.")

    def produce(self, key, func, *args, **kwargs):
        """Submit the task to the process pool."""
        future = self.executor.submit(_producer_task, key, func, args, kwargs)
        self.futures.append((key, future))

    def consume(self, key, global_barrier=True):
        """
        Retrieve and remove the data associated with the key.
        """
        try:
            data = self.data_store.get(key)
            if global_barrier:
                ht.global_comm_barrier_rpc()
            self.data_store.remove(key)
            return data
        except KeyError as e:
            logging.warning(f"Key does not exist: {key}")
            raise e

    def shutdown(self):
        """Shut down the executor and its worker processes."""
        for key, future in self.futures:
            try:
                result = future.result()
                if isinstance(result, Exception):
                    logging.error(f"Task {key} raised an exception: {result}")
            except Exception as e:
                logging.error(f"Task {key} raised an exception during future.result(): {e}")
        self.executor.shutdown(wait=True)
        logging.info("ProducerConsumer shutdown.")

    '''
    def __del__(self):
        """Ensure that the executor is shut down when the object is deleted."""
        self.shutdown()
    '''

# ------------------------------------------------------------
# Example usage

# Define compute_square at the global scope
def compute_square(n):
    return n * n

if __name__ == '__main__':
    client = KeyValueStoreClient(address='localhost:50051')
    prod_cons = ProducerConsumer(client, max_workers=2)  # Adjust workers as needed

    logging.info("Producing items...")
    for i in range(5):
        key = f'item_{i}'
        prod_cons.produce(key, compute_square, i)
        time.sleep(0.2)

    logging.info("Consuming items...")
    for i in range(5):
        key = f'item_{i}'
        item = None
        while item is None:
            try:
                item = prod_cons.consume(key)
            except KeyError:
                logging.info(f"Item {key} not ready yet. Waiting...")
                time.sleep(0.5)
        logging.info(f"Consumed {key}: {item}")

    # Shutdown the executor when done
    prod_cons.shutdown()
