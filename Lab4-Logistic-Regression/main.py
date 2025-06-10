import os
import sys

from task1.task1_earthquake import task1_earthquake
from task2.task2 import task2_simulation

sys.path.append(os.path.join(os.path.dirname(__file__), 'task1'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'task2'))

EARTHQUAKE_FILE = os.path.join(os.path.dirname(__file__), 'earthquake.txt')

def main():
    task1_earthquake(EARTHQUAKE_FILE)

    mse_full, mse_3vars = task2_simulation(L=100)

if __name__ == "__main__":
    main()
