#! /bin/bash

python main.py --lr 0.1
python main.py --lr 0.01
python main.py --lr 0.001

python main.py --optim A3 --lr 0.1
python main.py --optim A3 --lr 0.01
python main.py --optim A3 --lr 0.001

python main.py --optim A5 --lr 0.1
python main.py --optim A5 --lr 0.01
python main.py --optim A5 --lr 0.001

python main.py --optim A11 --lr 0.1
python main.py --optim A11 --lr 0.01
python main.py --optim A11 --lr 0.001

python main.py --optim A3 --lr 0.1
python main.py --optim A3 --lr 0.01
python main.py --optim A3 --lr 0.001

python main.py --optim A5 --lr 0.1
python main.py --optim A5 --lr 0.01
python main.py --optim A5 --lr 0.001
