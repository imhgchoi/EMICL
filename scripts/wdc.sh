

CUDA_VISIBLE_DEVICES=0 python main.py --data wdc --strategy base --verbose
CUDA_VISIBLE_DEVICES=0 python main.py --data wdc --strategy similarity --verbose
CUDA_VISIBLE_DEVICES=0 python main.py --data wdc --strategy uncertainty --uncertainty_func bin --verbose
CUDA_VISIBLE_DEVICES=2 python main.py --data wdc --strategy uncertainty --uncertainty_func cat --verbose
CUDA_VISIBLE_DEVICES=2 python main.py --data wdc --strategy diversity --verbose