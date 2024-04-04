

CUDA_VISIBLE_DEVICES=0 python main.py --data walmart-amazon --strategy base --verbose
CUDA_VISIBLE_DEVICES=0 python main.py --data walmart-amazon --strategy similarity --verbose
CUDA_VISIBLE_DEVICES=0 python main.py --data walmart-amazon --strategy uncertainty --uncertainty_func bin --verbose
CUDA_VISIBLE_DEVICES=1 python main.py --data walmart-amazon --strategy uncertainty --uncertainty_func cat --verbose
CUDA_VISIBLE_DEVICES=1 python main.py --data walmart-amazon --strategy diversity --verbose