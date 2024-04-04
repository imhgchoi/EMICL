
CUDA_VISIBLE_DEVICES=6 python main.py --data abt-buy --strategy base --verbose
CUDA_VISIBLE_DEVICES=7 python main.py --data abt-buy --strategy similarity --verbose
CUDA_VISIBLE_DEVICES=6 python main.py --data abt-buy --strategy uncertainty --uncertainty_func bin --verbose
CUDA_VISIBLE_DEVICES=7 python main.py --data abt-buy --strategy uncertainty --uncertainty_func cat --verbose
CUDA_VISIBLE_DEVICES=6 python main.py --data abt-buy --strategy diversity --verbose