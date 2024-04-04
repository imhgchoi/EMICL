
CUDA_VISIBLE_DEVICES=1 python main.py --data dblp-acm --strategy base --verbose
CUDA_VISIBLE_DEVICES=1 python main.py --data dblp-acm --strategy similarity --verbose
CUDA_VISIBLE_DEVICES=1 python main.py --data dblp-acm --strategy uncertainty --uncertainty_func bin --verbose
CUDA_VISIBLE_DEVICES=3 python main.py --data dblp-acm --strategy uncertainty --uncertainty_func cat --verbose
CUDA_VISIBLE_DEVICES=3 python main.py --data dblp-acm --strategy diversity --verbose