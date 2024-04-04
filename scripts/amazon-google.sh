
CUDA_VISIBLE_DEVICES=6 python main.py --data amazon-google --strategy base --verbose
CUDA_VISIBLE_DEVICES=7 python main.py --data amazon-google --strategy similarity --verbose
CUDA_VISIBLE_DEVICES=6 python main.py --data amazon-google --strategy uncertainty --uncertainty_func bin --verbose
CUDA_VISIBLE_DEVICES=7 python main.py --data amazon-google --strategy uncertainty --uncertainty_func cat --verbose
CUDA_VISIBLE_DEVICES=6 python main.py --data amazon-google --strategy diversity --verbose