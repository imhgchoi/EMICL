

# Environment
This is my environment I used for the ICML project; there can be some redundant modules :/

    conda env create -f environment.yml

# Datasets

Datasets are in "LLMForEM/data/". So you do not need to download data on your own!

# Experiment Example
To run base setting on the abt-buy dataset,

    CUDA_VISIBLE_DEVICES=0 python main.py --data abt-buy --strategy base

To run similarity baseline on the abt-buy dataset,

    CUDA_VISIBLE_DEVICES=0 python main.py --data abt-buy --strategy similarity

For Debug mode, use the "--debug" argument. This will reduce the number of data samples to 50, to speed up coding :)

    CUDA_VISIBLE_DEVICES=0 python main.py --data abt-buy --strategy similarity --debug
