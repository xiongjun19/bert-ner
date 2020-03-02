# CUDA_VISIBLE_DEVICES=3  nohup  python train.py --model_dir models_custom --batch_size 32 --top_rnns --lr 1e-3 --n_epochs 302 --corpus_opt custom > log_cus.txt &
CUDA_VISIBLE_DEVICES=3  nohup python ./train.py --trainset ../../ZEN/datasets/custom/train.tsv --validset ../../ZEN/datasets/custom/test.tsv --batch_size 32 --top_rnns --lr 1e-3 --epochs 30 > log.txt &
