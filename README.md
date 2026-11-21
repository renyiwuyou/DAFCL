## Requirements

python==3.10.16
torch==2.4.0
numpy==1.26.4
gensim==4.4.0


## Run the Code

To save time, we provide the original pre-trained model, which is the first stage in the paper. The model can be loaded by running the commands and moving to the second stage to further improve the model performance using our method.

On Beauty:
python main.py --data_name=Beauty --model_idx=1 --load_pretrain --beta=0.4 --attention_probs_dropout_prob=0.1 --hidden_dropout_prob=0.1 --n_pairs=2 --n_whole_level=3 --rate_min=0.2 --rate_max=0.71 --hard_negative_ratio=1

You are expected to get following results after training:
{'Epoch': 0, 'HIT@5': '0.0656', 'NDCG@5': '0.0457', 'HIT@10': '0.0896', 'NDCG@10': '0.0535', 'HIT@20': '0.1198', 'NDCG@20': '0.0610'}

On Sports_and_Outdoors:
python main.py  --data_name=Sports_and_Outdoors --model_idx=1 --load_pretrain --attention_probs_dropout_prob=0.1 --hidden_dropout_prob=0.1 --n_pairs=2 --contrast_tau=0.3 --n_whole_level=3 --rate_min=0.1 --rate_max=0.81

You are expected to get following results after training:
{'Epoch': 0, 'HIT@5': '0.0328', 'NDCG@5': '0.0220', 'HIT@10': '0.0493', 'NDCG@10': '0.0273', 'HIT@20': '0.0710', 'NDCG@20': '0.0328'}

On Toys_and_Games:
python main.py  --data_name=Toys_and_Games --model_idx=1 --load_pretrain --beta=0.4 --attention_probs_dropout_prob=0.1 --hidden_dropout_prob=0.1 --n_pairs=2 --n_whole_level=3 --rate_min=0.2 --rate_max=0.71

You are expected to get following results after training:
{'Epoch': 0, 'HIT@5': '0.0666', 'NDCG@5': '0.0468', 'HIT@10': '0.0907', 'NDCG@10': '0.0546', 'HIT@20': '0.1214', 'NDCG@20': '0.0624'}

## Log Files and the trained models
We also provide log files and the trained models on these three datasets in the `src/output` directory.

## Acknowledgement

Some models are implemented based on [BASRec](https://github.com/KingGugu/BASRec) and [RecBole](https://github.com/RUCAIBox/RecBole).

Thanks for providing efficient implementation.
