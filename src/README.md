# How to run

Plase change the gpu the train_size, the dataset and the sparsity according to your preferences

## Bräm approach 

To run the code please use one of the following commands or see the run.sh file
To use CPU instead of GPU use --GPU=-1

```
python main.py --GPU=0 --batch_size=100 --exp_name=Bram_680Pat --lr=0.005 --num_epochs=100 --patience=60 --save_fig_freq=10 --save_freq=10 --dataset=Abbvie_Sim_Case1 --train_size=680 --sparsity=1
```

```
python main.py --GPU=0 --batch_size=100 --exp_name=Bram_280Pat --lr=0.005 --num_epochs=100 --patience=60 --save_fig_freq=10 --save_freq=10 --dataset=Abbvie_Sim_Case1 --train_size=280 --sparsity=1
```

```
python main.py --GPU=0 --batch_size=96 --exp_name=Bram_96Pat --lr=0.005 --num_epochs=100 --patience=60 --save_fig_freq=10 --save_freq=10 --dataset=Abbvie_Sim_Case1 --train_size=96 --sparsity=1
```

```
python main.py --GPU=0 --batch_size=48 --exp_name=Bram_48Pat --lr=0.005 --num_epochs=100 --patience=60 --save_fig_freq=10 --save_freq=10 --dataset=Abbvie_Sim_Case1 --train_size=48 --sparsity=1
```
