# ProgressiveSearch

the details of this project is explained in my paper 逐层渐进式的高效神经网络结构搜索的研究和实现(Efficiently Layer-wise Progressive Neural Architecture Search)
https://github.com/984881878/ProgressiveSearch/blob/master/%E9%80%90%E5%B1%82%E6%B8%90%E8%BF%9B%E5%BC%8F%E7%9A%84%E9%AB%98%E6%95%88%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E6%90%9C%E7%B4%A2%E7%9A%84%E7%A0%94%E7%A9%B6%E5%92%8C%E5%AE%9E%E7%8E%B0.doc

requirement pytorch >= 1.0.


### search model architecture with the determined layer fixed(accuracy and loss weiht is respectively 0.8 0.2)

python cifar10_arch_search.py --share_classifier --train_batch_size 256 --test_batch_size 512 --pretrained_batch 256 --n_worker 16 --n_epochs 1 --pretrained_epochs 5 --fix_determined --determined_train --re_init "last_determined" --determined_train_epoch 5 --determined_train_batch 128 --depth 74 --alpha 48 --target_hardware "gpu1" --reg_loss_type "add" --reg_loss_acc 0.8 --reg_loss_latency 0.2


### search model architecture with the determined layer fixed(accuracy and loss weiht is respectively 0.9 0.1)

python cifar10_arch_search.py --share_classifier --train_batch_size 256 --test_batch_size 512 --pretrained_batch 256 --n_worker 16 --n_epochs 1 --pretrained_epochs 5 --fix_determined --determined_train --re_init "last_determined" --determined_train_epoch 5 --determined_train_batch 128 --depth 74 --alpha 48 --target_hardware "gpu1" --reg_loss_type "add" --reg_loss_acc 0.9 --reg_loss_latency 0.1

### search model architecture not fix determined layer(accuracy and loss weiht is respectively 0.8 0.2)

python cifar10_arch_search.py --share_classifier --train_batch_size 64 --test_batch_size 512 --pretrained_batch 256 --n_worker 16 --n_epochs 2 --pretrained_epochs 5 --determined_train --re_init "no_reinit" --determined_train_epoch 3 --determined_train_batch 128 --depth 56 --alpha 48 --target_hardware "gpu1" --reg_loss_type "add" --reg_loss_acc 0.8 --reg_loss_latency 0.2

### search model architecture not fix the determined layer(accuracy and loss weiht is respectively 0.9 0.1)

python cifar10_arch_search.py --share_classifier --train_batch_size 64 --test_batch_size 512 --pretrained_batch 256 --n_worker 16 --n_epochs 2 --pretrained_epochs 5 --determined_train --re_init "no_reinit" --determined_train_epoch 3 --determined_train_batch 128 --depth 56 --alpha 48 --target_hardware "gpu1" --reg_loss_type "add" --reg_loss_acc 0.9 --reg_loss_latency 0.1

### train found model with random init

python cifar10_run_exp.py --random_init --train --cutout --n_epoch 100 --path "your model architecture description document location" --train_batch_size 128 --test_batch_size 512

### train model with special init

python cifar10_run_exp.py --train --cutout --n_epoch 100 --path "your model architecture description and parameter chekpoint file location" --train_batch_size 256 --test_batch_size 512
