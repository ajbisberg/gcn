
#!bin/sh
# conda create --name gcn python=3.6
# conda install tensorflow=1.15 pandas scikit-learn
python setup.py install
cd gcn
# python train_lol.py --model gcn_cheby --dataset delta --seasons 2020
python cv_lol.py --model gcn_cheby --dataset delta --seasons 2020
# python feat_imp_lol.py --model gcn_cheby --dataset delta --seasons 2020
cd ..