#!/bin/bash
beta1=0.0001
beta2=0.0001
device=1
gamma1=1.0
gamma2=1.0
opt="adamax"
lr=0.01
run_start=1
run_end=30
train=true
test_n_particles=32
for ((i = $run_start; i <= $run_end; i++))
do
    model_name="PIB.512x2.$opt.$lr.g1.$gamma1-g2.$gamma2-beta1.$beta1-beta2.$beta2-run.$i"
    echo $model_name
    if $train ; then
        CUDA_VISIBLE_DEVICES=$device python config.py --optimizer=$opt\
            --mode=train --train_mode=validate\
            --model_name=$model_name \
            --layer_sizes=784,512,512,10 \
            --pib_gammas=1.,$gamma1,$gamma2 --pib_betas=$beta1,$beta2 --n_particles=32\
            --learning_rate=$lr --l1_kernel_reg=0 --l2_kernel_reg=0 --l1_prior_reg=0 --l2_prior_reg=0
    fi
    CUDA_VISIBLE_DEVICES=$device python config.py --optimizer=$opt\
        --mode=test --test_mode=val --restore_prefix=val \
        --model_name=$model_name \
        --layer_sizes=784,512,512,10 \
        --pib_gammas=1.,$gamma1,$gamma2 --pib_betas=$beta1,$beta2 --test_n_particles=$test_n_particles \
        --learning_rate=$lr --l1_kernel_reg=0 --l2_kernel_reg=0 --l1_prior_reg=0 --l2_prior_reg=0
    
    CUDA_VISIBLE_DEVICES=$device python config.py --optimizer=$opt\
        --mode=test --test_mode=test --restore_prefix=val\
        --model_name=$model_name \
        --layer_sizes=784,512,512,10 \
        --pib_gammas=1.,$gamma1,$gamma2 --pib_betas=$beta1,$beta2 --test_n_particles=$test_n_particles \
        --learning_rate=$lr --l1_kernel_reg=0 --l2_kernel_reg=0 --l1_prior_reg=0 --l2_prior_reg=0
done
