python src/train.py cnn IMDB outdir_textcnn_ibp_imdb -d 100 --pool mean -T 60 --full-train-epochs 20 -c 1.0  --dropout-prob 0.2 --epochs-per-save 0 -D "data_cached" --test --clip-grad-norm 0.25 --vocab-size 0 --initial-cert-frac 0 --initial-cert-eps 0 --only-model