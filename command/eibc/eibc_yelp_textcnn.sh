python src/train.py textcnn YELP outdir_textcnn_eibc+ibp_yelp_emb --only-emb -d 100 --pool max -T 20 --full-train-epochs 5 -c 0.0  --dropout-prob 0.2 -D "data_cached" --alpha 10.0 --unfreeze-wordvec --no-wordvec-layer --no-relu-wordvec --use_lrsch_in_fulltrain --test  --clip-grad-norm 0 --weight-decay 1e-4 -r 1e-3 && python src/train.py textcnn YELP outdir_textcnn_eibc+ibp_yelp --load-emb-dir outdir_textcnn_eibc+ibp_yelp_emb -d 200 --pool max -T 1 --full-train-epochs 0 -c 0.0  --dropout-prob 0.2 --epochs-per-save 0 -D "data_cached" --use_lrsch_in_fulltrain --test --clip-grad-norm 0 --load-emb --only-model