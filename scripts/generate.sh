python ../generate.py \
--params \
../checkpoints/params.json \
--ckpt \
../checkpoints/model.pkl \
--length 15 \
--strategy Multinomial \
--tau 0.05 \
--gen_dir ../gen/ \
--gen_name gen.txt \
