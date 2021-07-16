CWD=$(pwd)
cd ~
mkdir huggingface/luhua/chinese_pretrain_mrc_macbert_large -p
cd huggingface/luhua/chinese_pretrain_mrc_macbert_large

wget https://huggingface.co/luhua/chinese_pretrain_mrc_macbert_large/resolve/main/config.json
wget https://huggingface.co/luhua/chinese_pretrain_mrc_macbert_large/resolve/main/pytorch_model.bin
wget https://huggingface.co/luhua/chinese_pretrain_mrc_macbert_large/resolve/main/tokenizer.json
wget https://huggingface.co/luhua/chinese_pretrain_mrc_macbert_large/resolve/main/tokenizer_config.json
wget https://huggingface.co/luhua/chinese_pretrain_mrc_macbert_large/resolve/main/vocab.txt

cd "${CWD}"
