#! /bin/bash
#########################################################################################
#				validation BLEU reporting				#
#			    Author: Mohammad Mahdi Mahsuli				#
#########################################################################################
# Parameters:										#
#########################################################################################
# model_dir: directory under OpenNMT-py root directory containing the models		#
# openNMT_py_dir: root director of OpenNMT-py						#
# src: source test file									#
# tgt: target test file									#
# multiref_tgt: prefix of multi-reference target test files				#
# out_file: prefix of output filename							#
# file_suffix: suffix of output to address extra parameters used for translation like	#
# 			    '-oracle'. Leave it empty when there is no extra params.	#
# translation_extra_params: extra translation parameters like '-length_model oracle'.	#
# 			    Leave it empty when there is no extra params.		#
# exp_dir: directory of experiments under OpenNMT-py root directory			#
# test_single_ref [true | false]: wheather or not test using single-reference		#
# test_multi_ref [true | false]: wheather or not test using multi-reference		#
#########################################################################################

model_dir='/home/mahdi/PhD/experiments/OpenNMT-py/model/iwslt-deen-30'
openNMT_py_dir='/home/mahdi/PhD/experiments/OpenNMT-py'
src='/home/mahdi/PhD/experiments/OpenNMT-py/data/iwslt14.tokenized.de-en/valid-30.de'
tgt='/home/mahdi/PhD/experiments/OpenNMT-py/data/iwslt14.tokenized.de-en/valid-30.en'
multiref_tgt=''

translation_extra_params='' #'-length_model lstm -length_model_loc length_model_iwslt-30_ratio_epoch_13.pt -length_penalty_a 35 -length_penalty_b 15'

exp_dir='experiments/valid-bleu-reporting' 
out_file='iwslt-30'
file_suffix=''
test_single_ref=true
test_multi_ref=false

cd $openNMT_py_dir

if [ ! -d $exp_dir/$out_file ]; then
    mkdir -p $exp_dir/$out_file
fi

if [ -f ${exp_dir}/$out_file/${out_file}-valid-single-bleu-stats.txt ] && [ $test_single_ref = true ]; then
    rm ${exp_dir}/$out_file/${out_file}-valid-single-bleu-stats.txt
fi

if [ -f ${exp_dir}/$out_file/${out_file}-valid-multi-bleu-stats.txt ] && [ $test_multi_ref = true ]; then
    rm ${exp_dir}/$out_file/${out_file}-valid-multi-bleu-stats.txt
fi


if [ $test_single_ref = true ] ; then
    echo -e 'Models directory: '${model_dir} >> ${exp_dir}/$out_file/${out_file}-valid-single-bleu-stats.txt
    echo -e '======================================================' >> ${exp_dir}/$out_file/${out_file}-valid-single-bleu-stats.txt
    echo -e 'Model\t\tBLEU\tdetails' >> ${exp_dir}/$out_file/${out_file}-valid-single-bleu-stats.txt
fi

if [ $test_multi_ref = true ] ; then
    echo -e 'Models directory: '${model_dir} >> ${exp_dir}/$out_file/${out_file}-valid-multi-bleu-stats.txt
    echo -e '======================================================' >> ${exp_dir}/$out_file/${out_file}-valid-multi-bleu-stats.txt
    echo -e 'Model\t\tBLEU\tdetails' >> ${exp_dir}/$out_file/${out_file}-valid-multi-bleu-stats.txt
fi

for model in $(ls ${model_dir}/*.pt | perl -pe 's%^(.*_)(\d+).pt$%\2 \1\2.pt%' | sort -n | perl -pe 's%^\d+ %%'); do
    model_name=$(echo $model | perl -pe 's%^(.*/)(.*).pt$%\2%')
    trans_file=${model_name}${file_suffix}   
    echo '======================================================'
    echo Using ${model_name} as model...
    echo '======================================================'

    python3 translate.py -src $src -tgt $tgt -replace_unk -verbose -gpu 0 -model ${model} -output ${exp_dir}/$out_file/pred-valid-${trans_file}.txt -max_length 200 $translation_extra_params
    #single-reference BLEU calculation
    if [ $test_single_ref = true ] ; then
        bleu_details=$(perl tools/multi-bleu.perl  $tgt < ${exp_dir}/$out_file/pred-valid-${trans_file}.txt)
        echo -e ${model_name}'\t'$(echo $bleu_details | grep -o -P '(?<=BLEU = )\d+.\d{2}(?=, )')'\t'$bleu_details >> ${exp_dir}/$out_file/${out_file}-valid-single-bleu-stats.txt
    fi

    #multi-reference BLEU calculation
    if [ $test_multi_ref = true ] ; then
        bleu_details=$(perl tools/multi-bleu.perl  $multiref_tgt < ${exp_dir}/$out_file/pred-${file_suffix}.txt)
        echo -e ${model_name}'\t'$(echo $bleu_details | grep -o -P '(?<=BLEU = )\d+.\d{2}(?=, )')'\t'$bleu_details >> ${exp_dir}/$out_file/${out_file}-valid-multi-bleu-stats.txt
    fi
    done
echo '==========================='
echo 'All Done!'
