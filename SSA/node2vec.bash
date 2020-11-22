#!/bin/bash
set -e
## blue to echo
function blue(){
    echo -e "\033[35m$1\033[0m"
}
## green to echo
function green(){
    echo -e "\033[32m$1\033[0m"
}
## Error to warning with blink
function bred(){
    echo -e "\033[31m\033[01m\033[05m$1\033[0m"
}
## Error to warning with blink
function byellow(){
    echo -e "\033[33m\033[01m\033[05m$1\033[0m"
}
## Error
function red(){
    echo -e "\033[31m\033[01m$1\033[0m"
}
## warning
function yellow(){
    echo -e "\033[33m\033[01m$1\033[0m"
}
if [ ${USER} == "zero" ]; then
    echo "正在本地环境中运行，载入默认路径"
    # 本地
    node2vec='/home/zero/workspace/snap/examples/node2vec/node2vec'
    word2vec_standalone='/home/zero/workspace/wiki/node2vec/word2vec_standalone.py'
    WikiComputeSimilarity='/home/zero/workspace/wiki/wikisimilarity/WikiComputeSimilarity.py'
    fasttext='/home/zero/workspace/fastText/fasttext skipgram'
    fasttext_vec2word2vec_bin='/home/zero/workspace/wiki/node2vec/fasttext_vec2word2vec_bin.py'

    graph_node_path='/home/zero/workspace/wiki/Datas/WikiOutputParse2TrainCorpus/node2vecGraph_weight_SSA3'
    random_walk_path='/home/zero/workspace/wiki/Datas/WikiOutputParse2TrainCorpus/node2vec_random_walk'
    model_path='/home/zero/workspace/wiki/Datas/WikiOutputParse2TrainCorpus/node2vec_model'
elif [ ${USER} == "ailab" ]; then
    echo "服务器环境"
    # 服务器
    node2vec='/home/ailab/wiki/snap/examples/node2vec/node2vec'
    word2vec_standalone='/home/ailab/wiki/node2vec/word2vec_standalone.py'
    WikiComputeSimilarity='/home/ailab/wiki/wikisimilarity/WikiComputeSimilarity_forNorns.py'
    fasttext='/home/ailab/wiki/fastText/fasttext skipgram'
    fasttext_vec2word2vec_bin='/home/ailab/wiki/node2vec/fasttext_vec2word2vec_bin.py'

    graph_node_path='/home/ailab/wiki/Datas/WikiOutputParse2TrainCorpus/node2vecGraph_weight_SSA3'
    random_walk_path='/home/ailab/wiki/Datas/WikiOutputParse2TrainCorpus/node2vec_random_walk'
    model_path='/home/ailab/wiki/Datas/WikiOutputParse2TrainCorpus/node2vec_model'
else
    echo "未对当前环境配置"
    exit
fi
# 参数默认值
edgelist_fname="wiki-ssa123-OT-RmNoIdTitle-FinalId.edgelist"
l=80
r=10
p=4
q=0.5
w="-w" #为了方便测试，这里直接设置为固定值，发布时清空
dr="-dr" #为了方便测试，这里直接设置为固定值，发布时清空
# word2vec param
model="word2vec" # or fasttext
size=300
iter=5
min_count=400
window=5
sample=0.0001
hs=0
negative=5
alpha=0.025
cbow=0
save_model="0"  # [0, 1] 保存word2vec模型，可以重新训练
load_model=""  # 载入已经训练好的模型，继续进行训练
threads=10  # 开启多少个线程，不过由于Cython限制，最多能够占用10核
# for fasttext
lr=0.05  # learning rate [0.05]
lrUpdateRate=100  # change the rate of updates for the learning rate [100]
# compute similarity
# dataset="SemEval2017Task2En,RG65,WordSim353,WS353SIM,WS353REL"
dataset="All"

# 解析命令行传来的参数
TEMP=`getopt -o wl:r:p:q:s:m:i: --long dr,model:,input:,size:,iter,min_count:,window:,sample:,hs:,negative:,alpha:,cbow:,threads:,dataset:,lr:,lrUpdateRate:,save_model:,load_model:,cbow:,negative: \
     -n 'node2vec.bash' -- "$@"`
if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi
eval set -- "$TEMP"
while true ; do
    case "$1" in
        -l)
            l=$2 ; shift 2;;
        -r)
            r=$2 ; shift 2;;
        -p)
            p=$(printf "%.2f" $2) ; shift 2;;
        -q)
            q=$(printf "%.2f" $2) ; shift 2;;
        -w)
            w="-w"; shift ;;
        --dr)
            dr="-dr"; shift ;;
        --model)
            case "$2" in
                "word2vec"|"fasttext") model=$2; shift 2 ;;
                *) red "--model $2 must be 'word2vec' or fasttext"; exit;;
            esac ;;
        --input)
            edgelist_fname=${2}; shift 2;;
        -s|--size)
            size=${2}; shift 2;;
        -i|--iter)
            iter=${2}; shift 2;;
        -m|--min_count)
            min_count=${2}; shift 2;;
        --window)
            window=${2}; shift 2;;
        --sample)
            sample=${2}; shift 2;;
        --negative)
            negative=${2}; shift 2;;
        --hs)
            case "$2" in
                "0"|"1") hs=$2; shift 2 ;;
                *) red "--hs $2 must be 0 or 1"; exit;;
            esac ;;
        --cbow)
            case "$2" in
                "0"|"1") cbow=$2; shift 2 ;;
                *) red "--cbow $2 must be 0 or 1"; exit;;
            esac ;;
        --negative)
            negative=${2}; shift 2;;
        --alpha)
            alpha=${2}; shift 2;;
        --cbow)
            case "$2" in
                "0"|"1") cbow=$2; shift 2 ;;
                *) red "--cbow $2 must be 0 or 1"; exit;;
            esac ;;
        --lr)
            lr=${2}; shift 2;;
        --lrUpdateRate)
            lrUpdateRate=${2}; shift 2;;
        --save_model)
            case "$2" in
                "1"|"0") save_model=$2; shift 2 ;;
                *) red "--save_model $2 must be 0 or 1"; exit;;
            esac ;;
        --load_model)
            load_model="${2}"; shift 2;;
        --threads)
            threads=${2}; shift 2;;
        --dataset)
            case "$2" in
                "All") dataset="WordSim353,SemEval2017Task2En,RG65,MEN3000,MTURK771,RW2034,WS353SIM,WS353REL"; shift 2 ;;
                *)
                    dataset=""
                    i=1
                    while((1))
                    do
                        split=`echo $2|cut -d "," -f$i`
                        if [ "$split" != "" ]
                        then
                            ((i++))
                            if [ "$split" == "SemEval2017Task2En" ]; then
                                dataset="${dataset},${split}"
                            elif [ "$split" == "WordSim353" ]; then
                                dataset="${dataset},${split}"
                            elif [ "$split" == "RG65" ]; then
                                dataset="${dataset},${split}"
                            elif [ "$split" == "MEN3000" ]; then
                                dataset="${dataset},${split}"
                            elif [ "$split" == "MTURK771" ]; then
                                dataset="${dataset},${split}"
                            elif [ "$split" == "RW2034" ]; then
                                dataset="${dataset},${split}"
                            elif [ "$split" == "WS353SIM" ]; then
                                dataset="${dataset},${split}"
                            elif [ "$split" == "WS353REL" ]; then
                                dataset="${dataset},${split}"
                            fi
                        else
                            break
                        fi
                    done
                    shift 2
            esac ;;
        --) shift ; break ;;
        *) red "Internal error!" ; exit 1 ;;
    esac
done
random_walk_fname="${edgelist_fname}-l${l}-r${r}-p${p}-q${q}${w}${dr}.random_walk"
cbow_str=""
if [ "$cbow" == "1" ]; then
    cbow_str="_cbow1"
fi
model_fname="${random_walk_fname}-s${size}_al${alpha}_w${window}_mc${min_count}_sa${sample}_hs${hs}_it${iter}_ne${negative}${cbow_str}.bin"
# for fasttext
if [ "$model" == "fasttext" ]; then
    if [ "$save_model" == "1" ] || [ "$load_model" != "" ]; then
        red "fasttext dont supporte save model"
        exit
    fi
    if [ "$cbow" == "1" ]; then
        red "fasttext dont support cbow"
        exit
    fi
    fastext_model_fname="${random_walk_fname}-s${size}_al${alpha}_w${window}_mc${min_count}_sa${sample}_hs${hs}_it${iter}_ne${negative}-fasttext"
    model_fname="${fastext_model_fname}.bin"
fi

# 输出运行参数
echo "node2vec.bash"
echo "********* random walk parameters *********"
echo "--input:$(yellow $edgelist_fname)"
echo "OutFile random_walk_fname:$(yellow $random_walk_fname)"
echo "-l:$(yellow $l), -r:$(yellow $r), -p:$(yellow $p), -q:$(yellow $q), -w:$(yellow $w), -dr:$(yellow $dr)"
echo "********* word2vec parameters *********"
echo "Model: $(yellow $model)"
echo "OutFileVec_fname:$(yellow ${model_fname})"
if [ "$save_model" == "1" ]; then
    echo "OutFileModel_Fname: $(yellow ${model_fname}.model)"
fi
echo "--size(-s):$(yellow $size), --min_count(-m):$(yellow $min_count), --iter(-i):$(yellow $iter)"
echo "--window:$(yellow $window), --sample:$(yellow $sample), --hs:$(yellow $hs), --negative:$(yellow $negative)"
echo "--alpha:$(yellow $alpha), --cbow:$(yellow $cbow), --threads:$(yellow $threads), --save_model:$(yellow $save_model)"
if [ "$load_model" != "" ]; then
    echo "--load_model: $(yellow $load_model)"
fi
if [ "$model" == "fasttext" ]; then
    echo "--lr:$(yellow $lr), --lrUpdateRate:$(yellow $lrUpdateRate)"
fi
echo "********* compute similarity parameters *********"
echo "--dataset:$(yellow $dataset)"

read -p "Please check parameters[y/n]." var
if [ "$var" != "y" ]; then
    exit
fi

graph_node_file=${graph_node_path}/${edgelist_fname}

# 判断文件是否存在
if [ ! -f "$graph_node_file" ]; then
    red "${graph_node_file} 不存在"
    exit
fi

# 若需要载入预训练好的模型需要判断模型是否存在
if [ "$load_model" != "" ]; then
    if [ ! -f "${model_path}/${load_model}" ]; then
        red  "load_model:$load_model 不存在"
        exit
    fi
    load_model="-load_model ${model_path}/$load_model"
fi

# 生成random walk
echo "*********** generate random walk ***********"
if [ ! -f "${random_walk_path}/${random_walk_fname}" ]; then
    yellow "$ ${node2vec} -i:${graph_node_file} -o:${random_walk_path}/${random_walk_fname} -l:${l} -r:${r} -p:${p} -q:${q} -ow -v ${dr} ${w}"
    ${node2vec} -i:${graph_node_file} -o:${random_walk_path}/${random_walk_fname} -l:${l} -r:${r} -p:${p} -q:${q} -ow -v ${dr} ${w}
else
    yellow "randomw walk 已生成"
fi

# 训练词向量
echo "*********** train random walk ***********"
if [ ! -f "${model_path}/${model_fname}" ] || ([ "$save_model" == "1" ] && [ ! -f "${model_path}/${model_fname}.model" ]); then
    if [ "$model" == "word2vec" ]; then
        yellow "$ python ${word2vec_standalone} -train ${random_walk_path}/${random_walk_fname} -output ${model_path}/${model_fname} -window ${window} -size ${size} -sample ${sample} -hs ${hs} -negative ${negative} -iter ${iter} -min_count ${min_count} -alpha ${alpha} -cbow ${cbow} -threads ${threads} -binary 1 -save_model ${save_model} -load_model ${load_model}"
        python ${word2vec_standalone} \
        -train ${random_walk_path}/${random_walk_fname} \
        -output ${model_path}/${model_fname} \
        -window ${window} \
        -size ${size} \
        -sample ${sample} \
        -hs ${hs} \
        -negative ${negative} \
        -iter ${iter} \
        -min_count ${min_count} \
        -alpha ${alpha} \
        -cbow ${cbow} \
        -threads ${threads} -binary 1 -save_model ${save_model} ${load_model}
    elif [ "$model" == "fasttext" ]; then
        yellow "$ $fasttext -input ${random_walk_path}/${random_walk_fname} -output ${model_path}/${fastext_model_fname} -minCount $min_count -ws $window -dim $size -neg $negative -epoch $iter -thread $threads"
        $fasttext -input ${random_walk_path}/${random_walk_fname} -output ${model_path}/${fastext_model_fname} -minCount $min_count -ws $window -dim $size -neg $negative -epoch $iter -thread $threads
        yellow "$ python $fasttext_vec2word2vec_bin -input ${model_path}/${fastext_model_fname}.vec -output ${model_path}/${model_fname}"
        python $fasttext_vec2word2vec_bin -input "${model_path}/${fastext_model_fname}.vec" -output ${model_path}/${model_fname}
    fi
else
    yellow "model 已生成"
fi

# 计算相似度
echo "*********** compute similarity ***********"
yellow "$ python ${WikiComputeSimilarity} ${dataset} OnlyUseWordConceptFromNode2vec ${model_path}/${model_fname} --cache ../wikisimilarity/cache_hy_mc"
yes y | head -1 | python ${WikiComputeSimilarity} ${dataset} OnlyUseWordConceptFromNode2vec ${model_path}/${model_fname} --cache ../wiki/wikisimilarity/cache_hy_mc
