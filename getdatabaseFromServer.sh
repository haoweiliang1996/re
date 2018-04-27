server=$SIXNIGHT
#dataset_id=4
#dataset_part=cloth${dataset_id}
#echo $dataset_id
#echo $dataset_part
#hist,checkpoint,database,ssd1,ssd for color,lst file,
#checkpoint 除了上装都在68上
#ssd 都在69
#scp -r lhw@\[$SIXNIGHT\]:~/cloth/mxnet-classify/retrieval/clothall/$dataset_part/densenet201/densenet201_1920/cropus.npy ./retrieval/cropus/database/$dataset_id/cropus1920.npy
#scp -r lhw@\[$SIXNIGHT\]:/home/lhw/cloth/mxnet-classify/data/hist/clothall/cloth4/cropus_hist.npy ./retrieval/cropus/hist/$dataset_id
#scp -r lhw@\[$SIXNIGHT\]:/home/lhw/cloth/mxnet-classify/checkpoint/clothall/cloth4/densenet201/net_best.params ./retrieval/checkpoint/$dataset_id
#scp -r lhw@\[$SIXNIGHT\]:/home/lhw/cloth/mxnet-ssd/model/cloth4_bak041823/ssd-symbol.json ./ssd/model/maincolor/
#scp -r lhw@\[$SIXNIGHT\]:/home/lhw/cloth/mxnet-ssd/model/cloth4_bak041823/ssd-0225.params ./ssd/model/maincolor/ssd-0000.params

#scp -r lhw@\[$SIXNIGHT\]:/home/lhw/cloth/mxnet-ssd/model/cloth_t3/ssd-symbol.json ./ssd/model/type3/
#scp -r lhw@\[$SIXNIGHT\]:/home/lhw/cloth/mxnet-ssd/model/cloth_t3/ssd-0225.params ./ssd/model/type3/ssd-0000.params
#scp -r lhw@\[$SIXNIGHT\]:/home/lhw/cloth/mxnet-classify/mxsymbol/symbol_unittest.py ./retrieval/mxsymbol
#scp -r lhw@\[$SIXEIGHT\]:/home/lhw/cloth/mxnet-classify/checkpoint/clothall/cloth4c/densenet201/net_best.params ./retrieval/attr/checkpoint


for dataset_id in 5
do
dataset_part=cloth${dataset_id}
echo $dataset_part
#if[ $dataset_id = 5 ] ; then
model=resnet50_v2
#fi
#from=/home/lhw/cloth/mxnet-classify/retrieval/clothall/$dataset_part/$model/${model}_1920/cropus.npy
#to=./retrieval/cropus/database/$dataset_id/cropus1920.npy
#scp -r lhw@\[$SIXEIGHT\]:$from $to

### model ###
from=/home/lhw/cloth/mxnet-classify/checkpoint/clothall/$dataset_part/$model/net_fea.params
to=./retrieval/checkpoint/$dataset_id/net_best.params
#scp -r lhw@\[$SIXEIGHT\]:/home/lhw/cloth/mxnet-classify/checkpoint/clothall/$dataset_part/densenet201/net_best.params ./retrieval/checkpoint/$dataset_id/net_best.params
#md5 ./retrieval/checkpoint/$dataset_id/net_best.params
### hist ###
#from=/home/lhw/cloth/mxnet-classify/data/hist/clothall/$dataset_part/cropus_hist.npy
#to=./retrieval/cropus/hist/$dataset_id/cropus_hist.npy

### index ###
#from=/home/lhw/cloth/clothdata/clothindex/$dataset_id/cropus.lst
#to=./retrieval/cropus/index/$dataset_id/cropus.lst

### color classify ###
#from=/home/lhw/cloth/clothdata/clothindex/$dataset_id/cropus.lst
#to=./retrieval/cropus/index/$dataset_id/cropus.lst
ssh lhw@$SIXEIGHT md5sum $from
scp -r lhw@\[$SIXEIGHT\]:$from $to
md5 $to
done


##touch .gitkeep for data store folder
#hist,params,database