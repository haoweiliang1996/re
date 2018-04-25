#########################################################################
# File Name: test.sh
# Author: Bill
# mail: XXXXXXX@qq.com
# Created Time: 2017-11-05 21:25:55
#########################################################################
#!/bin/bash
#pic=http://p4.yokacdn.com/pic/YOKA/2017-10-11/U10053P1TS1507720074_81975.jpg
#pic=http://p3.yokacdn.com/pic/YOKA/2017-10-11/U10053P1TS1507722543_82292.jpg
#pic=http://cdn.watoo11.com/wardrobe/201709/2017092315374136080.jpg
pic=https://gd1.alicdn.com/imgextra/i1/525467170/TB2N7T0h3MPMeJjy1XdXXasrXXa_!!525467170.jpg
pic=http://cdn.watoo11.com/wardrobe/201702/2017021920525869343.jpg?x-oss-process=image/resize,w_310
#pic=http://cdn.watoo11.com/wardrobe/201711/2017111520534216361.jpg
#pic=http://cdn.watoo11.com/wardrobe/201801/2018011112353382845.jpg?x-oss-process=image/resize,w_400
#pic=http://cdn.watoo11.com/wardrobe/201709/2017091623415804047.jpg?x-oss-process=image/resize,w_310
#pic=http://cdn.watoo11.com/wardrobe/201706/2017062910004980303.jpg?x-oss-process=image/resize,w_310
#pic=http://cdn.watoo11.com/wardrobe/201706/2017062910004980303.jpg?x-oss-process=image/resize,w_310
#pic=https://gd1.alicdn.com/imgextra/i2/525467170/TB2CarQhPuhSKJjSspaXXXFgFXa_!!525467170.jpg
##pic=https://img.alicdn.com/bao/uploaded/i2/2089135099/TB1NXWBaiERMeJjSspiXXbZLFXa_!!0-item_pic.jpg
#curl -v -X POST "http://47.104.25.10:80/1" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"class1\":1}"
#curl -v -X POST "http://127.0.0.1:5000/10003" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"class1\":1}"
#curl -v -X POST "http://47.104.25.10:80/2" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"class1\":1}"
#curl -v -X POST "http://47.104.25.10:80/2" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"imgId\":1,\"class1\":4}" 
#curl -v -X POST "http://47.104.25.10:80/2" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"imgId\":1,\"class1\":5}" 
#curl -v -X POST "http://47.104.25.10:80/2" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"imgId\":1,\"class1\":7}" 
#curl -v -X POST "http://47.104.25.10:80/2" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"imgId\":1,\"class1\":10}"


#curl -v -X POST "http://127.0.0.1:5000/10002" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"class1\":1}"
curl -v -X POST "http://127.0.0.1:5000/1" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"class1\":1}"
curl -v -X POST "http://127.0.0.1:5000/2" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"imgId\":1,\"class1\":4}"
#curl -v -X POST "http://127.0.0.1:5000/2"  -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"imgId\":1,\"class1\":5}"
#curl -v -X POST "http://127.0.0.1:5000/2" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"imgId\":1,\"class1\":7}"
#curl -v -X POST  "http://127.0.0.1:5000/2" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"imgId\":1,\"class1\":10}"
#curl -v -X POST "http://127.0.0.1:5000/10004" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"class1\":4,\"color_level\":0,\"style_level\":2}"
#curl -v -X POST "http://127.0.0.1:5000/10003" -H "Content-Type: application/json" --data-ascii "{\"url\":\"$pic\",\"class1\":4,\"color_level\":0,\"style_level\":2}"
