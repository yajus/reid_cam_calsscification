#!/bin/bash
function mvpng(){
	num=0
	for filename in `ls $1`
	do
		for photoname in `ls $1"/"$filename`
		do
			pngdir=$1"/"$filename"/"$photoname
			echo $2"/"$num".png"
			cp $pngdir $2"/"$num".png"
			num=`expr $num + 1`
		done
	done
}

originpath="./prid_2011/multi_shot/cam_b"
targetpath="/Users/zhengxiaopeng/Desktop/LSPRC比赛/camlabel/datacamgather/1"
mvpng $originpath $targetpath
