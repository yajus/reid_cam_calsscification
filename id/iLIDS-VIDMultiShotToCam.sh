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

originpath="./iLIDS-VID/i-LIDS-VID/sequences/cam2"
targetpath="/Users/zhengxiaopeng/Desktop/LSPRC比赛/camlabel/datacamgather/3"
mvpng $originpath $targetpath
