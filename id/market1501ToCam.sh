#!/bin/bash
function mvpng(){
	num=0
	for filename in `ls $1`
	do
		num=${filename%s*}
		num=${num#*c}
		cp $1"/"$filename $2"/"$num
		echo $filename
	done
}

originpath="./market1501/query"
targetpath="./market1501"
mvpng $originpath $targetpath
