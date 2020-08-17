#!/bin/bash
function mvpng(){
	num=0
	for filename in `ls $1`
	do
		num=${filename%_*}
		num=${num#*c}
		cp $1"/"$filename $2"/"$num
		echo $filename
	done
}

originpath="./DukeMTMC-reID/query"
targetpath="./DukeMTMC-reID"
mvpng $originpath $targetpath
