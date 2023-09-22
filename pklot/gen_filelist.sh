#!/bin/sh

#occupied_list=`find ./PKLot/PKLotSegmented/UFPR04/Rainy/ -type d -iname "*occupied*" -print`
#empty_list=`find ./PKLot/PKLotSegmented/UFPR04/Rainy/ -type d -iname "*empty*" -print`
#for i in $occupied_list; do
#	printf \\n$i;
#	find $i -type f -iname "*.jpg" -exec printf '{}, occupied\n' >> filelist.txt \;
#done
#for i in $empty_list; do
#	printf \\n$i;
#	find $i -type f -iname "*.jpg" -exec printf '{}, empty\n' >> filelist.txt \;
#done

#for state in Occupied Empty ; do
#        for jpg in ./PKlot/PKLotSegmented/UFPR04/Rainy/*/"$state"/*.jpg ; do
#                printf '%s, %s\n' "$jpg" "$state"
#        done
#done >> filelist.txt

for state in Occupied Empty ; do
        for jpg in ./PKLot/PKLotSegmented/UFPR04/Rainy/2013-01-17/"$state"/*.jpg ; do
                printf '%s,%s\n' "$jpg" "$state"
        done
done > filelist_train.txt
for state in Occupied Empty ; do
        for jpg in ./PKLot/PKLotSegmented/UFPR04/Rainy/2012-12-21/"$state"/*.jpg ; do
                printf '%s,%s\n' "$jpg" "$state"
        done
done > filelist_test.txt
