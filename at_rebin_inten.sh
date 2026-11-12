CMD=". /usr/share/Modules/init/sh; module add compiler; /opt/python/miniconda3/bin/python3 /data/kylin/beamform/intensity2disk256.py"
## original, did not work on burstt12 because of python path
#CMD=". /usr/share/Modules/init/sh; module add compiler; /data/kylin/beamform/intensity2disk256.py"

DUR=3600

## define disk_id for each server
echo $HOSTNAME
case "$HOSTNAME" in
    burstt16)
        ids=(0 4);;
    burstt15)
        ids=(5 1);;
    burstt12)
        ids=(8 4);;
    burstt11)
        ids=(3 6);;
esac


CMD0="$CMD 0 $DUR --mode 16 -d ${ids[0]} --cbin --tbin 8" 
CMD1="$CMD 1 $DUR --mode 16 -d ${ids[1]} --cbin --tbin 8" 


## cron run (23:xx): schedule all 24 hours of the next day
## manual run: schedule only the remaining hours before midnight,
## the cron job at 23:30 will take care of the next day
now_h=$((10#$(date +%H)))
if (( now_h >= 23 )); then
    start=0
else
    start=$((now_h + 1))
fi
echo $start

for ((i=$start; i<=23; i++)); do
        hh="${i}:00"
        echo $CMD0 | at $hh
        echo $CMD1 | at $hh
done

