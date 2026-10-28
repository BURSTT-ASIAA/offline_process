CMD=". /usr/share/Modules/init/sh; module add compiler; /data/kylin/beamform/intensity2disk256.py"

DUR=3600

## define disk_id for each server
echo $HOSTNAME
case "$HOSTNAME" in
    burstt16)
        ids=(0 4);;
    burstt15)
        ids=(5 7);;
    burstt12)
        ids=(8 4);;
    burstt11)
        ids=(3 6);;
esac


CMD0="$CMD 0 $DUR --mode 16 -d ${ids[0]} --cbin --tbin 8" 
CMD1="$CMD 1 $DUR --mode 16 -d ${ids[1]} --cbin --tbin 8" 


for ((i=0; i<=23; i++)); do
        hh="${i}:00"
        echo $CMD0 | at $hh
        echo $CMD1 | at $hh
done

