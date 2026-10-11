
#CALID='20260613_022300Z'
#CALID='20260616_023700Z'
#CALID='20260618_034800Z'
#CALID='20260622_032600Z'
#CALID='20260625_040200Z'
#CALID='20260629_045000Z'
#CALID='20260819_035400Z'
#CALID='20260822_031000Z'
#CALID='20260828_023500Z'
#CALID='20260904_033001Z'
#CALID='20260905_033001Z'
#CALID='20260914_030500Z'
#CALID='20260918_030500Z'
#CALID='20260919_051500Z'
#CALID='20260921_032501Z'
#CALID='20261004_040000Z'
#CALID='20261008_032001Z'
#CALID='20261009_031000Z'
CALID='20261010_042500Z'


export PYTHONPATH=/data/kylin/bin:/opt/python/miniconda3/envs/bursttda/bin
export PATH=/data/kylin/bin:/opt/python/miniconda3/envs/bursttda/bin:$PATH
#/opt/python/miniconda3/envs/bursttda/bin/python /data/kylin/bin/obs_planner.py cyga > /burstt11/disk1/at_script/test_planner.log

OBSDATE=`date +%y%m%d` # local time
DATEM3=`date -d "-3 days" +%y%m%d` # three days before
CMDDEL="rm -f /burstt1{6,5,2,1}/disk*/intensity/*_${DATEM3}_*"
if [ $HOSTNAME = burstt16 ]; then
	echo $CMDDEL > /burstt11/disk1/at_script/test_remove.log
	$CMDDEL
fi

## TauA (no need to change beams)
TRANSIT=`/opt/python/miniconda3/envs/bursttda/bin/python /data/kylin/bin/obs_planner.py crab | tail -1|awk -F ' ' '{print $5}'`
TBEGIN=`date -d "-90 minutes $TRANSIT" +%H:%M`
TEND=`date -d "90 minutes $TRANSIT" +%H:%M`
OBSDATE=`date -d $TEND +%y%m%d`
## auto_inten
if [ $HOSTNAME = burstt16 ]; then
	echo "cd /data/kylin/241212_new_bf256/rebin_inten256; /data/kylin/bin/auto_inten_rebin.sh crab $OBSDATE 0.95 1.25 7" | at $TEND
fi

## CygA (3hrs)
TRANSIT=`/opt/python/miniconda3/envs/bursttda/bin/python /data/kylin/bin/obs_planner.py cyga | tail -1|awk -F ' ' '{print $5}'`
TBEGIN=`date -d "-90 minutes $TRANSIT" +%H:%M`
TEND=`date -d "90 minutes $TRANSIT" +%H:%M`
OBSDATE=`date -d $TEND +%y%m%d`
## change only row10 from 13.54 to 16.32
echo "cd /home/ubuntu/rudp256_300m; ./write_2nd_matrix_256_amx_channel_log.py --auto --dcal /burstt12/disk12/2nd_cal/cal_${CALID}.check --scale 0.5 -m '14 15' --theta '-44.64 -37.51 -31.01 -24.93 -19.14 -13.54 -8.08 -2.68 2.68 8.08 16.32 19.14 24.93 31.01 37.51 44.64'" | at $TBEGIN
## restore default
echo "cd /home/ubuntu/rudp256_300m; ./write_2nd_matrix_256_amx_channel_log.py --auto --dcal /burstt12/disk12/2nd_cal/cal_${CALID}.check --scale 0.5 -m '14 15'" | at $TEND
## auto_inten
if [ $HOSTNAME = burstt16 ]; then
	echo "cd /data/kylin/241212_new_bf256/rebin_inten256; /data/kylin/bin/auto_inten_rebin.sh cyga $OBSDATE 0.95 1.95 10" | at $TEND
fi

## CasA (3hrs)
TRANSIT=`/opt/python/miniconda3/envs/bursttda/bin/python /data/kylin/bin/obs_planner.py casa | tail -1|awk -F ' ' '{print $5}'`
TBEGIN=`date -d "-90 minutes $TRANSIT" +%H:%M`
TEND=`date -d "90 minutes $TRANSIT" +%H:%M`
OBSDATE=`date -d $TEND +%y%m%d`
## change only row13 from 31.01 to 34.20
echo "cd /home/ubuntu/rudp256_300m; ./write_2nd_matrix_256_amx_channel_log.py --auto --dcal /burstt12/disk12/2nd_cal/cal_${CALID}.check --scale 0.5 -m '14 15' --theta '-44.64 -37.51 -31.01 -24.93 -19.14 -13.54 -8.08 -2.68 2.68 8.08 13.54 19.14 24.93 34.20 37.51 44.64'" | at $TBEGIN
## restore default
echo "cd /home/ubuntu/rudp256_300m; ./write_2nd_matrix_256_amx_channel_log.py --auto --dcal /burstt12/disk12/2nd_cal/cal_${CALID}.check --scale 0.5 -m '14 15'" | at $TEND
## auto_inten
if [ $HOSTNAME = burstt16 ]; then
	echo "cd /data/kylin/241212_new_bf256/rebin_inten256; /data/kylin/bin/auto_inten_rebin.sh casa $OBSDATE 0.95 1.55 13" | at $TEND
fi

#echo "cd /data/kylin/241212_new_bf256/rebin_inten256; /data/kylin/bin/auto_inten_rebin.sh casa 261005 0.95 1.55 13 > /burstt11/disk1/at_script/test_auto_inten.log 2>&1" | at 01:03


