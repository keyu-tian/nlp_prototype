REL_PATH=../../

PYTHONPATH=${PYTHONPATH}:${REL_PATH} GLOG_vmodule=MemcachedClient=-1 \
spring.submit run --gpu -n1 \
--ntasks-per-node=1 \
--cpus-per-task=5 \
--job-name "test" "python -u -m test \"$1\""
