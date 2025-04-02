NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 8 -x PYTHONPATH=/home/Hetu/python python test_moe_sam.py --k=1 --num_local_experts=4 --batch_size=4
