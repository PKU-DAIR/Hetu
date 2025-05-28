TP_VALUES=(16)
PP_VALUES=(1)

# 循环 TP 和 PP 的值
for TP in "${TP_VALUES[@]}"; do
  for PP in "${PP_VALUES[@]}"; do
    # 跳过 TP * PP 大于 16 的情况
    if (( TP * PP < 9 )); then
      continue
    fi
    if (( TP * PP > 31 )); then
      continue
    fi
    # 定义 EXP_FILE 的路径
    EXP_FILE="./experiments/scale/compute_only_13b/tp${TP}_pp${PP}.txt"
    # 调用现有脚本
    bash scripts/train_hetu_exp_13b_two_node.sh "$TP" "$PP" "$EXP_FILE"
  done
done
