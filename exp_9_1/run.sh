#!/bin/bash
set -e

python3 ori-matrix-multiplication.py 2>&1 |tee result.log

awk '
  $1 ~ /^[0-9]+$/ && $1 < 10 {
    rows[$1] = ($6 < $5)
  }
  END {
    for(i = 0; i < 10; i++) {
      if (!(i in rows) || rows[i] == 0) exit 1
    }
    exit 0
  }
' result.log && echo "✅ Triton matmul-performance PASSED:90"

awk '
  $1 ~ /^[0-9]+$/ && $1 < 30 {
    rows[$1] = ($6 < $5)
  }
  END {
    for(i = 0; i < 30; i++) {
      if (!(i in rows) || rows[i] == 0) exit 1
    }
    exit 0
  }
' result.log && echo "✅ Triton matmul-performance PASSED:100"
