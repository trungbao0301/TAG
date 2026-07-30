#!/usr/bin/env bash
# Live one-line-per-episode view of a training run.
#
# The raw log prints a "Slower than 35fps" line on every step (the env warns
# whenever step_dt exceeds 28.6 ms, and the measured median is 40 ms, so that
# is every step), which buries the episode results. This keeps the score, the
# rolling mean, and the termination reason.
#
#   ./tools/watch_training.sh                  # follow ~/tag_logs live
#   ./tools/watch_training.sh /tmp/other.log   # follow a specific log
set -u

LOG="${1:-/tmp/tag_train.log}"

if [ ! -f "$LOG" ]; then
  echo "no log at $LOG" >&2
  exit 1
fi

printf '%-7s %-9s %-9s %-7s %s\n' EPISODE SCORE MEAN20 STEPS WHY
printf '%s\n' '-------------------------------------------------------'

# -F so the view survives the log being rotated or the run being restarted.
tail -F -n +1 "$LOG" 2>/dev/null | awk '
  /^Previous reward:/            { score = $3 + 0 }
  # env_tcp prints episode length in seconds (steps/60); recover the step count.
  /^Previous episode length:/    { steps = int($4 * 60 + 0.5) }
  /^\[Done\]: TIMEOUT/           { why = "timeout" }
  /BALL LOST|ball lost/          { why = "hole" }
  /anti_cheat|ANTI-CHEAT/        { why = "anticheat" }
  /SUCCESS|reached goal/         { why = "GOAL" }

  /^Episodes:/ {
    ep = $2 + 0
    if (ep <= 1) next                      # the pre-training reset has no result

    n++
    hist[n] = score
    lo = (n > 20) ? n - 19 : 1
    sum = 0
    for (i = lo; i <= n; i++) sum += hist[i]

    printf "%-7d %+9.4f %+9.4f %-7d %s\n", ep, score, sum / (n - lo + 1), steps, \
           (why == "" ? "-" : why)
    fflush()
    why = ""
  }
'
