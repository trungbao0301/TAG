# Passive CyberRunner hardware recording analysis

- Start: 2026-07-24T18:27:26.890719+00:00
- End: 2026-07-24T18:37:26.890804+00:00
- Duration: 600.0 s

## Rates and jitter

- Camera: 26129 messages, 43.550 Hz, interval std 14.808 ms, p99 83.703 ms, estimated missing 9427
- Camera source stamps: 26129 messages, 43.550 Hz, interval std 14.800 ms, p99 83.592 ms, estimated missing 9582
- State: 35611 messages, 59.359 Hz, interval std 6.465 ms, p99 30.323 ms, estimated missing 328
- State with TCP subimage: 35613 messages, 59.369 Hz, interval std 6.508 ms, p99 30.373 ms, estimated missing 300
- Motor commands: 6152 messages, 10.341 Hz, interval std 1393.266 ms, p99 98.954 ms, estimated missing n/a (event-driven)

## Observations and ranges

- Camera metadata: `{'data_bytes': 768000, 'encoding': 'bgr8', 'frame_id': 'camera', 'height': 400, 'step_bytes': 1920, 'width': 640}`
- Missing ball observations: 9471 (26.596%), longest run 10.352 s
- alpha: [-0.516657, 0.542060] rad
- beta: [-0.518454, 0.647400] rad
- Latest camera receipt to estimate_subimg receipt age: 11.725 ms mean, 80.438 ms p99
- vel_1: [-180.000, 180.000], limit hits 3872 (62.939%)
- vel_2: [-180.000, 180.000], limit hits 3844 (62.484%)
- Command rate while active: 10.341 Hz; median event cadence: 28.220 Hz; whole-session average: 10.253 Hz; maximum silent gap: 98.964 s

## Passive command-to-angle fit

- alpha: delay 320.0 ms, gains [0.00000078, -0.00014906] rad/command, R2=0.085, dominant cmd_2, secondary/dominant ratio 0.005
- beta: delay 190.0 ms, gains [-0.00014692, -0.00000179] rad/command, R2=0.020, dominant cmd_1, secondary/dominant ratio 0.012

> Closed-loop passive estimate only. Commands are policy-correlated with state; gains and delays are not causal identification results.

## Episode statistics

- 72 passively inferred episodes; no official Dreamer episode-event topic was present.
- 30 inferred intervals lasted at least 1 second; 34 were shorter than the visibility grace threshold.

## Limitations

- No active motor excitation was performed.
- Delay, gain, and coupling estimates are preliminary closed-loop fits.
- Episode rows are inferred from ball visibility, not Dreamer event messages.
- Estimated missing transport messages use median cadence because the recorded messages have no sequence counter.
- The latest-camera receipt age can be overestimated when the best-effort camera subscription misses a frame.
