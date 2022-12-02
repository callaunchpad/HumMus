from utils import pickle_load

import os, pickle
import numpy as np
from collections import Counter

data_dir = 'remi_dataset'
polyph_out_dir = 'remi_dataset/attr_cls/polyph'
rhythm_out_dir = 'remi_dataset/attr_cls/rhythm'
velocity_out_dir = 'remi_dataset/attr_cls/velocity'

# velocity_bounds: [3.1622776601683795, 4.268749491621899, 5.030462757595523, 5.674504383644443,
# 6.278430147824157, 6.947262972561018, 7.841197293781097, 16.218507946170636]
rhym_intensity_bounds = [0.2, 0.25, 0.32, 0.38, 0.44, 0.5, 0.63]
polyphonicity_bounds = [2.63, 3.06, 3.50, 4.00, 4.63, 5.44, 6.44]
velocity_bounds = [3.16, 4.26, 5.033, 5.67, 6.27, 6.94, 7.84]

def compute_velocity_variance(events, n_bars):
  velocity_record = np.zeros((n_bars, ))
  velocities = []
  cur_bar = -1

  for ev in events:
    if ev['name'] == 'Bar':
      if cur_bar != -1:
        if len(velocities) > 1:
          velocity_record[cur_bar] = np.std(np.array(velocities))
        else:
          velocity_record[cur_bar] = 0
      velocities = []
      cur_bar += 1
    if ev['name'] == 'Note_Velocity':
      velocities.append(int(ev['value']))
  if len(velocities) > 1:
    velocity_record[cur_bar] = np.std(np.array(velocities))
  else:
    velocity_record[cur_bar] = 0

  return velocity_record
  
def compute_velocity_bounds(all_velocities):
  velocity_bounds = []
  all_velocities = np.array(all_velocities)
  for i in range(8):
    percentile = (i + 1) * 12.5
    velocity_bounds.append(np.percentile(all_velocities, percentile))
  return velocity_bounds

def compute_polyphonicity(events, n_bars):
  poly_record = np.zeros( (n_bars * 16,) )

  cur_bar, cur_pos = -1, -1
  for ev in events:
    if ev['name'] == 'Bar':
      cur_bar += 1
    elif ev['name'] == 'Beat':
      cur_pos = int(ev['value'])
    elif ev['name'] == 'Note_Duration':
      duration = int(ev['value']) // 120
      st = cur_bar * 16 + cur_pos
      poly_record[st:st + duration] += 1
  
  return poly_record

def get_onsets_timing(events, n_bars):
  onset_record = np.zeros( (n_bars * 16,) )

  cur_bar, cur_pos = -1, -1
  for ev in events:
    if ev['name'] == 'Bar':
      cur_bar += 1
    elif ev['name'] == 'Beat':
      cur_pos = int(ev['value'])
    elif ev['name'] == 'Note_Pitch':
      rec_idx = cur_bar * 16 + cur_pos
      onset_record[ rec_idx ] = 1

  return onset_record

# poly, rhyth, velocity
def get_bins(file_name):
    # file = opened(file_name)

    bar_pos, events = pickle_load(os.path.join(data_dir, file_name))
    events = events[ :bar_pos[-1] ]

    polyph_raw = np.reshape(
      compute_polyphonicity(events, n_bars=len(bar_pos)), (-1, 16)
    )
    rhythm_raw = np.reshape(
      get_onsets_timing(events, n_bars=len(bar_pos)), (-1, 16)
    )
    velocity_raw = compute_velocity_variance(events, n_bars=len(bar_pos))

    polyph_cls = np.searchsorted(polyphonicity_bounds, np.mean(polyph_raw, axis=-1)).tolist()
    rfreq_cls = np.searchsorted(rhym_intensity_bounds, np.mean(rhythm_raw, axis=-1)).tolist()
    velocity_cls = np.searchsorted(velocity_bounds, velocity_raw).tolist()

    polyph_bin = avg_bin(polyph_cls)
    rfreq_bin = avg_bin(rfreq_cls)
    velocity_bin = avg_bin(velocity_cls)

    return polyph_bin, rfreq_bin, velocity_bin

# bins are zero-indexed
def avg_bin(cl):
    # [7, 9, 3]
    # sum -> half -> count
    all_cls = sum(cl)
    curr = all_cls // 2
    bin_num = -1
    while curr > 0:
        curr -= cl[bin_num]
        bin_num += 1
    return bin_num

if __name__ == "__main__":
  pieces = [p for p in sorted(os.listdir(data_dir)) if '.pkl' in p]
  all_r_cls = []
  all_p_cls = []
  all_v_cls = []

  if not os.path.exists(polyph_out_dir):
    os.makedirs(polyph_out_dir)
  if not os.path.exists(rhythm_out_dir):
    os.makedirs(rhythm_out_dir)  
  if not os.path.exists(velocity_out_dir):
    os.makedirs(velocity_out_dir)  

  # all_velocities = []
  # for p in pieces:
  #   bar_pos, events = pickle_load(os.path.join(data_dir, p))
  #   all_velocities.append(compute_velocity_variance(events, n_bars=len(bar_pos)))

  # flattened_arr = []
  # for arr in all_velocities:
  #   for n in arr:
  #     flattened_arr.append(n)
  #   
  # velocity_bounds = compute_velocity_bounds(flattened_arr)
  # print("velocity_bounds:", velocity_bounds)

  for p in pieces:
    bar_pos, events = pickle_load(os.path.join(data_dir, p))
    events = events[ :bar_pos[-1] ]

    polyph_raw = np.reshape(
      compute_polyphonicity(events, n_bars=len(bar_pos)), (-1, 16)
    )
    rhythm_raw = np.reshape(
      get_onsets_timing(events, n_bars=len(bar_pos)), (-1, 16)
    )
    velocity_raw = compute_velocity_variance(events, n_bars=len(bar_pos))
    
    polyph_cls = np.searchsorted(polyphonicity_bounds, np.mean(polyph_raw, axis=-1)).tolist()
    rfreq_cls = np.searchsorted(rhym_intensity_bounds, np.mean(rhythm_raw, axis=-1)).tolist()
    velocity_cls = np.searchsorted(velocity_bounds, velocity_raw).tolist()


    pickle.dump(polyph_cls, open(os.path.join(
      polyph_out_dir, p), 'wb'
    ))
    pickle.dump(rfreq_cls, open(os.path.join(
      rhythm_out_dir, p), 'wb'
    ))
    pickle.dump(velocity_cls, open(os.path.join(
      velocity_out_dir, p), 'wb'
    ))

    all_r_cls.extend(rfreq_cls)
    all_p_cls.extend(polyph_cls)
    all_v_cls.extend(velocity_cls)

  print ('[rhythm classes]', Counter(all_r_cls))
  print ('[polyph classes]', Counter(all_p_cls))
  print ('[velocity classes]', Counter(all_v_cls))