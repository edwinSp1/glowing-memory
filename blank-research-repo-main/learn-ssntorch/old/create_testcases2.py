import random
import torch
def has_line(line):
  return sum(line) / len(line) > 0.7 and all(x > 0.5 for x in line)


def create_line_testcases(N, num_samples):
  def create_tc(must_line=False):
    tc = [random.random() for i in range(N)]
    if must_line:
      tc = [random.random() + 0.5 for i in range(N)]
    label = [0, 0]
    line = has_line(tc)
    label[line] = 1
    return \
      (
        (torch.tensor(tc), torch.tensor(label, dtype=torch.float)),
        line
      )
  
  res = []
  amt = 0
  for _ in range(num_samples):
    tc, line = create_tc() 
    amt += line
    res.append(tc)
  
  while amt * 2 < len(res):
    tc, line = create_tc(must_line=True)
    res.append(tc)
    amt += line

  return res