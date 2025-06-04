
import random

from helper import fusion_emotion

test = [
    ('hap', 1, 'positive', 1),
    ('ang', -1, 'negative', -1),
    ('sad', -0.5, 'neutral', 0),
    ('neu', 0, 'positive', 1),
    ('hap', 1, 'negative', -1)
]


for a_lab, a_v, t_lab, t_v in test:
    print(fusion_emotion(a_lab, a_v, t_lab, t_v))