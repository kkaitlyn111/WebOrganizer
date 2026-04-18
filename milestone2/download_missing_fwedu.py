# download using snapshot, not git lfs
# skips alrdy present files

import json, os
from huggingface_hub import snapshot_download

shards = json.load(open('data/sampled_shards.json'))['shards']
missing = [i for i in shards
           if not os.path.exists(f'../Corpus-200B/scores_fineweb-edu/CC_shard_{i:08d}_processed.npy')]
patterns = [f'scores_fineweb-edu/CC_shard_{i:08d}_processed.npy' for i in missing]
print(f'Downloading {len(patterns)} files...')
snapshot_download(
    'WebOrganizer/Corpus-200B',
    repo_type='dataset',
    local_dir='../Corpus-200B',
    allow_patterns=patterns,
)
print('done.')
