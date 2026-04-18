from huggingface_hub import snapshot_download
snapshot_download('WebOrganizer/Corpus-200B', repo_type='dataset', local_dir='.')