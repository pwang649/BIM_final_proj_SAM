# # Run this line by line ONCE.

# # Authenticate with Hugging Face CLI
hf auth login

# # Then download the checkpoints (need to physically request access first)
# # https://huggingface.co/facebook/sam-3d-objects
cd sam-3d-objects
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
  
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download