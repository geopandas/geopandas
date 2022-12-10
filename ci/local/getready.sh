source /home/vagrant/miniconda3/etc/profile.d/conda.sh
conda info > /dev/null
for i in $(seq ${CONDA_SHLVL}); do     conda deactivate; done
conda info
conda activate base
conda activate go-dev
export CONDA=~/miniconda3/
export TRUST_CONDA=true
export PATH=$PATH:$HOME/bin
unset ADD_INCLUDE
unset ENV_YAML
export ADD_INCLUDE=true
export ENV_YAML="*.yaml"
conda info
