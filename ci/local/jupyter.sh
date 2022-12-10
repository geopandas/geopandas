source /home/vagrant/miniconda3/etc/profile.d/conda.sh
# install kernel into each env if required, skip base env
for e in `conda env list --json | jq --raw-output .envs[1:][]`
do 
    e=`basename $e`

    if [[ $( conda list -n $e -f ipython --json | jq --raw-output .[].name ) != "ipython" ]]
    then
        mamba install -n $e ipykernel
        conda activate $e
        ipython kernel install --user --name=$e
        conda deactivate
    fi
done
jupyter lab --ip='*' --IdentityProvider.token='' --IdentityProvider.password=''
