cp $( vagrant ssh-config | grep Identity | cut -d ' ' -f4 ) ~/.ssh/vagrant_key
sed '/^gpd\.local/d' ~/.ssh/known_hosts > ~/.ssh/known_hosts
scp ~/.ssh/vagrant_key rraymond@macbook-2017.local:.ssh/.