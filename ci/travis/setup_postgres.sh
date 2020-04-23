#!/bin/bash -e

echo "Setting up Postgresql"

mkdir -p ${HOME}/var
rm -rf ${HOME}/var/db

pg_ctl initdb -D ${HOME}/var/db
pg_ctl start -D ${HOME}/var/db

echo -n 'waiting for postgres'
while [ ! -e /tmp/.s.PGSQL.5432 ]; do
    sleep 1
    echo -n '.'
done

createuser -U travis -s postgres
createdb --owner=postgres test_geopandas
psql -d test_geopandas -q -c "CREATE EXTENSION postgis"

echo "Done setting up Postgresql"
