#!/bin/bash -e

echo "Setting up Postgresql"

mkdir -p ${HOME}/var
rm -rf ${HOME}/var/db

pg_ctl initdb -D ${HOME}/var/db
pg_ctl start -D ${HOME}/var/db
trap "stop_db; exit 0" HUP TERM TSTP
trap "stop_db; exit 130" INT

echo -n 'waiting for postgres'
while [ ! -e /tmp/.s.PGSQL.5432 ]; do
    sleep 1
    echo -n '.'
done

createuser
