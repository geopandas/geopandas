#!/bin/sh
set -e

if [ -z "${PGUSER}" ] || [ -z "${PGPORT}" ]; then
    echo "Environment variables PGUSER and PGPORT must be set"
    exit 1
fi

PGDATA=$(mktemp -d /tmp/postgres.XXXXXX)
echo "Setting up PostgreSQL in ${PGDATA} on port ${PGPORT}"

pg_ctl -D ${PGDATA} initdb
pg_ctl -D ${PGDATA} start

SOCKETPATH="/tmp/.s.PGSQL.${PGPORT}"
echo -n 'waiting for postgres'
while [ ! -e ${SOCKETPATH} ]; do
    sleep 1
    echo -n '.'
done
echo

echo "Done setting up PostgreSQL. When finished, stop and cleanup using:"
echo
echo "    pg_ctl -D ${PGDATA} stop"
echo "    rm -rf ${PGDATA}"
echo

createuser -U ${USER} -s ${PGUSER}
createdb --owner=${PGUSER} test_geopandas
psql -d test_geopandas -q -c "CREATE EXTENSION postgis"

echo "PostGIS server ready."
