As part of making Prefect more generally available, we had the option of spinning it up on our own infrastructure, or using Prefect Cloud. Prefect Cloud ended up not working, due to the limited run retention when not on a negotiated enterprise agreement. As such, we stood up Prefect using Spin at NERSC. To aid the reproduction of this work, this directory contains the yaml definitions of each workload, of which we have:

* `prefect` - the Prefect Server instance
* `postgres` - the postgres database used by Prefect. Note that postgres has a persistent volume claim as well, `postgres-data.yaml`
* `dashboard` - the streamlit dashboard used to graph all the flow metrics.
* `prefect-hpc-worker` - the Superfacility-API compatible Prefect worker.