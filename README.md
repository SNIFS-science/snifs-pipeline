# SNIFS Pipeline

TODO:

1. Use Superfacilty API in prefect-hpc-worker to submit jobs
2. Create examples of submitting jobs from my local machine
3. Create a job which mimics a full reduction:
   1. Specify a run_id and have it preprocess all files, then do a shit continuum subtract, extract spectra, stitch red and blue arms together
4. Add return objects to the flows
5. To the dashboard add the ability to see diagnostic images
6. Refactor directory structure:
   1. filestore and sqlite should go into output directory
   2. each task should generate a specific directory for its run so we can see historical results (maybe prefixed by the time its easier to delete things)


Salient points:

* HPC orchestration (intra and interfacility dependencies)
* Lineage
* Flexible logging / metrics / etc